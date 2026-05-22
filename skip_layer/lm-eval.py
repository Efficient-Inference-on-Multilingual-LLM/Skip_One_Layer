import json
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import login
import os

from dotenv import load_dotenv
load_dotenv()

login(token=os.getenv("HF_TOKEN"))

os.environ["HF_ALLOW_CODE_EVAL"] = "1" # needed for humaneval tasks

BACKEND = "vllm"  # "hf" or "vllm"
GPU_MEMORY_UTILIZATION = 0.8  # vLLM only
DEVICE = "cuda:0"
BATCH_SIZE = "auto"

TASK_LIST = [
    # "longbench2_history_tasks", # too long token
    # "longbench2_incontext_tasks", # too long token
    # "longbench2_multi_tasks", # too long token
    # "longbench2_single_tasks", # too long token
    # "longbench2_structured_tasks", # too long token
    # "math_word_problems", # not supported
    # "toxigen", # ok
    # "bbq", # ok tapi agak lama karena datanya 700k
    # "humaneval_64_instruct", # ERROR
    # "logiqa", # not supported
    # "hellaswag", # ok
    "gsm8k",
    "ifeval", # generate_until task lama bgt
    "gpqa", # restricted access
    "bbh", # generate_until task lama bgt
    "humaneval_instruct",
    "squadv2", # genereate_until task lama bgt
    # "score_robustness_mmlu_pro", # generate_until task
    # "score_robustness_agieval", # generate_until task
    # "score_robustness_math", # generate_until task
    # "aime",
    # "anli", # ok
]
MODEL_NAMES = [
    # "google/gemma-3-1b-it",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
]

ERROR_LOG_PATH = Path("error_log.txt")

for MODEL_NAME in MODEL_NAMES:

    from transformers import AutoConfig as _AutoConfig
    NUM_LAYERS = _AutoConfig.from_pretrained(MODEL_NAME).num_hidden_layers

    for TASKS in TASK_LIST:
        config_dir = Path(f"eval-config/{MODEL_NAME}")
        result_dir = Path(f"eval-results/{MODEL_NAME}/{TASKS}")

        config_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)


        def log_eval_error(output_path, label, exc):
            """Write the failure details into the run's own output directory and
            append a one-line summary to the global error log, then keep going."""
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            err_path = output_path / "error.log"
            with open(err_path, "w") as f:
                f.write(f"Failed: {label}\n")
                f.write(traceback.format_exc())
            with open(ERROR_LOG_PATH, "a") as f:
                f.write(f"{label}: {exc}\n")
            print(f"[ERROR] {label} failed: {exc}\n        see {err_path}")


        def wait_for_free_gpu(min_free_fraction=0.9, timeout=120):
            """Poll until most of the GPU memory is free again.  CUDA releases
            memory asynchronously after a process is killed, so the next vLLM
            run can otherwise still see an occupied GPU and fail its startup
            memory check.  Gives up after `timeout` seconds and proceeds anyway."""
            deadline = time.time() + timeout
            while time.time() < deadline:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True,
                )
                free, total = (int(x) for x in out.stdout.splitlines()[0].split(","))
                if free >= min_free_fraction * total:
                    return
                time.sleep(2)


        def run_lm_eval(model_name, model_args, output_path):
            cmd = [
                "lm_eval",
                "--model",
                model_name,
                "--model_args",
                model_args,
                "--tasks",
                TASKS,
                "--batch_size",
                BATCH_SIZE,
                "--output_path",
                str(output_path),
                "--include_path",
                str(Path(__file__).parent),
                # required for code-eval tasks (e.g. humaneval) that run model output
                "--confirm_run_unsafe_code",
            ]
            if BACKEND == "hf":
                cmd += ["--device", DEVICE]

            print("Running:", " ".join(cmd))

            # Capture lm_eval's output into the run's own directory while still
            # echoing it to the console, so a failure can be inspected afterward.
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            log_path = output_path / "run.log"

            with open(log_path, "w") as log_file:
                # Run in its own process group so any orphaned vLLM children
                # can be reaped together once the run finishes.
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    start_new_session=True,
                )
                pgid = process.pid  # equals the new process group id
                for line in process.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)
                returncode = process.wait()

            # vLLM v1 runs its engine as a separate child process that can be
            # orphaned (e.g. when cleanup crashes with SIGABRT) and keep holding
            # GPU memory.  Kill the whole process group and wait for the memory
            # to be released so the next run starts with a clean GPU.
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            wait_for_free_gpu()

            # vLLM's background engine can crash with SIGABRT (-6) during cleanup
            # even after results are successfully written.  Treat that as success;
            # find_results_json will raise if the output is actually missing.
            if returncode not in (0, -6):
                raise subprocess.CalledProcessError(returncode, cmd)


        def find_results_json(output_dir):
            json_files = list(Path(output_dir).rglob("*.json"))

            if len(json_files) == 0:
                raise FileNotFoundError(f"No JSON file found in {output_dir}")

            return json_files[0]


        def json_to_csv(json_path, csv_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            rows = []

            results = data["results"]

            for task_name, metrics in results.items():
                row = {"task": task_name}

                for metric_name, value in metrics.items():
                    clean_name = metric_name.replace(",none", "")

                    if isinstance(value, (int, float)):
                        row[clean_name] = value

                rows.append(row)

            df = pd.DataFrame(rows)

            df.to_csv(csv_path, index=False)

            print(f"Saved CSV: {csv_path}")


        # =====================================================
        # Base model
        # =====================================================

        base_output_dir = result_dir / "base"

        try:
            run_lm_eval(
                model_name=BACKEND,
                model_args=f"pretrained={MODEL_NAME}" + (f",enforce_eager=True,gpu_memory_utilization={GPU_MEMORY_UTILIZATION}" if BACKEND == "vllm" else ""),
                output_path=base_output_dir,
            )

            base_json = find_results_json(base_output_dir)

            json_to_csv(
                base_json,
                result_dir / "base.csv",
            )
        except Exception as e:
            # Base run failed -> the whole task is unusable; skip to the next task.
            log_eval_error(base_output_dir, f"base ({TASKS})", e)
            continue

        # # delete json directory afterward
        # shutil.rmtree(base_output_dir)

        # =====================================================
        # Skip layers
        # =====================================================

        for i in range(NUM_LAYERS):
            print(f"\n===== Skipping layer {i} =====")

            steer_config = {
                f"layers.{i}": {
                    "action": "skip"
                },
            }

            config_path = config_dir / f"config_skip_layer_{i}.pt"

            torch.save(steer_config, config_path)

            output_dir = result_dir / f"skip_layer_{i}"

            try:
                run_lm_eval(
                    model_name="vllm_steered" if BACKEND == "vllm" else "steered",
                    model_args=(
                        f"pretrained={MODEL_NAME},"
                        f"steer_path={config_path}"
                        + (f",gpu_memory_utilization={GPU_MEMORY_UTILIZATION}" if BACKEND == "vllm" else "")
                    ),
                    output_path=output_dir,
                )

                result_json = find_results_json(output_dir)

                json_to_csv(
                    result_json,
                    result_dir / f"skip_layer_{i}.csv",
                )
            except Exception as e:
                # This layer failed -> log it and move on to the next layer.
                log_eval_error(output_dir, f"skip_layer_{i} ({TASKS})", e)
                continue
            # break  # only do the first layer for a quick test run

            # # remove json files afterward
            # shutil.rmtree(output_dir)

        print(f"Done for {TASKS} with model {MODEL_NAME}.")
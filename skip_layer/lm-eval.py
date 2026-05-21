import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import torch

MODEL_NAME = "google/gemma-3-1b-it"
BACKEND = "vllm"  # "hf" or "vllm"
DEVICE = "cuda:0"
BATCH_SIZE = "8"
NUM_LAYERS = 26

for TASKS in ["global_mmlu_full_id_humanities_tasks", "global_mmlu_full_ko_humanities_tasks", "global_mmlu_full_ja_humanities_tasks"]:
    config_dir = Path(f"eval-config/{MODEL_NAME}")
    result_dir = Path(f"eval-results/{MODEL_NAME}/{TASKS}")

    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)


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
        ]
        if BACKEND == "hf":
            cmd += ["--device", DEVICE]

        print("Running:", " ".join(cmd))

        subprocess.run(cmd, check=True)


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

    run_lm_eval(
        model_name=BACKEND,
        model_args=f"pretrained={MODEL_NAME}" + (",enforce_eager=True" if BACKEND == "vllm" else ""),
        output_path=base_output_dir,
    )

    base_json = find_results_json(base_output_dir)

    json_to_csv(
        base_json,
        result_dir / "base.csv",
    )

    # delete json directory afterward
    shutil.rmtree(base_output_dir)

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

        run_lm_eval(
            model_name="vllm_steered" if BACKEND == "vllm" else "steered",
            model_args=(
                f"pretrained={MODEL_NAME},"
                f"steer_path={config_path}"
            ),
            output_path=output_dir,
        )

        result_json = find_results_json(output_dir)

        json_to_csv(
            result_json,
            result_dir / f"skip_layer_{i}.csv",
        )

        # remove json files afterward
        shutil.rmtree(output_dir)

    print("Done.")
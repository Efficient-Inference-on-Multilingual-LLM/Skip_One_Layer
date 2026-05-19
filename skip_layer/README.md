# How to Run
1. Install LM Evaluation Harness library.
``` 
pip3 install lm_eval 
```
2. Locate `./<YOUR ENV>/lib/python3.12/site-packages/lm_eval/models/hf_steered.py` and change that file to the updated `hf_steered.py`.

3. Run `lm-eval.py`. Don't forget to change `MODEL_NAME`, `NUM_LAYERS` and `TASKS`.

4. Evaluate and visualize the results. Notebook Example: `neuroanatomy_lm_eval.ipynb`.

## Original Script (from the library documentation)

1. Normal Flow
```
    lm_eval --model hf \
    --model_args pretrained=<MODEL NAME> \
    --tasks <TASK NAME> \
    --device <DEVICE> \
    --batch_size <BATCH SIZE>
```

2. Define Steering Config (Run Using Python First)
    - ADD (adding some vectors to the original)
        ```python
            import torch

            steer_config = {
                "layers.3": {
                    "steering_vector": torch.randn(1, 768),
                    "steering_coefficient": 1,
                    "action": "add"
                },
            }
            torch.save(steer_config, "steer_config.pt")
        ```

    - CLAMP (replacing the original with some vectors)
        ```python
            import torch

            steer_config = {
                "layers.3": {
                    "steering_vector": torch.randn(1, 768),
                    "bias": torch.randn(1, 768),
                    "steering_coefficient": 1,
                    "action": "add"
                },
            }
            torch.save(steer_config, "steer_config.pt")
        ```

    - SKIP (not processing)
        ```python
            import torch

            steer_config = {
                "layers.3": {
                    "action": "skip"
                },
            }
            torch.save(steer_config, "steer_config.pt")
        ```

5. Steered Flow
```
    lm_eval --model steered \
    --model_args pretrained=<MODEL NAME>,steer_path=<STEER CONFIG> \
    --tasks <TASK NAME>\
    --device <DEVICE> \
    --batch_size <BATCH SIZE>
```
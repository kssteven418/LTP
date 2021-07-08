# Installation

We follow the same installation procedure as the original [huggingface transformer](https://github.com/huggingface/transformers) repo.
```
pip install sklearn scipy datasets torch
pip install -e .  # in the top directory
```

# Run Learnable Absolute Threshold
Prepare a checkpoint file that has been finetuned on the target downstream task. 
Add the following lines in the configuration file `{CKPT}/config.json`.
```
"prune_mode": "absolute_threshold",
"final_token_threshold": 0.01, 
"scoring_mode": "mean",
```

`final_token_threshold` determines the token threshold of the last layer, and the thresholds of the remaining layers will be linearly scaled.
For instance, the thresholds for the 3rd, 6th, and 9th layers will be 0.0025, 0.005, and 0.0075, respectively, when setting the `final_token_threshold` , i.e., the threshold for the last (12th) layer, to 0.01.
This number is a hyperparameter, and we found that 0.01 works well in many cases.

The learnable mode consists of 2 stages: soft threshold and hard threshold.
Please refer to our [paper](https://arxiv.org/abs/2107.00910) for more details.

## 1. Soft threshold
We first train the model using the soft threshold mode. 
This trains the thresholds as well as the model parameters to search for the best threshold configuration.

Run the following command:
```
python run.py --arch pibert-base --task {TASK} --restore {CKPT} --lr 2e-5 --temperature {T}\
  --lambda 0.15 --weight_decay 0 --bs 64 --masking_mode soft --epoch 1 --save_step 100 --no_load
```

Hyperparameter space: `lr 2e-5`, `bs 64`, `epoch 1` for smaller datasets (e.g., RTE, MRPC) and `epoch 10` for larger datasets (e.g., SST2, QNLI, MRPC), `weight_decay 0`.
`--no_load` flag will not load the best model at the end of the training (i.e., the final checkpoint will be the one at the end of training).
For `final_token_threshold` in the config file, `0.01` worked well.
For `lambda`, 0.01 ~ 0.2 worked well for SST2.
Set `temperature` to be around `1e-3 ~ 1e-5`.  

The final model will be checkpointed in `{CKPT_soft} = checkpoints/base/{TASK}/absolute_threshold/rate_{final_token_threshold}/temperature_{T}/lambda_{lambda}/lr_{lr}`.
Remove `trainer_state.json` from the checkpoint file in `{CKPT_soft}`.


## 2. Hard threshold
Once we learn the thresholds, we fix those values, turn back to the hard threshold mode, and finetune the model parameters only.

Run the following command:
```
python run.py --arch pibert-base --task {TASK} --restore {CKPT_soft} --lr {LR} --bs 64 --masking_mode hard --epoch 5 --save_step 500
```

Hyperparameter space: `lr = {0.5, 1, 2}e-5`, `bs 64`, `epoch 10` (with `save_step 500`) for small datasets, `epoch 5` (with `save_step 500`) for large datasets.
The final model will be checkpointed in `{CKPT_soft}/hard/lr_{LR}`.


# Run Top-k
Add the following lines in `{CKPT}/config.json`.
```
"prune_mode": "topk",
"token_keep_rate": 0.2,
```

`token_keep_rate` determines the keep rate of the last layer, and the keep rates of the remaining layers will be linearly scaled accordingly.
You can also assign negative number as keep rate for each layer will be assigned as `max(0, keep_rate)`.

Then, run the following command.

```
python run.py --arch pibert-base --task {TASK} --restore{CKPT} --lr 2e-5 --bs 64 --masking_mode hard --epoch 5 --save_step 500
```

Hyperparameter space: `lr = {0.5, 1, 2}e-5`, `bs 64`, `epoch 10` for small datasets, `epoch 5` (with `save_step 500`) for large datasets.
For `token_keep_rate` in the config file, 0.4 ~ -0.4 should work well for SST2 and MRPC.
The final model will be checkpointed in `{CKPT}/topk/lr_{LR}`.


# Run Non-leanrable (Baseline) Absolute Threshold
Add the following lines in `{CKPT}/config.json`.
```
"prune_mode": "absolute_threshold",
"final_token_threshold": 0.004, 
"scoring_mode": "mean",
```

`final_token_threshold` determines the token threshold of the last layer, and the thresholds of the remaining layers will be linearly scaled.


Run the following command:
```
python run.py --arch pibert-base --task {TASK} --restore {CKPT} --lr {LR} --bs 64 --masking_mode hard --epoch 5 --save_step 500
```

Hyperparameter space: `lr = {0.5, 1, 2}e-5`, `bs 64`, `epoch 10` (without `save_step 500`)for small datasets, `epoch 5` (with `save_step 500`) for large datasets.
For `final_token_threshold` in the config file, 0.006 ~ 0.016 should work well for SST2 and MRPC.
The final model will be checkpointed in `{CKPT}/hard/lr_{LR}`.



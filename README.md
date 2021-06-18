All the methods below require you to have a checkpoint file finetuned on a downstream task.

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

# Run Learnable Absolute Threshold
Add the following lines in `{CKPT}/config.json` (Same as non-learnable mode).
```
"prune_mode": "absolute_threshold",
"final_token_threshold": 0.01, 
"scoring_mode": "mean",
```

`final_token_threshold` determines the token threshold of the last layer, and the thresholds of the remaining layers will be linearly scaled.

The learnable mode consists of 2 stages

## 1. Soft threshold using sigmoid
In this stage, instead of masking the tokens, we apply soft masking on the tokens to be pruned by multiplying `sigmoid((score - threshold) / T)` at the end of each feed forward layers.

Run the following command:
```
python run.py --arch pibert-base --task {TASK} --restore {CKPT} --lr 2e-5 --temperature {T}\
  --lambda 0.15 --weight_decay 0 --bs 64 --masking_mode soft --epoch 1 --save_step 100 --no_load
```

Hyperparameter space: `lr = 2e-5`, `bs 64`, `epoch 1`, `weight_decay 0` (`lr`, `epoch` can be modified, for small dataset like MRPC, `lr 2e-5` and `epoch 10` can work). 
Note that `--no_load` flag will not load the best model at the end of the training (i.e., the final model will be the one at the end of training).
For `final_token_threshold` in the config file, `0.01` worked well.
For `lambda`, 0.01 ~ 0.2 worked well for SST2.
Set `temperature` to be around `1e-3 ~ 1e-5`.  
The final model will be checkpointed in `{CKPT_soft} = ibert_checkpoints/base/{TASK}/absolute_threshold/rate_{final_token_threshold}/temperature_{T}/lambda_{lambda}/lr_{lr}`.



## 2. Hard threshold
Because the model trained above does not mask tokens completely, we need another stage of finetuning after masking those tokens (i.e., assigning `-10000` to those tokens). 

Run the following command:
```
python run.py --arch pibert-base --task {TASK} --restore {CKPT_soft} --lr {LR} --bs 64 --masking_mode hard --epoch 5 --save_step 500
```

Hyperparameter space: `lr = {0.5, 1, 2}e-5`, `bs 64`, `epoch 10` (with `save_step 500`) for small datasets, `epoch 5` (with `save_step 500`) for large datasets.
The final model will be checkpointed in `{CKPT_soft}/hard/lr_{LR}`.

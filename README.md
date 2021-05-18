All the methods below require you to have a checkpoint file finetuned on a downstream task.

## Run Top-k
Add the following lines in `{CKPT}/config.json`.
```
"prune_mode": "topk",
"token_keep_rate": 0.2,
```

`token_keep_rate` determines the keep rate of the last layer, and the keep rates of the remaining layers will be linearly scaled accordingly.
You can also assign negative number as keep rate for each layer will be assigned as `max(0, keep_rate)`.

Then, run the following command.

```
# set task, lr, cuda, bs (batch size) accordingly
python run-pibert.py --arch pibert-base --task MRPC --restore {CKPT} --lr 1e-5 --cuda 1 --bs 64
```

This will checkpoint the final model to `ibert_checkpoints/base/{TASK}/topk/rate_{keep_token_rate}/lr_{LR}`


## Run Absolute Threshold
Add the following lines in `{CKPT}/config.json`.
```
"prune_mode": "absolute_threshold",
"final_token_threshold": 0.004, 
"scoring_mode": "mean",
```

`final_token_threshold` determines the token threshold of the last layer, and the thresholds of the remaining layers will be linearly scaled.

1. To run the baseline (non-learnable) threshold mode, use the following command:
```
# set task, lr, cuda, bs (batch size) accordingly
python run-pibert.py --arch pibert-base --task MRPC --restore {CKPT} --lr 1e-5 --cuda 3 --bs 64
```

2. To run the learnable threshold mode, use the following command:
```
# set task, lr, cuda, bs (batch size), lr_threshold, lambda accordingly
python run-pibert.py --arch pibert-base --task MRPC --restore {CKPT} --lr 1e-5 --lr_threshold 1e-5 --lambda 1e-5 --cuda 3 --bs 64
```

Note that if `lambda` is set as some value, the code will automatically run with the learnable mode. 
This parameter determines the number of tokens to be pruned (the higher the value, the more tokens to be pruned).
The learable mode requires you an additional parameter `lr_threshold`, which determines the learning rate for the thresholds.
Model parameters other than the thresholds will still be trained with `lr`.
This will checkpoint the final model to `ibert_checkpoints/base/MRPC/absolute_threshold/rate_0.004/lambda_{lambda}/lr_{lr}/tlr_{lr_threshold}`.

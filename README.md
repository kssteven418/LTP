# LTP: Learned Token Pruning for Transformers

![Screenshot from 2021-07-08 13-39-02](https://user-images.githubusercontent.com/50283958/124863552-376e7e80-dff2-11eb-8930-707664858adc.png)

**Check our [paper](https://arxiv.org/abs/2107.00910) for more details.**

# Installation
We follow the same installation procedure as the original [Huggingface transformer](https://github.com/huggingface/transformers) repo.
```
pip install sklearn scipy datasets torch
pip install -e .  # in the top directory
```

# Prepare Checkpoints
LTP is implemented on top of Huggingface transformer's [I-BERT implementation](https://github.com/huggingface/transformers/tree/master/src/transformers/models/ibert).
Therefore, we first need to generate a checkpoint file of ibert finetuned on the target downstream task.
While you can do this on the original Huggingface repository, 
we also support our [base branch](https://github.com/kssteven418/LTP/tree/ltp/base) `ltp/base` where you can run the following code to finetune `ibert` on the GLUE tasks.

```
git checkout ltp/base
cd examples/text-classification
python run_glue.py --model_name_or_path kssteven/ibert-roberta-base --output_dir {CKPT} --task {TASK} --do_train --do_eval {--some_more_arguments}
```

* `{TASK}`: RTE, MRPC, STSB, SST2, QNLI, QQP, MNLI
* Please refer to the [Huggingface tutorial](https://huggingface.co/transformers/v2.3.0/examples.html) and the [official documentation](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments) for more details in arguments and hyperparameters.
* Note that as default ibert behaves the same as roberta (see this [tutorial](https://huggingface.co/kssteven/ibert-roberta-base)), 
hence the resulting model will be the same as `roberta-base` finetuned on the target GLUE task.

The final model will be checkpointed in `{CKPT}`. 
* Remove `{CKPT}/trainer_state.json`.
* In the configuration file `{CKPT}/config.json`, change (1) `"architectures"` to `["LTPForSequenceClassification"]` and (2) `"model_type"` to `"ltp"`.


# Run Learned Token Pruning
Add the following lines in the configuration file `{CKPT}/config.json`.
```
"prune_mode": "absolute_threshold",
"final_token_threshold": 0.01, 
```

`final_token_threshold` determines the token threshold of the last layer, and the thresholds of the remaining layers will be linearly scaled.
For instance, the thresholds for the 3rd, 6th, and 9th layers will be 0.0025, 0.005, and 0.0075, respectively, when setting the `final_token_threshold` , i.e., the threshold for the last (12th) layer, to 0.01.
This number is a hyperparameter, and we found that 0.01 works well in many cases.

The learnable mode consists of 2 stages: soft threshold and hard threshold.
Please refer to our [paper](https://arxiv.org/abs/2107.00910) for more details.

## 1. Soft Threshold
We first train the model using the soft threshold mode. 
This trains the thresholds as well as the model parameters to search for the best threshold configuration.

Run the following command:
```
python run.py --arch ltp-base --task {TASK} --restore {CKPT} --lr 2e-5 --temperature {T}\
  --lambda 0.1 --weight_decay 0 --bs 64 --masking_mode soft --epoch {epoch} --save_step 100 --no_load
```

* `{TASK}`: RTE, MRPC, STSB, SST2, QNLI, QQP, MNLI
* You can assign different learning rate for `lr`, but 2e-5 worked fine.
* We set `{epoch}` to be 10 for smaller datasets (e.g., RTE, MRPC) and 1 for larger datasets (e.g., SST2, QNLI, MRPC).
* `--no_load` flag will not load the best model at the end of the training (i.e., the final checkpoint will be the one at the end of training).
* `lambda` is an important hyperparameter than controls the pruning level: the higher the value, the more we prune tokens. 0.01 ~ 0.2 worked well in many cases, but we recommend the user to empirically search for the best number for it.
* `temperature` is another hyperparameter, and 1e-3 ~ 1e-5 worked well. In the paper, we searched over {1e−4, 2e−4, 5e−4, 1e−3, 2e−3}.

The final model will be checkpointed in `{CKPT_soft} = checkpoints/base/{TASK}/absolute_threshold/rate_{final_token_threshold}/temperature_{T}/lambda_{lambda}/lr_{lr}`.
Remove `trainer_state.json` from the checkpoint file in `{CKPT_soft}`.


## 2. Hard Threshold
Once we learn the thresholds, we fix those values, turn back to the hard threshold mode, and finetune the model parameters only.

Run the following command:
```
python run.py --arch ltp-base --task {TASK} --restore {CKPT_soft} --lr {LR} --bs 64 --masking_mode hard --epoch 5 
```

* We used `{LR}` {0.5, 1, 2}e-5 in the paper.
* You can additionally set `--save_step 500` for more frequent evaluation/logging. The default setting will evaluate for every 1 epoch.

The final model will be checkpointed in `{CKPT_soft}/hard/lr_{LR}`.


# Run Baseline Methods
We additionally provide code to reproduce the baseline methods used in our paper (i.e., top-k and manual threshold).

## Top-k Token Pruning
Add the following lines in `{CKPT}/config.json`.
```
"prune_mode": "topk",
"token_keep_rate": 0.2,
```

The token keep rates of the first three layers and the last layer are 1 and `token_keep_rate`, respectively. 
The keep rates of the remaining layers are scaled linearly.
The smaller `token_keep_rate` is, the more aggressive we prune tokens.
You can also assign negative number for `token_keep_rate` and, in that case, the keep rate of each layer will be assigned as `max(0, keep_rate)`.

Run the following command:

```
python run.py --arch ltp-base --task {TASK} --restore {CKPT} --lr {LR} --bs 64 --masking_mode hard --epoch 5
```

* We used `{LR}` {0.5, 1, 2}e-5 in the paper.
* You can additionally set `--save_step 500` for more frequent evaluation/logging. The default setting will evaluate for every 1 epoch.


The final model will be checkpointed in `{CKPT}/topk/lr_{LR}`.


## Manual(Non-leanrable) Threshold Pruning
Add the following lines in `{CKPT}/config.json`.
```
"prune_mode": "absolute_threshold",
"final_token_threshold": 0.01, 
```

Run the following command:
```
python run.py --arch ltp-base --task {TASK} --restore {CKPT} --lr {LR} --bs 64 --masking_mode hard --epoch 5 --save_step 500
```

* We used `{LR}` {0.5, 1, 2}e-5 in the paper.
* You can additionally set `--save_step 500` for more frequent evaluation/logging. The default setting will evaluate for every 1 epoch.
* Note that the only difference from the learned token pruning mode is that we run the hard threshold mode from the beginning.


The final model will be checkpointed in `{CKPT}/hard/lr_{LR}`.



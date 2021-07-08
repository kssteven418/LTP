import os
import sys
import subprocess
import argparse
from time import gmtime, strftime
import json

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--arch', type=str, help='model architecture',
                        choices=[
			    'kssteven/ibert-roberta-base',
                            'kssteven/ibert-roberta-large', 
			    'roberta-base',
                            'roberta-large', 
                            'pibert-base',
                            'pibert-lase',]
                        )
    parser.add_argument('--task', type=str, help='finetuning task',
                        choices=[
			    'RTE', 'SST2', 'MNLI', 'QNLI', 'COLA',
                            'QQP', 'MRPC', 'STSB',]
                        )
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sparsity', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=None)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--from_bottom', action='store_true') #TODO merge this to symmetric mode
    parser.add_argument('--restore_file', type=str, default=None,
                        help='finetuning from the given checkpoint')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--prune_percentage', type=int, default=None,
                        help='Pruning percentage')
    parser.add_argument('--num_training_data', type=int, default=None)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--lambda_threshold', type=float, default=None)
    parser.add_argument('--weight_decay_threshold', type=float, default=None)
    parser.add_argument('--lr_threshold', type=float, default=None)
    parser.add_argument('--masking_mode', type=str, 
                        choices=['hard', 'soft', 'mixed'], default='hard') 
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--save_all', action='store_true') 
    parser.add_argument('--no_load_best', action='store_true') 

    args = parser.parse_args()
    return args

args = arg_parse()

with open(os.path.join(args.restore_file, "config.json")) as f:
    config = json.load(f)

prune_mode = config['prune_mode']
scoring_mode = None
if prune_mode == 'topk':
    rate = config['token_keep_rate']
elif prune_mode in ['absolute_threshold', 'rising_threshold']:
    rate = config['final_token_threshold']
    if prune_mode == 'absolute_threshold':
        scoring_mode = config['scoring_mode']
else:
    rate = None

if args.task is None:
    print('please specify --task')
    sys.exit()

if args.arch is None:
    print('please specify --arch')
    sys.exit()

DEFAULT_OUTPUT_DIR = 'checkpoints'

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
max_epochs = str(args.epoch)

task_specs = {
    'RTE' : {
        #'save_steps': '78',
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
    'MRPC' : {
        #'save_steps': '115',
        'lr': '2e-5',
        'metric': 'eval_combined_score',
    },
    'COLA' : {
        #'save_steps': '268',
        'metric': 'eval_matthews_correlation',
        'lr': '4e-5',
    },
    'STSB' : {
        #'save_steps': '180',
        'lr': '4e-5',
        'metric': 'eval_combined_score',
    },
    'SST2' : {
        #'save_steps': '2105',
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
    'QNLI' : {
        #'save_steps': '3274',
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
    'QQP' : {
        #'save_steps': '11371',
        'lr': '4e-5',
        'metric': 'eval_combined_score',
    },
    'MNLI' : {
        #'save_steps': '12272',
        'lr': '4e-5',
        'metric': 'eval_accuracy',
    },
}

model_path = args.arch
if args.restore_file is not None:
    model_path = args.restore_file

if args.eval and args.restore_file is None:
    print('Please specify --restore_file for the eval mode')
    sys.exit()
    
spec = task_specs[args.task]
lr = spec['lr']
is_large = ('large' in args.arch)
assert 'metric' in spec, 'please specify metric for %s' % args.task
metric = spec['metric']

# set learning rate
if args.lr:
    lr = str(args.lr)
    print('lr is set as %s' % lr)

#output_dir = args.output_dir
#if output_dir is None:
output_dir = DEFAULT_OUTPUT_DIR  + ('/large' if is_large else '/base')
if args.num_training_data is not None:
    task = '%s_num_data_%s' % (args.task, str(args.num_training_data))
else:
    task = args.task

#output_file = '%s/%s/tkr_%s/%s' % (args.task, prune_mode, rate, lr)
if args.output_dir is None:
    if prune_mode == 'topk':
        #output_file = '%s/%s/rate_%s/lr_%s' % (args.task, prune_mode, rate, lr)
        output_file = os.path.join(args.restore_file, f"topk/lr_{lr}")
        output_path = output_file
    else:
        assert prune_mode == 'absolute_threshold'
        if args.masking_mode == 'soft':
            _temperature = args.temperature if args.temperature is not None else 1e-3
            output_file = f"{args.task}/{prune_mode}/rate_{rate}/temperautre_{_temperature}/lambda_{args.lambda_threshold}/lr_{lr}"
            output_path = os.path.join(output_dir, output_file)
        elif args.masking_mode == 'hard':
            output_file = os.path.join(args.restore_file, f"hard/lr_{lr}")
            output_path = output_file
        else:
            raise NotImplementedError
        '''
        output_file = '%s/%s/rate_%s/lambda_%s/tlr_%s/lr_%s/%s' % \
                (args.task, prune_mode, rate, args.lambda_threshold, 
                 args.lr_threshold, lr, args.masking_mode)
        '''
else:
    output_file = '%s/%s/rate_%s/lambda_%s/%s/tlr_%s/lr_%s/%s' % \
            (args.task, prune_mode, rate, args.lambda_threshold, args.output_dir, 
             args.lr_threshold, lr, args.masking_mode)
    output_path = os.path.join(output_dir, output_file)

print('output path: ', output_path)


if args.sparsity:
    run_file = 'examples/text-classification/run_glue_sparsity_temp.py' #TODO change this
else:
    if 'pibert' in args.arch:
        run_file = 'examples/text-classification/run_glue_pibert.py'
    else:
        run_file = 'examples/text-classification/run_glue.py'

subprocess_args = [
    'python', run_file,
    '--model_name_or_path', model_path,
    '--task_name', args.task,
    '--do_eval',
    '--max_seq_length', '128',
    '--per_device_train_batch_size', str(args.bs),
    '--per_device_eval_batch_size', str(args.bs),
    '--masking_mode', args.masking_mode,
    ]

# Training mode
if not args.eval:
    if args.debug:
        output_path = 'tmp/temp'
        subprocess_args.append('--overwrite_output_dir')
        if args.eval_steps is not None:
            eval_steps = args.eval_steps

    subprocess_args.append('--do_train')
    if args.save_steps is None:
        subprocess_args += ['--evaluation_strategy', 'epoch'] 
        subprocess_args += ['--logging_strategy', 'epoch'] 
    else:
        subprocess_args += ['--evaluation_strategy', 'steps'] 
        subprocess_args += ['--eval_steps', str(args.save_steps)]
        subprocess_args += ['--logging_strategy', 'steps'] 
        subprocess_args += ['--logging_steps', str(args.save_steps)]
    subprocess_args += [
                   '--metric_for_best_model', metric,
                   '--learning_rate', lr, 
                   '--num_train_epochs', max_epochs, 
                   '--output_dir', output_path, 
                   ]
    if not args.no_load_best:
        subprocess_args += ['--load_best_model_at_end', 'True']
    if not args.save_all:
        subprocess_args += ['--save_total_limit', '3']

    if args.lr_threshold is not None:
        subprocess_args += ['--lr_threshold', str(args.lr_threshold)]

    if args.weight_decay_threshold is not None:
        subprocess_args += ['--weight_decay_threshold', str(args.weight_decay_threshold)]

    if args.lambda_threshold is not None:
        subprocess_args += ['--lambda_threshold', str(args.lambda_threshold)]

    if args.from_bottom:
        subprocess_args.append('--from_bottom')

    if args.num_training_data is not None:
        subprocess_args += ['--max_train_samples', str(args.num_training_data)]

    if args.temperature is not None:
        subprocess_args += ['--temperature', str(args.temperature)]


# Eval-only mode
else:
    subprocess_args += ['--output_dir', '/tmp/temp', '--overwrite_output_dir']

# Pruning percentage
if args.prune_percentage is not None:
    subprocess_args += ['--prune_percentage', str(args.prune_percentage)]


print(subprocess_args)
subprocess.call(subprocess_args)

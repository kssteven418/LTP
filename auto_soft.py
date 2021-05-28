#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs.
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.


import threading
import time
import os
import numpy as np
import sys


import gpustat
import logging

import itertools

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


exitFlag = 0
GPU_MEMORY_THRESHOLD = 2000  # MB?


def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        return_list = []
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            #if memory_used < GPU_MEMORY_THRESHOLD and i in [0, 1, 2, 3]:
            if memory_used < GPU_MEMORY_THRESHOLD and i in [0, 1, 2, 3, 4, 5 ,6, 7]:
                # return_list.append(i)
                # if len(return_list) == ??:
                    # return return_list
                return i

        logger.info("Waiting on GPUs")
        time.sleep(20)


class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
        logger.info("Starting " + self.name)
        # print_time(self.name, self.counter, 5)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):

            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(
                1, f"{i}th + {bash_command}", 1, cuda_device, bash_command)
            thread1.start()
            import time
            time.sleep(100)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device},{int(self.cuda_device+1)}'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device}'
        bash_command = self.bash_command

        logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        # ACTIVATE
        os.system(bash_command + '--cuda %s' % os.environ['CUDA_VISIBLE_DEVICES'])
        import time
        import random
        time.sleep(random.random() % 5)

        logger.info("Finishing " + self.name)


BASH_COMMAND_LIST = []

#TASK = 'QNLI'
#TASK = 'MRPC'
#TASK = 'SST2'
for TASK in ['MNLI']:
    for L in ['0.065', '0.06', '0.055']:
        for LR in ['2e-5']:
            CKPT = f'{TASK}_abs_0.01' 
            #CKPT = 'ibert_checkpoints/base/MRPC/absolute_threshold/mean/tkr_0.01/reg5e-8_2'
            #print(TASK, ROBERTA_ARCH)
            command = f"python run.py --arch pibert-base --task {TASK} --restore {CKPT} --lr {LR} --lambda {L} --weight_decay 0 --bs 64 --masking_mode soft --epoch 1 --no_load "
            BASH_COMMAND_LIST.append(command)

# Create new threads
dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST)

# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

time.sleep(5)

logger.info("Exiting Main Thread")

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from code import interact
import os
import torch
import shutil
import random
import logging
import argparse
import numpy as np

from datetime import timedelta

logger = logging.getLogger()

retriever_params_to_save = [
    'retriever_model_cfg',
    'pooling',
    'unshare',
    'no_title',
    'passage_length',
    'query_length'
]

generator_params_to_save = [
    'generator_model_cfg',
    'no_prefix',
    'use_answer',
    'passage_length',
    'query_length'
]

ranker_params_to_save = [
    'ranker_model_cfg',
    'passage_length',
    'query_length',
    'use_answer'
]

def add_encoder_params(parser: argparse.ArgumentParser):

    parser.add_argument("--retriever_model_cfg", default='xlm-roberta-base', type=str)
    parser.add_argument("--generator_model_cfg", default='google/mt5-base', type=str)
    parser.add_argument("--ranker_model_cfg", default='xlm-roberta-large', type=str)

    parser.add_argument("--query_length", type=int, default=128, help="Max length of the encoder input query")
    parser.add_argument("--passage_length", type=int, default=128, help="Max length of the encoder input passage")
    parser.add_argument("--pooling", default="cls", help="which pooling function to use")

    parser.add_argument("--mode", default='supervised', type=str, help="training mode")

    parser.add_argument("--unshare", action='store_true')
    parser.add_argument("--no_prefix", action='store_true')
    parser.add_argument("--use_answer", action='store_true')
    parser.add_argument("--no_title", action='store_true')

    parser.add_argument("--retriever_file", default=None, type=str,help="Saved retriever checkpoint file to initialize the model")
    parser.add_argument("--generator_file", default=None, type=str,help="Saved generator checkpoint file to initialize the model")
    parser.add_argument("--ranker_file", default=None, type=str,help="Saved ranker checkpoint file to initialize the model")

def add_cuda_params(parser: argparse.ArgumentParser):

    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--world_size", type=int, default=1, help="world_size for distributed training on gpus")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

def add_training_params(parser: argparse.ArgumentParser):

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization and dataset shuffling")

    parser.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default='(0.9, 0.999)', type=str, help="Betas for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")

    parser.add_argument("--warm_ratio", default=0.1, type=float, help="Linear warmup over warmup steps.")
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument("--print_per_step", default=100, type=int, help="print loss and other information")
    parser.add_argument("--eval_per_step", default=500, type=int, help="print loss and other information")
    parser.add_argument("--train_steps", default=10000, type=int, help="iteration numbers")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Amount of queries per batch for train set")
    parser.add_argument("--dev_batch_size", default=16, type=int, help="Amount of questions per batch for dev set")

    parser.add_argument("--train_num", default=2, type=int, help="amount of ctx per question")
    parser.add_argument("--dev_num", default=2, type=int, help="amount of ctx per question")

    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for generator")

    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)

    parser.add_argument("--over_write", action='store_true')
    parser.add_argument("--save_last", action='store_true')

    parser.add_argument("--add_positive", action='store_true')

    parser.add_argument("--iter_num", default=1, type=int)
    parser.add_argument("--local_num", default=1, type=int)

    parser.add_argument("--samples", default=0, type=int)

def add_common_params(parser: argparse.ArgumentParser):

    parser.add_argument("--global_loss_buf_sz", type=int, default=15000000, help='Increase this if you see errors like "encoded data exceeds max_size ..."')
    
    parser.add_argument("--output_dir", default=None, type=str, help="output directory")

    parser.add_argument("--restart", action='store_true', help="reset the optimizer and schedular states")
    parser.add_argument('--tao', default=1, type=float)

def add_retriever_params(parser: argparse.ArgumentParser):

    parser.add_argument('--shuffle_positive', action='store_true')

    parser.add_argument("--fix_query_encoder", action='store_true')
    parser.add_argument("--fix_ctx_encoder", action='store_true')

    parser.add_argument("--use_generator", action='store_true')
    parser.add_argument("--use_ranker", action='store_true')

    parser.add_argument("--alpha", default=1, type=float)

    parser.add_argument("--align_thresh", default=0.5, type=float)
    
    parser.add_argument('--no_sampling', action='store_true')

def add_generator_params(parser: argparse.ArgumentParser):
    pass

def add_ranker_params(parser: argparse.ArgumentParser):
    pass

def get_encoder_params_state(args, name):

    if name == 'retriever':
        return {param: getattr(args, param) for param in retriever_params_to_save}
    elif name == 'generator':
        return {param: getattr(args, param) for param in generator_params_to_save}
    else:
        return {param: getattr(args, param) for param in ranker_params_to_save}

def set_encoder_params_from_state(state, args, name):

    if name == 'retriever':
        params_to_save = retriever_params_to_save
    elif name == 'generator':
        params_to_save = generator_params_to_save
    else:
        params_to_save = ranker_params_to_save

    for k, v in state.items():
        if k in params_to_save:
            logger.info('Overriding parameter value. Param = %s, value = %s',  k, v)
            setattr(args, k, v)
    
    return args

def set_seed(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def setup_args_gpu(args):

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.local_rank == -1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=120))

    args.device = device
    world_size = os.environ.get('WORLD_SIZE')

    args.world_size = int(world_size) if world_size else 1

    logger.info(
        'Initialized as local_rank=%d world size=%d device=%s ', 
        args.local_rank, 
        args.world_size,
        device,
    )

def print_args(args):
    max_len = max([len(k) for k in vars(args).keys()]) + 4
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (max_len - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")

def set_env(args):

    setup_args_gpu(args)
    set_seed(args)

    if args.local_rank in [-1, 0]:
        if args.over_write and os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    
    if args.local_rank != -1:
        torch.distributed.barrier()

    for name in ['train.log', 'retriever.pt', 'generator.pt']:
        if os.path.exists(os.path.join(args.output_dir, name)):
            print("Path %s exists! Please set over_write as true" % args.output_dir)
            exit(0)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.local_rank != -1:
        torch.distributed.barrier()
    
    if args.iter_num > 1:
        args.restart = args.local_num == 1
    
    if args.query_length == 0:
        args.query_length = None
    
    if args.passage_length == 0:
        args.passage_length = None
    
    set_logger(args)
    print_args(args)

def set_logger(args, name='train.log'):  

    logger.handlers.clear()
    
    if args.local_rank in [-1, 0]:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    log_format = '[%(asctime)s] [Rank {} - %(levelname)s] [%(filename)s - %(lineno)d] %(message)s'.format(args.local_rank)
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    logger.addHandler(console)

    file = logging.FileHandler(os.path.join(args.output_dir, name), mode='a')
    file.setFormatter(log_format)
    logger.addHandler(file)

    

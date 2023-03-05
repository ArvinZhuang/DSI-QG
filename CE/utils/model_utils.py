#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import collections

from torch.serialization import default_restore_location

logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        'model_dict', 
        'optimizer_dict', 
        'scheduler_dict', 
        'encoder_params'
    ]
)

def setup_for_distributed_mode(model, optimizer, local_rank, fp16, fp16_opt_level='O2'):

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        if optimizer is None:
            model = amp.initialize(model, None, opt_level=fp16_opt_level)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    return model, optimizer

def get_model_obj(model):

    return model.module if hasattr(model, 'module') else model

def load_states_from_checkpoint(model_file):

    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, _: default_restore_location(s, 'cpu'))

    return CheckpointState(**state_dict)

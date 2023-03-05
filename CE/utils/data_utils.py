#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import pickle
import random
import logging

logger = logging.getLogger()


def read_data_from_json_files(paths, samples=0, merge=False):

    paths.sort()

    results = []
    for path in paths:

        logger.info('Reading file %s' % path)

        if path.endswith('json'):
            with open(path, 'r', encoding="utf-8") as fr:
                data = json.load(fr)
        elif path.endswith('pkl'):
            with open(path, 'rb') as fr:
                data = pickle.load(fr)
        else:
            raise NotImplementedError
        
        data = [d for d in data if len(d['positive_ctxs']) > 0]

        if samples > 0:
            data = data[:samples]
        
        results.append(data)

    logger.info('Aggregated data size: {}'.format(sum([len(result) for result in results])))

    if merge:
        new = []
        for r in results:
            new.extend(r)
        results = [new]

    return results


class MultiSetDataIterator(object):

    def __init__(
        self, 
        datasets, 
        seed = 0,
        offset = 0,
        shuffle = True,
        batch_size = 1, 
        local_rank = 0, 
        world_size = 1,
        total_steps = -1
    ):
        super().__init__()

        self.datasets = datasets
        self.shuffle = shuffle
        self.shuffle_seed = seed
        self.batch_size = batch_size
        self.development = total_steps == -1
        
        if self.development:
            nums = [math.ceil(len(data) / world_size) * world_size for data in datasets]
            total_steps = sum([math.ceil(num / batch_size) for num in nums])
        else:
            lengths = [len(data) for data in datasets]
            ratio = [length / sum(lengths) for length in lengths]

            steps = [round(r * total_steps) for r in ratio]
            steps[-1] = total_steps - sum(steps[:-1])

            nums = [batch_size * step * world_size for step in steps]

        self.iteration = []

        rand_shuffle = random.Random(seed)

        if self.development:
            for i, num, data in zip(range(len(nums)), nums, datasets):
                
                num_per_rank = int(num // world_size)

                ids = list(range(len(data)))

                if len(ids) < num:
                    ids = ids + random.sample(ids, num - len(ids))

                if world_size > 1:
                    per_rank = len(ids) // world_size
                    start_idx = local_rank * per_rank
                    end_idx = start_idx + per_rank
                else:
                    start_idx, end_idx, per_rank = 0, len(ids), len(ids)
                
                samples = [(i, _id) for _id in ids[start_idx:end_idx]]
                self.iteration.extend([samples[start: start + batch_size] for start in range(0, per_rank, batch_size)])
        else:
            for i, num, data in zip(range(len(nums)), nums, datasets):

                num_per_rank = int(num // world_size // batch_size)

                if world_size > 1:
                    per_rank = len(data) // world_size // batch_size * batch_size
                    start_idx = local_rank * per_rank
                    end_idx = start_idx + per_rank
                else:
                    per_rank = len(data) // batch_size * batch_size
                    start_idx, end_idx = 0, per_rank
                
                iteration = []
                while len(iteration) < num_per_rank:
                    ids = list(range(len(data)))
                    rand_shuffle.shuffle(ids)
                    
                    samples = [(i, _id) for _id in ids[start_idx:end_idx]]

                    iteration.extend([samples[start: start + batch_size] for start in range(0, per_rank, batch_size)])

                self.iteration.extend(iteration[:num_per_rank])
            
            rand_shuffle.shuffle(self.iteration)
            
        self.step = offset
        self.total_steps = total_steps
        
        self.local_samples = sum([len(batch) for batch in self.iteration])
        self.total_samples = sum(nums)

        logger.info('Local samples=%d total samples=%d', self.local_samples, self.total_samples)
    
    def iterate_data(self):

        for i in range(self.step, len(self.iteration)):
            items = [self.datasets[data_id][sample_id] for data_id, sample_id in self.iteration[i]]
            self.step = self.step + 1
            yield items
        
        self.step = 0

#!/usr/bin/env python3

import os
import glob
import torch
import random
import logging
import argparse

import torch.nn.functional as F

from utils.dist_utils import all_gather_list
from transformers import get_linear_schedule_with_warmup

from models.ranker import (
    get_ranker_components,
    sentence_to_inputs
)

from utils.data_utils import (
    MultiSetDataIterator, 
    read_data_from_json_files
)

from utils.model_utils import (
    get_model_obj, 
    setup_for_distributed_mode,
    load_states_from_checkpoint,
    CheckpointState
)

from utils.options import (
    set_env,
    add_cuda_params,
    add_common_params,
    add_encoder_params,
    add_training_params, 
    add_ranker_params,
    get_encoder_params_state, 
    set_encoder_params_from_state
)

logger = logging.getLogger()


class RankerTrainer(object):

    def __init__(self, args):
        self.args       = args
        self.world_size = args.world_size
        self.local_rank = args.local_rank

        if args.ranker_file and os.path.exists(args.ranker_file):
            state = load_states_from_checkpoint(args.ranker_file)
            set_encoder_params_from_state(state.encoder_params, self.args, 'ranker')
        else:
            state = None

        self.ranker, self.optimizer, self.tokenizer = get_ranker_components(args)

        self.train_iterator, args.train_steps  = self.get_iterator(args.train_file, args.train_batch_size, args.train_steps, 0)

        if args.dev_file is not None:
            self.dev_iterator = self.get_iterator(args.dev_file, args.dev_batch_size, -1, 0)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps = args.train_steps * args.iter_num // args.gradient_accumulation_steps * args.warm_ratio, 
            num_training_steps = args.train_steps * args.iter_num // args.gradient_accumulation_steps 
        )

        self.ranker, self.optimizer = setup_for_distributed_mode(self.ranker, self.optimizer, args.local_rank, args.fp16, args.fp16_opt_level)

        self._load_ranker(state, args.restart)
    
    def get_iterator(self, file, batch_size, total_steps, start_step=0):

        args = self.args

        data_files = glob.glob(file)
        data = read_data_from_json_files(data_files, args.samples)

        if min([len(d) for d in data]) < self.world_size:
            return None
        
        if total_steps < 0:
            num_per_batch = args.world_size * batch_size
            per_epoch = [len(d) // num_per_batch for d in data]

            total_steps = - total_steps * sum(per_epoch)

        iterator = MultiSetDataIterator(
            data,
            offset=start_step,
            shuffle=True,
            seed=args.seed,
            batch_size=batch_size, 
            local_rank=self.local_rank, 
            world_size=self.world_size,
            total_steps=total_steps
        )

        return iterator, total_steps
    
    def create_input(self, samples, nums, shuffle=True):

        args = self.args

        queries, contexts, answers = [], [], []
        
        for sample in samples:

            ctxs = sample['hard_negative_ctxs']

            if len(ctxs) == 0:
                continue

            if shuffle:
                random.shuffle(ctxs)
            
            if len(ctxs) < nums - 1:
                ctxs = ctxs * (nums - 1)
            
            ctxs = [sample['positive_ctxs'][0]] + ctxs

            for ctx in ctxs[:nums]:
                queries.append(sample['question'])
                answers.append(sample['answers'][0] if 'answers' in sample else None)
                contexts.append(ctx)
            
        inputs = sentence_to_inputs(queries, contexts, answers, self.tokenizer, args)

        return inputs['input_ids'], inputs['attention_mask']

    def _forward(self, samples, num, shuffle=True):

        input_ids, attention_mask = self.create_input(samples, num, shuffle)
        outs = self.ranker(input_ids, attention_mask)

        scores = outs['scores'].view(-1, num)

        loss = F.cross_entropy(scores, torch.zeros(scores.size(0), device=self.args.device).long())

        return loss

    def run_train(self):
        args = self.args

        if not args.save_last:
            best_loss = self.validate()
            self._save_best_ranker()

        logger.info("***** Genreator Training begin *****")
        logger.info("Total steps=%d", self.train_iterator.total_steps)

        train_loss = []

        for i, samples in enumerate(self.train_iterator.iterate_data()):

            self.ranker.train()

            loss = self._forward(samples, args.train_num)

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.ranker.parameters(), args.max_grad_norm)
                    
            if i % args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            global_loss = loss.item()
            if self.world_size > 1:
                stats = all_gather_list([global_loss], max_size=1000)
                global_loss = sum([item[0] for item in stats]) / self.world_size
            
            train_loss.append(global_loss)

            if (i + 1) % args.print_per_step == 0:
                loss = sum(train_loss) / len(train_loss)

                logger.info(
                    "Training ranker steps=%d/%d Last %d steps average loss=%.4f lr=%.2g", 
                    self.train_iterator.step, 
                    self.train_iterator.total_steps,
                    args.print_per_step, loss,
                    self.optimizer.param_groups[0]['lr']
                )

                train_loss.clear()
            
            if not args.save_last and (i + 1) % args.eval_per_step == 0:

                eval_loss = self.validate()

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self._save_best_ranker()

        if args.save_last:
            self._save_best_ranker()

        logger.info("***** ranker training end *****")
    
    @torch.no_grad()
    def validate(self):

        if not self.args.dev_file:
            return float('inf')

        args = self.args

        eval_loss = []

        self.ranker.eval()

        for samples in self.dev_iterator.iterate_data():

            loss = self._forward(samples, args.dev_num, False)

            loss = loss.item()
            if self.world_size > 1:
                stats = all_gather_list([loss], max_size=1000)
                loss = sum([item[0] for item in stats]) / self.world_size
            eval_loss.append(loss)

        eval_loss = sum(eval_loss) / len(eval_loss)
        logger.info(
            "Eval steps=%d/%d eval loss=%.4f", 
            self.train_iterator.step, 
            self.train_iterator.total_steps,
            eval_loss
        )

        return eval_loss
        
    def _save_best_ranker(self):

        if self.local_rank in [-1, 0]:
            path = os.path.join(self.args.output_dir, 'ranker.pt')

            model_to_save = get_model_obj(self.ranker)
            meta_params = get_encoder_params_state(self.args, 'ranker')

            best_state = CheckpointState(
                model_to_save.state_dict(),
                self.optimizer.state_dict(),
                self.scheduler.state_dict(),
                meta_params
            )

            torch.save(best_state._asdict(), path)
            logger.info("Save best state to %s", path)

    def _load_ranker(self, state, restart):

        if state:
            model_to_load = get_model_obj(self.ranker)
            logger.info('Loading saved model state ...')
            model_to_load.load_state_dict(state.model_dict)

            if not restart and state.optimizer_dict:
                logger.info('Loading saved optimizer state ...')
                self.optimizer.load_state_dict(state.optimizer_dict)

            if not restart and state.scheduler_dict:
                logger.info('Loading saved scheduler state ...')
                self.scheduler.load_state_dict(state.scheduler_dict)
        else:
            logger.info('Don\'t find the ranker checkpoint')

def main():
    parser = argparse.ArgumentParser()

    add_cuda_params(parser)
    add_encoder_params(parser)
    add_training_params(parser)

    add_common_params(parser)
    add_ranker_params(parser)
    
    args = parser.parse_args()
    
    set_env(args)

    trainer = RankerTrainer(args)

    try:
        trainer.run_train()
    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":

    logger = logging.getLogger()

    main()   

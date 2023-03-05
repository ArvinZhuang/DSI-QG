#!/usr/bin/env python3

import torch
import logging
import torch.nn as nn

from transformers import AdamW
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger()

def sentence_to_inputs(queries, ctxs, answers, tokenizer, args):

    if args.use_answer:
        if args.no_title:
            contexts = ['%s %s' % (answer, ctx['text']) for answer, ctx in zip(answers, ctxs)]
        else:
            contexts = ['%s %s %s' % (answer, ctx['title'], ctx['text']) for answer, ctx in zip(answers, ctxs)]
    else:
        if args.no_title:
            contexts = [ctx['text'] for ctx in ctxs]
        else:
            contexts = ['%s %s' % (ctx['title'], ctx['text']) for ctx in ctxs]

    contexts = ['%s%s%s' % (q, tokenizer.sep_token, c) for q, c in zip(queries, contexts)]

    inputs = tokenizer(
        contexts, 
        max_length=args.query_length + args.passage_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(args.device)

    return inputs

def get_ranker_components(args, inference=False):

    model = Ranker(args)
    tokenizer = AutoTokenizer.from_pretrained(args.ranker_model_cfg)

    if inference:
        optimizer = None
    else:
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_eps)

    model.to(args.device)

    return model, optimizer, tokenizer


class Ranker(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.backbone = AutoModel.from_pretrained(args.ranker_model_cfg)

        self.hidden_size = self.backbone.config.hidden_size

        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        
        outs = self.backbone(input_ids, attention_mask, output_attentions=True)
        pooling_output = outs["last_hidden_state"][:, 0, :]

        scores = self.fc(pooling_output).squeeze(1)
        outputs = dict(scores=scores, last_hidden_state=outs["last_hidden_state"], attentions=outs['attentions'])
            
        return outputs

import json
import random
import argparse
import os
import datasets

random.seed(313)


def read_query(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            qid, text = line.split('\t')
            dict[qid] = text.strip()
    return dict


def read_qrel(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, _ = line.split('\t')
            docid = int(docid)
            if docid not in dict:
                dict[docid] = [qid]
            else:
                dict[docid].append(qid)
    return dict


parser = argparse.ArgumentParser()
parser.add_argument("--train_num", type=int)
parser.add_argument("--eval_num", type=int, default=6980)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

print("Creating MS MARCO dataset...")

NUM_TRAIN = args.train_num
NUM_EVAL = args.eval_num
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

DSI_train_data = []
DSI_dev_data = []
corpus_data = []

data = datasets.load_dataset('Tevatron/msmarco-passage-corpus', cache_dir='cache')['train']
corpus = [item for item in data]
random.shuffle(corpus)
dev_query = read_query('msmarco_data/dev.query.tsv')
dev_qrel = read_qrel('msmarco_data/qrels.dev.small.tsv')
train_query = read_query('msmarco_data/train.query.tsv')
train_qrel = read_qrel('msmarco_data/qrels.train.tsv')

train_ids = list(train_qrel.keys())
random.shuffle(train_ids)
train_ids = train_ids[:NUM_TRAIN]
dev_ids = list(set(dev_qrel.keys()).difference(set(train_qrel.keys())))  # make sure no data leakage
random.shuffle(dev_ids)
dev_ids = dev_ids[:NUM_EVAL]

rand_ids = list(range(NUM_TRAIN + NUM_EVAL))
random.shuffle(rand_ids)


current_ind = 0
for docid in train_ids:
    passage = data[docid]['text']
    question = train_query[train_qrel[docid][0]]

    DSI_train_data.append({'text_id': rand_ids[current_ind], 'text': 'Passage: ' + passage})
    DSI_train_data.append({'text_id': rand_ids[current_ind], 'text': 'Question: ' + question})
    corpus_data.append(f"{rand_ids[current_ind]}\t{passage}")
    current_ind += 1

for item in corpus:
    if current_ind >= NUM_TRAIN:
        break
    passage = item['text']
    DSI_train_data.append({'text_id': rand_ids[current_ind],
                           "text": f"Passage: {passage}"})
    corpus_data.append(f"{rand_ids[current_ind]}\t{passage}")
    current_ind += 1

for docid in dev_ids:
    passage = data[docid]['text']
    question = dev_query[dev_qrel[docid][0]]

    if len(DSI_dev_data) < NUM_EVAL:
        DSI_train_data.append({'text_id': rand_ids[current_ind],
                               "text": f"Passage: {passage}"})
        DSI_dev_data.append({'text_id': rand_ids[current_ind],
                             "text": f"Question: {question}"})
        corpus_data.append(f"{rand_ids[current_ind]}\t{passage}")
        current_ind += 1

with open(f'{args.save_dir}/msmarco_DSI_train_data.json', 'w') as tf, \
        open(f'{args.save_dir}/msmarco_DSI_dev_data.json', 'w') as df:
    [tf.write(json.dumps(item) + '\n') for item in DSI_train_data]
    [df.write(json.dumps(item) + '\n') for item in DSI_dev_data]

with open(f'{args.save_dir}/msmarco_corpus.tsv', 'w') as f:
    [f.write(item + '\n') for item in corpus_data]
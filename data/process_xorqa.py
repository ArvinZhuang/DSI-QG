import json
import random
import argparse
import os

random.seed(313)
lang2mT5 = dict(
    ar='Arabic',
    bn='Bengali',
    en='English',
    fi='Finnish',
    ja='Japanese',
    ko='Korean',
    ru='Russian',
    te='Telugu'
)

parser = argparse.ArgumentParser()
parser.add_argument("--train_num", type=int)
parser.add_argument("--eval_num", type=int)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

print("Creating XORQA dataset...")

NUM_TRAIN = args.train_num
NUM_EVAL = args.eval_num
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

docTquery_train_data = []
docTquery_dev_data = []
DSI_train_data = []
DSI_dev_data = []
corpus_data = []

with open('xorqa_data/en_wiki.tsv', 'r') as f:
    en_wiki = [item for item in f]
random.shuffle(en_wiki)


rand_inds = list(range(NUM_TRAIN + NUM_EVAL))
random.shuffle(rand_inds)

current_ind = 0

qid_set = set()
with open('xorqa_data/gp_squad_train_data.json', 'r') as tf, \
        open('xorqa_data/gp_squad_dev_data.json', 'r') as df:
    train_data = json.load(tf)['data']
    random.shuffle(train_data)
    for item in train_data:
        title = item['title'].strip()
        title = title[6:title.find('_parentSection')]
        passage = item['paragraphs'][0]['context'].strip().replace('\n', ' ')

        question = item['paragraphs'][0]['qas'][0]['question'].strip()
        lang = item['paragraphs'][0]['qas'][0]['lang'].strip()
        qid = item['paragraphs'][0]['qas'][0]['id']

        if qid not in qid_set:  # remove duplicates
            docTquery_train_data.append({"text_id": question,
                                         "text": f"Generate {lang2mT5[lang]} question: {title}</s>{passage}"})

            if current_ind < NUM_TRAIN:
                DSI_train_data.append({'text_id': rand_inds[current_ind],
                                       "text": f"Passage: {title}</s>{passage}"})
                DSI_train_data.append({'text_id': rand_inds[current_ind],
                                       "text": f"Question: {question}"})
                corpus_data.append(f"{rand_inds[current_ind]}\t{passage}\t{title}")
                current_ind += 1

        qid_set.add(qid)

    for item in en_wiki:
        if current_ind >= NUM_TRAIN:
            break
        _, passage, title = item.split('\t')
        title = title.strip()
        passage = passage.replace('\n', ' ')
        DSI_train_data.append({'text_id': rand_inds[current_ind],
                               "text": f"Passage: {title}</s>{passage}"})
        corpus_data.append(f"{rand_inds[current_ind]}\t{passage}\t{title}")
        current_ind += 1

    dev_data = json.load(df)['data']
    random.shuffle(dev_data)
    for item in dev_data:
        title = item['title'].strip()
        title = title[6:title.find('_parentSection')]
        passage = item['paragraphs'][0]['context'].strip().replace('\n', ' ')
        question = item['paragraphs'][0]['qas'][0]['question'].strip()
        lang = item['paragraphs'][0]['qas'][0]['lang'].strip()
        qid = item['paragraphs'][0]['qas'][0]['id']

        if qid not in qid_set:  # remove duplicates
            docTquery_dev_data.append({"text_id": question,
                                       "text": f"Generate {lang2mT5[lang]} question: {title}</s>{passage}"})

            if len(DSI_dev_data) < NUM_EVAL:
                DSI_train_data.append({'text_id': rand_inds[current_ind],
                                       "text": f"Passage: {title}</s>{passage}"})
                DSI_dev_data.append({'text_id': rand_inds[current_ind],
                                     "text": f"Question: {question}"})
                corpus_data.append(f"{rand_inds[current_ind]}\t{passage}\t{title}")
                current_ind += 1
        qid_set.add(qid)

with open(f'{args.save_dir}/xorqa_docTquery_train_data.json', 'w') as tf, \
        open(f'{args.save_dir}/xorqa_docTquery_dev_data.json', 'w') as df:
    [tf.write(json.dumps(item) + '\n') for item in docTquery_train_data]
    [df.write(json.dumps(item) + '\n') for item in docTquery_dev_data]

with open(f'{args.save_dir}/xorqa_DSI_train_data.json', 'w') as tf, \
        open(f'{args.save_dir}/xorqa_DSI_dev_data.json', 'w') as df:
    [tf.write(json.dumps(item) + '\n') for item in DSI_train_data]
    [df.write(json.dumps(item) + '\n') for item in DSI_dev_data]

with open(f'{args.save_dir}/xorqa_corpus.tsv', 'w') as f:
    [f.write(item + '\n') for item in corpus_data]
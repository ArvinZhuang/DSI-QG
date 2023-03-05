# Train cross-encoder and rank generated queries.

> Note: this part of code is still in refactoring..


## Training data
download and unzip training data from https://drive.google.com/drive/folders/1JtHDWS6kW-pkzHZZ8P6F723pAe9CT1Tc
(xorqa_dpr_data_query=L_hard_negative=1.zip)

## Train re-ranker
```
outdir=runs/Ranker
model_cfg=xlm-roberta-large

python -m torch.distributed.launch --nproc_per_node=8 train_ranker.py --fp16 \
--max_grad_norm 2.0 --warm_ratio 0.1 --seed 42 --ranker_model_cfg ${model_cfg} \
--train_file xorqa_dpr_data_query=L_hard_negative=1/dpr_train_data.json --output_dir ${outdir} --train_steps 2000 \
--train_batch_size 4 --train_num 6 --learning_rate 1e-05 --print_per_step 100 \
--save_last
```
## Re-rank generated queries
```
for lang in ar bn fi ja ko ru te; do
    python -m torch.distributed.launch --nproc_per_node=8 re-rank.py --fp16 \
    --in_file ../data/xorqa_data/100k/xorqa_corpus.tsv.${lang}.q100.docTquery \
    --out_file ../data/xorqa_data/100k_ranked_test/${lang}.jsonl --ranker_file ${outdir}/ranker.pt
done
```

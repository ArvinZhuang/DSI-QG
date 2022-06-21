# XORQA experiments data
# Download gold passage data
#cd data
#wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/tydi_xor_gp.zip
#unzip tydi_xor_gp.zip
#mv tydi_xor_gp xorqa_data
#
## Download en wiki passage corpus
#wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/models/enwiki_20190201_w100.tsv -O xorqa_data/en_wiki.tsv
#
#python3 process_xorqa.py --train_num 98000 --eval_num 2000 --save_dir xorqa_data/100k
#cd ..

# MSMARCO experiments data
cd data
python3 process_marco.py --train_num 93020 --eval_num 6980 --save_dir msmarco_data/100k
cd ..
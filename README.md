# Provenance for Natural Language Claims
This is the code and the data for paper “Who said it, and Why?” Provenance for Natural Language Claims.

The experimental evaluation of the paper is to study the effective of the source extraction techniques proposed in the paper.

##Dataset:
We use MPQA 2.04 (Choi et al., 2005) as the corpus to train and test our models. It can be download by http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/.

Then you can run python process_mpqa.py to generate our training and test data for source extraction. We keep a copy of the output in the folder cross_validation.



To reproduce the results in the paper, you can simply run:

for i in {0..9}

do

	python extract_bert_etr_pair.py --bert_model bert-base-uncased --do_train --do_eval --do_lower_case --train_file cross_validation/$i/train_pairwise_positive_srl_replaced_full.json --predict_file cross_validation/$i/test_new_pairwise_both.json --train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 4.0 --max_seq_length 250  --output_dir cross_validation_result/$i/pairwise_positive_srl_replaced_full  --num_choice 2

done

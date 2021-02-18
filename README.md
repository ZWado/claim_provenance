# claim_provenance
This is the code and data for paper “Who said it, and Why?” Provenance for Natural Language Claims.

To reproduce the results in the paper, you can simply run:

for i in {0..9}

do

	python extract_bert_etr_pair.py --bert_model bert-base-uncased --do_train --do_eval --do_lower_case --train_file cross_validation/$i/train_pairwise_positive_srl_replaced_full.json --predict_file cross_validation/$i/test_new_pairwise_both.json --train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 4.0 --max_seq_length 250  --output_dir cross_validation_result/$i/pairwise_positive_srl_replaced_full  --num_choice 2

done

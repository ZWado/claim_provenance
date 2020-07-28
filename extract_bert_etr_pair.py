# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json
import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MarginRankingLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult", ["unique_id", "qid", "result", "label"])

class BertForSequenceClassificationPairwise(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationPairwise, self).__init__(config)
        self.hidden_size2 = int(0.5 * config.hidden_size)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.network = nn.Linear(config.hidden_size, self.hidden_size2)

        self.classifier = nn.Linear(self.hidden_size2, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels = None, input_ids2 = None, token_type_ids2 = None, attention_mask2 = None, labels2 = None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        relu_output = torch.tanh(self.network(pooled_output))#.clamp(min = 0)
        logits = self.classifier(relu_output)

        if labels is not None:
            _, pooled_output2 = self.bert(input_ids2, token_type_ids2, attention_mask2, output_all_encoded_layers = False)
            pooled_output2 = self.dropout(pooled_output2)
            relu_output2 = torch.tanh(self.network(pooled_output2))#.clamp(min = 0)

            logits2 = self.classifier(relu_output2)

            loss_fct = CrossEntropyLoss()
            pairwise_loss_fct = MarginRankingLoss(margin=1)

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + loss_fct(logits2.view(-1, self.num_labels), labels2.view(-1))
            pairwise_loss = pairwise_loss_fct(relu_output.float(), relu_output2.float(), labels.float().view(-1))
            return 0.1 * loss + 0.9 * pairwise_loss
        else:
            return logits

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, eid, guid1, text_a1, text_b1, label1 = None, \
                 guid2 = None, text_a2 = None, text_b2 = None, label2 = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.eid = eid
        self.guid1 = guid1
        self.text_a1 = text_a1
        self.text_b1 = text_b1
        self.label1 = label1
        self.guid2 = guid2
        self.text_a2 = text_a2
        self.text_b2 = text_b2
        self.label2 = label2

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, eid, qid1, input_ids1, input_mask1, segment_ids1, label_id1 = None, qid2 = None, \
                 input_ids2 = None, input_mask2 = None, segment_ids2 = None, label_id2=None):
        self.eid = eid
        self.qid1 = qid1
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.segment_ids1 = segment_ids1
        self.label_id1 = label_id1
        self.qid2 = qid2
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.label_id2 = label_id2

def read_etr_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    examples = []
    for (eid, entry) in enumerate(input_data):
        e_id = eid

        if is_training:
            qas_id = entry[0]['id']
            premise = entry[0]['premise']
            hypo = entry[0]['hypothesis']
            label = entry[0]['label']

            qas_id2 = entry[1]['id']
            premise2 = entry[1]['premise']
            hypo2 = entry[1]['hypothesis']
            label2 = entry[1]['label']

            example = InputExample(
                eid = e_id,
                guid1=qas_id,
                text_a1=premise,
                text_b1=hypo,
                label1=label,
                guid2=qas_id2,
                text_a2=premise2,
                text_b2=hypo2,
                label2=label2
            )
        else:
            qas_id = entry['id']
            premise = entry['premise']
            hypo = entry['hypothesis']
            label = entry['label']

            example = InputExample(
                eid = e_id,
                guid1=qas_id,
                text_a1=premise,
                text_b1=hypo,
                label1=label
            )

        examples.append(example)

    return examples

def convert_examples_to_features(examples, is_training, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        eid = example.eid
        qid1 = example.guid1

        if ex_index % 100 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a1 = tokenizer.tokenize(example.text_a1)
        tokens_b1 = None
        if example.text_b1:
            tokens_b1 = tokenizer.tokenize(example.text_b1)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a1, tokens_b1, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a1) > max_seq_length - 2:
                tokens_a1 = tokens_a1[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens1 = ["[CLS]"] + tokens_a1 + ["[SEP]"]
        segment_ids1 = [0] * len(tokens1)

        if tokens_b1:
            tokens1 += tokens_b1 + ["[SEP]"]
            segment_ids1 += [1] * (len(tokens_b1) + 1)

        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids1))
        input_ids1 += padding
        input_mask1 += padding
        segment_ids1 += padding

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        if output_mode == "classification":
            label_id1 = label_map[example.label1]
        elif output_mode == "regression":
            label_id1 = float(example.label1)
        else:
            raise KeyError(output_mode)

        if is_training:

            qid2 = example.guid2
            tokens_a2 = tokenizer.tokenize(example.text_a2)
            tokens_b2 = None
            if example.text_b2:
                tokens_b2 = tokenizer.tokenize(example.text_b2)
                _truncate_seq_pair(tokens_a2, tokens_b2, max_seq_length - 3)
            else:
                if len(tokens_a2) > max_seq_length - 2:
                    tokens_a2 = tokens_a2[:(max_seq_length - 2)]
            tokens2 = ["[CLS]"] + tokens_a2 + ["[SEP]"]
            segment_ids2 = [0] * len(tokens2)
            if tokens_b2:
                tokens2 += tokens_b2 + ["[SEP]"]
                segment_ids2 += [1] * (len(tokens_b2) + 1)
            input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
            input_mask2 = [1] * len(input_ids2)
            padding = [0] * (max_seq_length - len(input_ids2))
            input_ids2 += padding
            input_mask2 += padding
            segment_ids2 += padding

            assert len(input_ids2) == max_seq_length
            assert len(input_mask2) == max_seq_length
            assert len(segment_ids2) == max_seq_length

            if output_mode == "classification":
                label_id2 = label_map[example.label2]
            elif output_mode == "regression":
                label_id2 = float(example.label2)
            else:
                raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid1))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids1]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask1]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids1]))
            logger.info("label: %s (id = %d)" % (example.label1, label_id1))

        if is_training:
            features.append(
                InputFeatures(eid = eid,
                              qid1 = qid1,
                              input_ids1=input_ids1,
                              input_mask1=input_mask1,
                              segment_ids1=segment_ids1,
                              label_id1=label_id1,
                            qid2=qid2,
                            input_ids2=input_ids2,
                            input_mask2=input_mask2,
                            segment_ids2=segment_ids2,
                            label_id2=label_id2))
        else:
            features.append(
                InputFeatures(eid = eid,
                              qid1 = qid1,
                              input_ids1=input_ids1,
                              input_mask1=input_mask1,
                              segment_ids1=segment_ids1,
                              label_id1 = label_id1))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def acc_and_recall(preds, labels):
    rec = 0
    total = 0
    for id, i in enumerate(labels):
        if i == 1:
            if preds[id] == 1:
                rec += 1
            total += 1
    recall = float(rec)/float(total)
    acc = simple_accuracy(preds, labels)
    return {"acc":acc,
            "rec":recall,

    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return acc_and_recall(preds, labels)
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--num_choice',
                        type=int, default=2)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_etr_examples(input_file = args.train_file, is_training=True)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassificationPairwise.from_pretrained(args.bert_model,
                                                          cache_dir=cache_dir,
                                                          num_labels=args.num_choice)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.do_train:
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_mode = 'classification'
    label_list = [0,1]
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, True, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids1 = torch.tensor([f.input_ids1 for f in train_features], dtype=torch.long)
        all_input_mask1 = torch.tensor([f.input_mask1 for f in train_features], dtype=torch.long)
        all_segment_ids1 = torch.tensor([f.segment_ids1 for f in train_features], dtype=torch.long)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in train_features], dtype = torch.long)
        all_input_mask2 = torch.tensor([f.input_mask2 for f in train_features], dtype = torch.long)
        all_segment_ids2 = torch.tensor([f.segment_ids2 for f in train_features], dtype = torch.long)

        if output_mode == "classification":
            all_label_ids1 = torch.tensor([f.label_id1 for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids1 = torch.tensor([f.label_id1 for f in train_features], dtype=torch.float)

        if output_mode == "classification":
            all_label_ids2 = torch.tensor([f.label_id2 for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids2 = torch.tensor([f.label_id2 for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                   all_label_ids1, all_input_ids2, all_input_mask2, all_segment_ids2, all_label_ids2)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids1, input_mask1, segment_ids1, label_ids1, input_ids2, input_mask2, segment_ids2, label_ids2 = batch

                # define a new function to compute loss values for both output_modes
                loss = model(input_ids = input_ids1, token_type_ids = segment_ids1, attention_mask = input_mask1, labels=label_ids1,\
                               input_ids2 = input_ids2, token_type_ids2 = segment_ids2, attention_mask2 = input_mask2, labels2 = label_ids2)

                #if output_mode == "classification":
                #    loss_fct = CrossEntropyLoss()
                #    loss = loss_fct(logits.view(-1, args.num_choice), label_ids.view(-1))
                #elif output_mode == "regression":
                #    loss_fct = MSELoss()
                #    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids1.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassificationPairwise.from_pretrained(args.output_dir, num_labels=args.num_choice)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassificationPairwise.from_pretrained(args.bert_model, num_labels=args.num_choice)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_etr_examples(input_file = args.predict_file, is_training=False)
        eval_features = convert_examples_to_features(
            eval_examples, False, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids1 for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask1 for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids1 for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype = torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id1 for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id1 for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        all_results = []

        for input_ids, input_mask, segment_ids, label_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_choice), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

            for i, example_index in enumerate(example_indices):
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.eid)
                result = logits[i].view(-1).cpu().data.numpy().tolist()
                qid = eval_feature.qid1
                all_results.append(RawResult(unique_id = unique_id, qid = qid, result = result , label = int(eval_feature.label_id1)))

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics("rte", preds, all_label_ids.numpy())
        loss = tr_loss / global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        output_prediction_file = os.path.join(args.output_dir, "prediction.json")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        with open(output_prediction_file, "w") as writer:
            #logger.info("***** Pred results *****")
            writer.write(json.dumps(all_results, indent=4) + "\n")


if __name__ == "__main__":
    main()
from cmath import tanh
from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import BertTokenizer,BertModel
from paddle import dtype, optimizer
import numpy as np
import paddle
import paddle.nn
import sys
import pickle
import util
import datetime

class MyDataset(Dataset):
    def __init__(self, data, tokenizer: BertTokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.SEG_Q = 0
        self.SEG_P = 1
        self.ID_PAD = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        context, query, answers = item["context"], item["query"], item["answer"]
        # 首先编码input_ids ==> 分为Q和P两部分
        query_tokens = [i for i in query]
        context_tokens = [i for i in context]

        # add bert special tokens 
        # Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. 
        # A BERT sequence has the following format:
        # single sequence: [CLS] X [SEP]
        # pair of sequences: [CLS] A [SEP] B [SEP]
        start = 1 + 1 + len(query_tokens) + answers["start"]  # 第一个1代表前插的[CLS],第二个1代表前插的[SEP_A]
        end = 1 + 1 + len(query_tokens) + answers["end"]  # 第一个1代表前插的[CLS],第二个1代表前插的[SEP_A]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_len - 1:
            c = c[:self.max_len-1]
        c += ["[SEP]"]

        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        input_mask = [1] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0] * extra
            input_seg += [self.SEG_P] * extra

        context_start = 2 + len(query_tokens)
        context_end = len(input_ids) - 1
        start_seq_label, end_seq_label = [0] * self.max_len, [0] * self.max_len
        seq_mask = [0] * context_start + [1] * len(context_tokens) + [0] * (self.max_len - context_start - len(context_tokens))
        span_label = np.zeros(shape=(self.max_len, self.max_len), dtype=np.int32)
        triggers = []
        span_mask = np.zeros(shape=(self.max_len, self.max_len), dtype=np.float32)
        for item in answers:
            triggers.append(item["trigger"])
            start_seq_label[context_start + item["start"]] = 1
            end_seq_label[context_start + item["end"]] = 1
            span_label[context_start + item["start"], context_start + item["end"]] = 1
        for i in range(context_start, context_end):
            for j in range(i, context_end):
                span_mask[i, j] = 1.0


        return {
            "input_ids": paddle.to_tensor(input_ids,dtype='int64'),
            "input_seg": paddle.to_tensor(input_seg,dtype='int64'),
            "input_mask": paddle.to_tensor(input_mask,dtype='int32'),
            "context": context,
            "context_range": "%d-%d" % (context_start, context_end),  # 防止被转化成tensor
            "triggers": "&".join(triggers),
            "seq_mask": paddle.to_tensor(seq_mask,dtype='float32'), # TODO
            "start_seq_label": paddle.to_tensor(start_seq_label,dtype='int64'),
            "end_seq_label": paddle.to_tensor(end_seq_label,dtype='int64'),
            "span_label": paddle.to_tensor(span_label,dtype='int64'),
            "span_mask": paddle.to_tensor(span_mask,dtype='float32')
        }


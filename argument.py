from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import BertTokenizer,BertModel
from paddle import dtype, optimizer
import numpy as np
import paddle
import paddle.nn
import pickle

# import torch
import util
import datetime

class MyDataset(Dataset):
    def __init__(self, data, tokenizer: BertTokenizer, max_len, special_query_token_map: dict):
        self.data = data
        self.map = special_query_token_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.SEG_Q = 0
        self.SEG_P = 1
        self.ID_PAD = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        context, query, answer, _type = item["context"], item["query"], item["answer"], item["type"]
        # 首先编码input_ids ==> 分为Q和P两部分
        query_tokens = []
        for i in query:
            if i in self.map.keys():
                query_tokens.append(self.map[i])
            else:
                query_tokens.append(i)
        context_tokens = [i for i in context]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_len - 1:
            c = c[:self.max_len-1]
        c += ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        input_mask = [1] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        context_end = len(input_ids) - 1
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0] * extra
            input_seg += [self.SEG_P] * extra
        x = len(query_tokens) + 2
        return {
            "input_ids": paddle.to_tensor(input_ids,dtype='int64'), 
            "input_seg": paddle.to_tensor(input_seg,dtype='int64'),
            "input_mask": paddle.to_tensor(input_mask,dtype='float32'), 
            "context": context,
            "context_range": "%d-%d" % (2 + len(query_tokens), context_end),  # 防止被转化成tensor
            "cls": answer["is_exist"], 
            "label": answer["argument"],
            "start_index": x + answer["start"], 
            "end_index": x + answer["end"],
            "object_mask": 1.0 if _type == "object" else 0.0, 
            "subject_mask": 1.0 if _type == "subject" else 0.0,
            "time_mask": 1.0 if _type == "time" else 0.0, 
            "location_mask": 1.0 if _type == "location" else 0.0,
            "type": _type
        }
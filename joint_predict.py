#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020-07-23
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage: 
-----------------------------
    Description: 联合预测生成最后测试答案
-----------------------------
"""
import csv
from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import BertTokenizer,BertModel
from paddle import dtype, optimizer
import numpy as np
import paddle
import paddle.nn
import pickle


class DomTrigger(paddle.nn.Module):
    def __init__(self, pre_train_dir: str):
        """
        :param pre_train_dir: 预训练RoBERTa或者BERT文件夹
        """
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=768, out_features=768),
            paddle.nn.Tanh()
        )
        self.start_layer = paddle.nn.Linear(in_features=768, out_features=2)
        self.end_layer = paddle.nn.Linear(in_features=768, out_features=2)
        self.span1_layer = paddle.nn.Linear(in_features=768, out_features=1, bias_attr=False)
        self.span2_layer = paddle.nn.Linear(in_features=768, out_features=1, bias_attr=False)  # span1和span2是span_layer的拆解, 减少计算时的显存占用
        self.epsilon = 1e-6

    def forward(self, input_ids, input_mask, input_seg, span_mask):
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        encoder_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[0]  # (bsz, seq, axis)
        encoder_rep = self.encoder_linear(encoder_rep)
        start_logits = self.start_layer(encoder_rep)  # (bsz, seq, 2)
        end_logits = self.end_layer(encoder_rep)  # (bsz, seq, 2)
        span1_logits = self.span1_layer(encoder_rep)  # (bsz, seq, 1)
        span2_logits = self.span2_layer(encoder_rep).squeeze(axis=-1)  # (bsz, seq)
        # 将两个span组合 => (bsz, seq, seq)
        span_logits = span1_logits.repeat(1, 1, seq_len) + span2_logits[:, None, :].repeat(1, seq_len, 1)
        start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=-1)  # (bsz, seq, 2)
        end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=-1)  # (bsz, seq, 2)
        # 使用span_mask
        span_logits.masked_fill_(span_mask == 0, -1e30)
        span_prob = paddle.nn.functional.softmax(span_logits, axis=-1)  # (bsz, seq, seq)
        return start_prob_seq, end_prob_seq, span_prob


class AuxTrigger(paddle.nn.Module):
    def __init__(self, pre_train_dir: str):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=768, out_features=768),
            paddle.nn.Tanh(),
        )
        self.start_layer = paddle.nn.Linear(in_features=768, out_features=1)
        self.end_layer = paddle.nn.Linear(in_features=768, out_features=1)
        self.epsilon = 1e-6

    def forward(self, input_ids, input_mask, input_seg):
        encoder_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[0]  # (bsz, seq, axis)
        encoder_rep = self.encoder_linear(encoder_rep)
        start_logits = self.start_layer(encoder_rep).squeeze(axis=-1)  # (bsz, seq)
        end_logits = self.end_layer(encoder_rep).squeeze(axis=-1)  # (bsz, seq)
        # adopt softmax function across length axisension with masking mechanism
        mask = input_mask == 0.0
        start_logits.masked_fill_(mask, -1e30)
        end_logits.masked_fill_(mask, -1e30)
        start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=1)
        end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=1)
        return start_prob_seq, end_prob_seq


class Argument(paddle.nn.Module):
    def __init__(self, pre_train_dir: str):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=768, out_features=768),
            paddle.nn.Tanh(),
        )
        self.cls_layer = paddle.nn.Linear(in_features=768, out_features=2)
        self.start_layer = paddle.nn.Linear(in_features=768, out_features=1)
        self.end_layer = paddle.nn.Linear(in_features=768, out_features=1)
        self.epsilon = 1e-6

    def forward(self, input_ids, input_mask, input_seg):
        encoder_rep, cls_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[:2]  # (bsz, seq, axis)
        encoder_rep = self.encoder_linear(encoder_rep)
        cls_logits = self.cls_layer(cls_rep)
        start_logits = self.start_layer(encoder_rep).squeeze(axis=-1)  # (bsz, seq)
        end_logits = self.end_layer(encoder_rep).squeeze(axis=-1)  # (bsz, seq)
        # adopt softmax function across length axisension with masking mechanism
        mask = input_mask == 0.0
        start_logits.masked_fill_(mask, -1e30)
        end_logits.masked_fill_(mask, -1e30)
        start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=1)
        end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=1)
        return cls_logits, start_prob_seq, end_prob_seq
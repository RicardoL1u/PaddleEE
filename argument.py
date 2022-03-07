from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import BertTokenizer,BertModel
from paddle import dtype, optimizer
import numpy as np
import paddle
import paddle.nn
import pickle

# import paddle
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


class MyModel(paddle.nn.Layer):
    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=768, out_features=768),
            paddle.nn.Tanh(),
            paddle.nn.Dropout(p=dropout_rate)
        )
        self.cls_layer = paddle.nn.Linear(in_features=768, out_features=2)
        self.start_layer = paddle.nn.Linear(in_features=768, out_features=1)
        self.end_layer = paddle.nn.Linear(in_features=768, out_features=1)
        self.object_cls_lfc = paddle.nn.CrossEntropyLoss(reduction="none", weight=paddle.tensor([1.0, 1.0]).float().to(device))
        self.subject_cls_lfc = paddle.nn.CrossEntropyLoss(reduction="none", weight=paddle.tensor([10.0, 0.6]).float().to(device))
        self.time_cls_lfc = paddle.nn.CrossEntropyLoss(reduction="none", weight=paddle.tensor([0.76, 1.45]).float().to(device))
        self.location_cls_lfc = paddle.nn.CrossEntropyLoss(reduction="none", weight=paddle.tensor([0.6, 5.5]).float().to(device))
        self.epsilon = 1e-6

    def forward(self, input_ids, input_mask, input_seg, cls_label=None, start_index=None, end_index=None,
                object_mask=None, subject_mask=None, time_mask=None, location_mask=None):
        encoder_rep, cls_rep = self.roberta_encoder(input_ids=input_ids, token_type_ids=input_seg)[:2]  # (bsz, seq, axis)
        encoder_rep = self.encoder_linear(encoder_rep)
        cls_logits = self.cls_layer(cls_rep)
        start_logits = paddle.squeeze(self.start_layer(encoder_rep))  # (bsz, seq)
        end_logits = paddle.squeeze(self.end_layer(encoder_rep))  # (bsz, seq)
        # adopt softmax function across length dimension with masking mechanism
        util.masked_fill(start_logits, input_mask == 0.0, -1e30)
        util.masked_fill(end_logits, input_mask == 0.0, -1e30)
        start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=1)
        end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=1)
        if start_index is None or end_index is None or cls_label is None:
            return cls_logits, start_prob_seq, end_prob_seq
        else:
            object_loss = self.object_cls_lfc(input=cls_logits, target=cls_label)
            subject_loss = self.subject_cls_lfc(input=cls_logits, target=cls_label)
            time_loss = self.time_cls_lfc(input=cls_logits, target=cls_label)
            location_loss = self.location_cls_lfc(input=cls_logits,target=cls_label)
            cls_loss = object_loss * object_mask + subject_loss * subject_mask + time_loss * time_mask + location_loss * location_mask
            # indices select
            start_prob = (start_prob_seq.gather(index=start_index.unsqueeze(axis=-1), axis=1) + self.epsilon).squeeze(axis=-1)
            end_prob = (end_prob_seq.gather(index=end_index.unsqueeze(axis=-1), axis=1) + self.epsilon).squeeze(axis=-1)
            start_loss = -paddle.log(start_prob)
            end_loss = -paddle.log(end_prob)
            span_loss = (start_loss + end_loss) / 2  # (bsz)
            #  TODO: what dose this mean?
            # (bsz)  => when sample label is 0, the span loss is not required.
            span_loss = span_loss * cls_label
            sum_loss = cls_loss + span_loss
            avg_loss = paddle.mean(sum_loss)
            return avg_loss

    
# learning rate decay strategy
class WarmUp_LinearDecay:
    def __init__(self, optimizer: optimizer.AdamW, init_rate, warm_up_steps, decay_steps, min_lr_rate):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.warm_up_steps = warm_up_steps
        self.decay_steps = decay_steps
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= (self.warm_up_steps + self.decay_steps):
            rate = (1.0 - ((self.optimizer_step - self.warm_up_steps) / self.decay_steps)) * self.init_rate
        else:
            rate = self.min_lr_rate
        self.optimizer.set_lr(rate)
        self.optimizer.step()
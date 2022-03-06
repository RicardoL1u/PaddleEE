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

class MyModel(paddle.nn.Layer):
    def __init__(self,pre_train_dir: str, dropout_rate: float, alpha, beta):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=768,out_features=768),
            paddle.nn.Tanh(),
            paddle.nn.Dropout(),
        )
        self.start_layer = paddle.nn.Linear(in_features=768,out_features=2)
        self.end_layer = paddle.nn.Linear(in_features=768,out_features=2)
        self.span1_layer = paddle.nn.Linear(in_features=1024, out_features=1, bias=False)
        self.span2_layer = paddle.nn.Linear(in_features=1024, out_features=1, bias=False)  # span1和span2是span_layer的拆解, 减少计算时的显存占用
        self.selfc = paddle.nn.CrossEntropyLoss(weight=paddle.to_tensor([1.0,10.0],dtype='float32'), reduction="none")
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
    
    
    def forward(self, input_ids, input_mask, input_seg, span_mask,
                start_seq_label=None, end_seq_label=None, span_label=None, seq_mask=None):        

        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        encoder_rep = self.roberta_encoder(input_ids=input_ids,  token_type_ids=input_seg)[0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        
        start_logits = paddle.squeeze(self.start_layer(encoder_rep)) # (bsz, seq)
        end_logits = paddle.squeeze(self.end_layer(encoder_rep))  # (bsz, seq)
        span1_logits = self.span1_layer(encoder_rep)  # (bsz, seq, 1)
        span2_logits = paddle.squeeze(self.span2_layer(encoder_rep))  # (bsz, seq)
        span_logits = paddle.tile(span1_logits,repeat_times=[1, 1, seq_len]) + paddle.tile(span2_logits[:, None, :],repeat_times=[1, seq_len, 1])

        # adopt softmax function across length dimension with masking mechanism
        util.masked_fill(start_logits, input_mask == 0.0, -1e30)
        util.masked_fill(end_logits, input_mask == 0.0, -1e30)
        start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=1)
        end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=1)
        
        util.masked_fill(span_logits,span_mask==0,-1e30)
        span_prob = paddle.nn.functional.softmax(span_logits,axis=-1) # (bsz,seq,seq)
        
        if start_seq_label is None or end_seq_label is None or span_label is None or seq_mask is None:
            return start_prob_seq, end_prob_seq, span_prob
        else:
            # 计算start和end的loss
            start_loss = self.selfc(input=paddle.reshape(start_logits,[-1, 2]), target=paddle.reshape(start_seq_label,[-1,]))
            end_loss = self.selfc(input=paddle.reshape(end_logits,[-1, 2]), target=paddle.reshape(end_seq_label,[-1,]))
            sum_loss = start_loss + end_loss
            sum_loss *= paddle.reshape(seq_mask,[-1,])
            avg_se_loss = self.alpha * paddle.sum(sum_loss) / (paddle.nonzero(seq_mask, as_tuple=False).shape[0])

            # 计算span loss
            span_loss = (-paddle.log(span_prob + self.epsilon)) * span_label
            avg_span_loss = self.beta * paddle.sum(span_loss) / (paddle.nonzero(span_label, as_tuple=False).shape[0])
            return avg_se_loss + avg_span_loss

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
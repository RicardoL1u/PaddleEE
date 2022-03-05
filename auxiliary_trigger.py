from cmath import tanh
from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import RobertaTokenizer,RobertaModel
import paddle
import paddle.nn
import sys
import pickle
import util

class MyDataset(Dataset):
    def __init__(self, data, tokenizer: RobertaTokenizer, max_len):
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
        context, query, answer = item["context"], item["query"], item["answer"]
        # 首先编码input_ids ==> 分为Q和P两部分
        query_tokens = [i for i in query]
        context_tokens = [i for i in context]
        start = 1 + 1 + len(query_tokens) + answer["start"]  # 第一个1代表前插的[CLS],第二个1代表前插的[SEP_A]
        end = 1 + 1 + len(query_tokens) + answer["end"]  # 第一个1代表前插的[CLS],第二个1代表前插的[SEP_A]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_len - 1:
            c = c[:self.max_len-1]
        c += ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        context_end = len(input_ids) - 1
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0.0] * extra
            input_seg += [self.SEG_P] * extra
        return {
            "input_ids": paddle.to_tensor(input_ids).long(),
            "input_seg": paddle.to_tensor(input_seg).long(),
            "input_mask": paddle.to_tensor(input_mask).float(),
            "start_index": start,
            "end_index": end,
        }

class MyModel(paddle.nn.Layer):
    def __init__(self,pre_train_dir: str, dropout_rate: float, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.roberta_encoder = RobertaModel.from_pretrained(pre_train_dir)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=1024,out_features=1024),
            paddle.nn.Tanh(),
            paddle.nn.Dropout(),
        )
        self.start_layer = paddle.nn.Linear(in_features=1024,out_features=1)
        self.end_layer = paddle.nn.Linear(in_features=1024,out_features=1)
        self.epsilon = 1e-6
    
    
    def forward(self, input_ids, input_mask, input_seg, start_index=None, end_index=None):
        encoder_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        # TODO: why squeeze here? origin size is (bsz,seq,1)?
        start_logits = self.start_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        end_logits = self.end_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        # adopt softmax function across length dimension with masking mechanism
        mask = input_mask == 0.0
        util.masked_fill(start_logits, mask, -1e30)
        util.masked_fill(end_logits, mask, -1e30)
        start_prob_seq = paddle.nn.functional.softmax(start_logits, dim=1)
        end_prob_seq = paddle.nn.functional.softmax(end_logits, dim=1)
        if start_index is None or end_index is None:
            return start_prob_seq, end_prob_seq
        else:
            # indices select
            start_prob = start_prob_seq.gather(index=start_index.unsqueeze(dim=-1), dim=1) + self.epsilon
            end_prob = end_prob_seq.gather(index=end_index.unsqueeze(dim=-1), dim=1) + self.epsilon
            # TODO: this is multi classification CE?
            start_loss = -paddle.log(start_prob) 
            end_loss = -paddle.log(end_prob)
            sum_loss = (start_loss + end_loss) / 2
            avg_loss = paddle.mean(sum_loss)
            return avg_loss



if __name__ == "__main__":
    print("Hello RoBERTa Event Extraction.")
    args = {
        "device": "cuda:%s" % sys.argv[1][-1],
        "init_lr": 2e-5,
        "batch_size": 12,
        "weight_decay": 0.01,
        "warm_up_steps": 500,
        "lr_decay_steps": 1500,
        "max_steps": 2000,
        "min_lr_rate": 1e-9,
        "print_interval": 20,
        "save_interval": 500,
        "max_len": 512,
        "save_path": "ModelStorage/auxiliary_trigger.pth",
        "pre_train_dir": "/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/",
        "clip_norm": 0.25,
        "dropout_rate": 0.1
    }

    with open("DataSet/process.p", "rb") as f:
        x = pickle.load(f)

    tokenzier = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
    train_dataset = MyDataset(data=x["train_aux_trigger_items"], tokenizer=tokenzier, max_len=args["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)

    # m = Main(train_loader, args)
    # m.train()
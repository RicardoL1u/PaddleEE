from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import BertTokenizer,BertModel
from paddle import dtype, optimizer
import paddle
import paddle.nn
import pickle
import util
from util import WarmUp_LinearDecay
import datetime
class AuxDataset(Dataset):
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
        input_mask = [1] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        context_end = len(input_ids) - 1
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0] * extra
            input_seg += [self.SEG_P] * extra
        return {
            "input_ids": paddle.to_tensor(input_ids,dtype='int64'),
            "input_seg": paddle.to_tensor(input_seg,dtype='int64'),
            "input_mask": paddle.to_tensor(input_mask,dtype='int32'),
            "start_index": start,
            "end_index": end,
        }

class AuxTrigger(paddle.nn.Layer):
    # TODO: the name_scope and dtype is actually useless
    def __init__(self,pre_train_dir: str, dropout_rate: float, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=768,out_features=768),
            paddle.nn.Tanh(),
            paddle.nn.Dropout(p=dropout_rate),
        )
        self.start_layer = paddle.nn.Linear(in_features=768,out_features=1)
        self.end_layer = paddle.nn.Linear(in_features=768,out_features=1)
        self.epsilon = 1e-6
    
    
    def forward(self, input_ids, input_mask, input_seg, start_index=None, end_index=None):
        encoder_rep = self.roberta_encoder(input_ids=input_ids,  token_type_ids=input_seg)[0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        # TODO: why squeeze here? origin size is (bsz,seq,1)? YES
        start_logits = paddle.squeeze(self.start_layer(encoder_rep),axis=-1) # (bsz, seq)
        end_logits = paddle.squeeze(self.end_layer(encoder_rep),axis=-1)  # (bsz, seq)
        # adopt softmax function across length dimension with masking mechanism
        mask = input_mask == 0.0
        start_logits = util.masked_fill(start_logits, mask, -1e30)
        end_logits = util.masked_fill(end_logits, mask, -1e30)
        start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=1)
        end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=1)
        if start_index is None or end_index is None:
            return start_prob_seq, end_prob_seq
        else:
            # indices select
            start_prob = start_prob_seq.gather(index=start_index.unsqueeze(axis=-1), axis=1) + self.epsilon
            end_prob = end_prob_seq.gather(index=end_index.unsqueeze(axis=-1), axis=1) + self.epsilon
            # TODO: this is multi classification CE?
            start_loss = -paddle.log(start_prob) 
            end_loss = -paddle.log(end_prob)
            sum_loss = (start_loss + end_loss) / 2
            avg_loss = paddle.mean(sum_loss)
            return avg_loss



class AuxTrain(object):
    def __init__(self, train_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.model = AuxTrigger(pre_train_dir=args["pre_train_dir"], dropout_rate=args["dropout_rate"])

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args["weight_decay"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        # print(optimizer_grouped_parameters)
        self.optimizer = optimizer.AdamW(parameters=optimizer_grouped_parameters, learning_rate=args["init_lr"])
        self.schedule = WarmUp_LinearDecay(optimizer=self.optimizer, init_rate=args["init_lr"],
                                           warm_up_steps=args["warm_up_steps"],
                                           decay_steps=args["lr_decay_steps"], min_lr_rate=args["min_lr_rate"])
        self.model.to(device=args["device"])

    def train(self):
        self.model.train()
        steps = 0
        while True:
            if steps >= self.args["max_steps"]:
                break
            for item in self.train_loader:
                input_ids, input_mask, input_seg, start_index, end_index = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["start_index"], item["end_index"]
                self.optimizer.clear_gradients()
                loss = self.model(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_seg=input_seg,
                    start_index=start_index,
                    end_index=end_index
                )
                loss.backward()
                # TODO:what does this mean here?
                paddle.nn.ClipGradByGlobalNorm(group_name=self.model.parameters(), clip_norm=self.args["clip_norm"])
                self.schedule.step()
                steps += 1
                if steps % self.args["print_interval"] == 0:
                    print("{} || [{}] || loss {:.3f}".format(
                        datetime.datetime.now(), steps, loss.item()
                    ))
                if steps % self.args["save_interval"] == 0:
                    paddle.save(obj=self.model.state_dict(), path=self.args["save_path"])
                    print("current model checkpoint has been saved successfully in ModelStorage")
                if steps >= self.args["max_steps"]:
                    break


if __name__ == "__main__":
    print("Hello RoBERTa Event Extraction.")
    args = {
        "device": "gpu:0",
        "init_lr": 2e-5,
        "batch_size": 32,
        "weight_decay": 0.01,
        "warm_up_steps": 500,
        "lr_decay_steps": 1500,
        "max_steps": 2000,
        "min_lr_rate": 1e-9,
        "print_interval": 20,
        "save_interval": 500,
        "max_len": 512,
        "save_path": "ModelStorage/auxiliary_trigger.pth",
        "pre_train_dir": "bert-wwm-chinese",
        "clip_norm": 0.25,
        "dropout_rate": 0.1
    }
    paddle.set_device('gpu:0')
    with open("DataSet/process.p", "rb") as f:
        x = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
    train_dataset = AuxDataset(data=x["train_aux_trigger_items"], tokenizer=tokenizer, max_len=args["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)

    m = AuxTrain(train_loader, args)
    m.train()
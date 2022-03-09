import unittest
from paddlenlp.transformers import RobertaModel,RobertaTokenizer,BertModel,BertTokenizer
from auxiliary_trigger import MyDataset
from paddle.io import DataLoader
import pickle
import paddle
import paddle.nn
import sys
class TestAux(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        with open("DataSet/process.p", "rb") as f:
            self.x = pickle.load(f)
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
        # model = RobertaModel.from_pretrained('roberta-wwm-ext')
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
        self.model = BertModel.from_pretrained('bert-wwm-chinese')

    # def test_dataloader(self):
    #     with open("DataSet/process.p", "rb") as f:
    #         x = pickle.load(f)

    #     # tokenizer = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")
    #     tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
    #     train_dataset = MyDataset(data=x["train_aux_trigger_items"], tokenizer=tokenizer, max_len=256)
    #     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    #     # for item in train_loader
    #     tokens = tokenizer("这是一次测试，我不知道测试结果会咋样")
    #     print(tokens)
    #     print(len(train_loader))
    def test_bert(self):
        inputs = self.tokenizer("欢迎使用百度飞桨")
        inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
        sequence_output, pooled_output = self.model(**inputs)[:2]
        print(sequence_output.shape)
        print(pooled_output.shape)

    def test_dataset(self):
        train_dataset = MyDataset(data=self.x["train_aux_trigger_items"], tokenizer=self.tokenizer, max_len=256)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        for item in train_loader:
            input_ids, input_mask, input_seg, start_index, end_index = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["start_index"], item["end_index"]
            # print(input_ids)
            # print(input_mask)
            # print(input_seg)
            print(start_index)
            encoder_rep = self.model(input_ids=input_ids, token_type_ids=input_seg)[0]  # (bsz, seq, dim)
            start_layer = paddle.nn.Linear(in_features=768,out_features=1)
            start_logits = paddle.squeeze(start_layer(encoder_rep))
            start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=1)
            start_prob = start_prob_seq.gather(index=start_index.unsqueeze(axis=-1), axis=1) + 1e-6
            print(start_prob)
            # print(start_logits)
            # print(encoder_rep)
            break
if __name__ == "__main__":
    unittest.main()

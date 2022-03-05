import unittest
from paddlenlp.transformers import RobertaModel,RobertaTokenizer
from auxiliary_trigger import MyDataset
from paddle.io import DataLoader
import pickle
import paddle
import sys
class TestAux(unittest.TestCase):
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
    def test_roberta(self):
        with open("DataSet/process.p", "rb") as f:
            x = pickle.load(f)

        # tokenizer = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
        roberta = RobertaModel.from_pretrained('roberta-wwm-ext')
        
        inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
        inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
        sequence_output, pooled_output = roberta(**inputs)
        print(inputs)
        print(len(inputs))
        print()
        print(sequence_output)
        # train_dataset = MyDataset(data=x["train_aux_trigger_items"], tokenizer=tokenizer, max_len=256)
        # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        # for item in train_loader:
        #     input_ids, input_mask, input_seg, start_index, end_index = \
        #             item["input_ids"], item["input_mask"], item["input_seg"], item["start_index"], item["end_index"]
        #     print(input_ids)
        #     print(input_mask)
        #     print(input_seg)
        #     encoder_rep = roberta(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[0]  # (bsz, seq, dim)
        #     print(encoder_rep)
        #     break
if __name__ == "__main__":
    unittest.main()

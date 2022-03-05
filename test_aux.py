import unittest
from paddlenlp.transformers import BertTokenizer,RobertaTokenizer
from auxiliary_trigger import MyDataset
from paddle.io import DataLoader
import pickle

class TestAux(unittest.TestCase):
    def test_dataloader(self):
        with open("DataSet/process.p", "rb") as f:
            x = pickle.load(f)

        # tokenzier = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")
        tokenzier = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
        train_dataset = MyDataset(data=x["train_aux_trigger_items"], tokenizer=tokenzier, max_len=256)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        # for item in train_loader
        tokens = tokenzier("这是一次测试，我不知道测试结果会咋样")
        print(tokens)
        print(len(train_loader))

if __name__ == "__main__":
    unittest.main()

import unittest
from paddlenlp.transformers import BertTokenizer,RobertaTokenizer
from auxiliary_trigger import MyDataset
from paddle.io import DataLoader
import pickle
import sys
class TestAux(unittest.TestCase):
    # def test_dataloader(self):
    #     with open("DataSet/process.p", "rb") as f:
    #         x = pickle.load(f)

    #     # tokenzier = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")
    #     tokenzier = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
    #     train_dataset = MyDataset(data=x["train_aux_trigger_items"], tokenizer=tokenzier, max_len=256)
    #     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    #     # for item in train_loader
    #     tokens = tokenzier("这是一次测试，我不知道测试结果会咋样")
    #     print(tokens)
    #     print(len(train_loader))
    
    def test_args(self):
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
            "pre_train_dir": "roberta-wwm-ext",
            "clip_norm": 0.25,
            "dropout_rate": 0.1
        }
        print(args)

if __name__ == "__main__":
    unittest.main()

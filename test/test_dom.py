import unittest
import sys
sys.path.append("..")

import pickle
import dominant_trigger
from paddlenlp.transformers import BertTokenizer
from paddle.io import DataLoader

class TestDom(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.max_len = 64
        self.bsz = 2
        with open("DataSet/process.p", "rb") as f:
            self.x = pickle.load(f)
        self.tokenizer = BertTokenizer.from_pretrained("bert-wwm-chinese")
        
    def test_data(self):
        train_dataset = dominant_trigger.MyDataset(data=self.x["train_dominant_trigger_items"], tokenizer=self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=self.bsz, shuffle=False, num_workers=4)
        for item in train_loader:
            print(item)
            break
        print("hi")
    
    def test_validset(self):
        valid_dataset =  dominant_trigger.MyDataset(data=self.x["valid_dominant_trigger_items"], tokenizer=self.tokenizer, max_len=128)
        # valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=4)
        # cnt = 0
        for item in valid_dataset:
            # cnt += 1
            if "&" in item["triggers"]:
                print(item)
                break
    
    def test_join(self):
        x = ["mike","jack"]
        x = "".join(x)
        for k in x:
            print(k)

if __name__ == "__main__":
    unittest.main()        
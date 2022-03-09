import unittest
import pickle
import dominant_trigger
from paddlenlp.transformers import BertTokenizer
from paddle.io import DataLoader

class TestDom(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.max_len = 128
        self.bsz = 4
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

if __name__ == "__main__":
    unittest.main()        
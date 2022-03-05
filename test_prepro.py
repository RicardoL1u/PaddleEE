import unittest
import  pickle
import csv
import jieba

class TestPreprocess(unittest.TestCase):
    def test_pfile(self):
        with open("DataSet/process.p", "rb") as f:
            x = pickle.load(f)
        print(type(x))
        # print(x["train_aux_trigger_items"][123])
        # print(x["train_dominant_trigger_items"][123])
        # print(x["valid_dominant_trigger_items"][123])
        print(len(x["train_aux_trigger_items"]))
        print(len(x["train_dominant_trigger_items"]))
        print(len(x["valid_dominant_trigger_items"]))
    
    def test_csvreader(self):
        sample_file = open('trains_sample.csv','r',encoding='UTF-8')
        sample_reader = csv.reader(sample_file)
        next(sample_reader)
        for item in sample_reader:
            print(item)
            _context = item[1]
            break
        x = list(jieba.tokenize(_context))  # 切词带索引
        y = jieba.lcut(_context)  # 单纯的切词序列
        print(x)
        print(y)

if __name__ == "__main__":
    unittest.main()

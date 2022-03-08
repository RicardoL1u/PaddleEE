import unittest
import  pickle
import csv
import jieba
import joint_predict

class TestPreprocess(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        with open("DataSet/process.p", "rb") as f:
            self.x = pickle.load(f)

    def test_pfile(self):
        print(type(self.x))
        # print(x["train_aux_trigger_items"][123])
        # print(x["train_dominant_trigger_items"][123])
        # print(x["valid_dominant_trigger_items"][123])
        print(len(self.x["train_aux_trigger_items"]))
        print(len(self.x["train_dominant_trigger_items"]))
        print(len(self.x["valid_dominant_trigger_items"]))
    
    # the way to test single method
    # python3 test_prepro.py TestPreprocess.test_csvreader
    # python3 file.py class.method
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
    
    def test_testcases(self):
        test_items, special_map = self.x["test_items"], self.x["argument_query_special_map_token"]
        # print(special_map)
        cnt = 0
        for item in test_items:
            print(item)
            cnt+=1
            if cnt == 3:
                break


if __name__ == "__main__":
    unittest.main()

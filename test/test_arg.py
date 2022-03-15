import unittest
import sys
sys.path.append("..")

from paddlenlp.transformers import RobertaModel,RobertaTokenizer,BertModel,BertTokenizer
from argument import MyDataset
from paddle.io import DataLoader
import pickle
import paddle
import paddle.nn

class TestArg(unittest.TestCase):
    def test_cls(self):
        with open("../DataSet/process.p", "rb") as f:
            x = pickle.load(f)

        tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
        # model = BertModel.from_pretrained('bert-wwm-chinese')

        train_dataset = MyDataset(data=x["train_argument_items"], tokenizer=tokenizer, max_len=256, special_query_token_map=x["argument_query_special_map_token"])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=3)
        for item in train_loader:
            input_ids, input_mask, input_seg, cls_label, start_index, end_index, obj_mask, sub_mask, tim_mask, loc_mask = \
                item["input_ids"], item["input_mask"], item["input_seg"], item["cls"], item["start_index"], item["end_index"], \
                item["object_mask"], item["subject_mask"], item["time_mask"], item["location_mask"]
            print(item["context"][1])
            print(item["start_index"][1])
            print()
            print(item["context"][2])
            print(item["start_index"][2])
            print(cls_label)
            break

if __name__ == "__main__":
    unittest.main()

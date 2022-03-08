import unittest
import pickle
import joint_predict
from paddlenlp.transformers import BertTokenizer,BertModel
import paddle

class TestJoint(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        paddle.set_printoptions(threshold=256*256)
        with open("DataSet/process.p", "rb") as f:
            x = pickle.load(f)
            self.test_items, self.special_map = x["test_items"], x["argument_query_special_map_token"]
        self.max_len = 128
        self.Tokenizer = BertTokenizer.from_pretrained("bert-wwm-chinese")
        # self.encode_obj = InputEncoder(max_len=max_len, tokenizer=self.Tokenizer, special_query_token_map=special_map)
    
    def test_encode(self):
        encode_obj = joint_predict.InputEncoder(max_len=self.max_len, tokenizer=self.Tokenizer, special_query_token_map=self.special_map)
        
        for item in self.test_items:
            id, context, n_triggers = item["id"], item["context"], item["n_triggers"]
            trigger_input = encode_obj.trigger_enc(context=context, is_dominant=True)
            # print(trigger_input)
            print(len(trigger_input['span_mask']))
            # print(trigger_input['span_mask'][0][:][11:70])
            print(trigger_input['span_mask'][0][11][11:70])
            print(trigger_input['span_mask'][0][10][11:70])
            break


if __name__ == "__main__":
    unittest.main()

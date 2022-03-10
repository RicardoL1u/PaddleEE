import unittest
import pickle
import joint_predict
from paddlenlp.transformers import BertTokenizer,BertModel
import paddle
import paddle.nn
import torch.nn.functional

from util import paddle2torch

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
            print(trigger_input['span_mask'][0][10][11:70]) ## all zero
            break
    
    # TODO: one bug here!
    def test_span(self):
        seq_len = 4
        span1_logits = paddle.randn([1,seq_len,1])
        span2_logits = paddle.randn([1,seq_len])
        span_logits = paddle.tile(span1_logits,repeat_times=[1, 1, seq_len]) + paddle.tile(span2_logits[:, None, :],repeat_times=[1, seq_len, 1])
        # print(span1_logits)
        # print(paddle.tile(span1_logits,repeat_times=[1, 1, seq_len]))
        # print(span2_logits)
        # print(paddle.tile(span2_logits[:, None, :],repeat_times=[1, seq_len, 1]))
        print(span_logits.reshape([1,-1]))
        span_prob = paddle.nn.functional.softmax(span_logits.reshape([1,-1]), axis=1).reshape([1,seq_len,-1])
        print(span_prob)
        span_prob = paddle.nn.functional.softmax(span_prob,axis=1)
        print(span_prob)
        unit = span_logits[0,:,2]
        print(unit)
        print(paddle.nn.functional.softmax(unit))
        # print()  # (bsz, seq, seq)
        # print(torch.nn.functional.softmax(paddle2torch(span_logits), dim=1))  # (bsz, seq, seq)
        # print(torch.softmax(paddle2torch(span_logits),dim=1))

    def test_softmax(self):
        x = paddle.randn([1,4,4])
        print(x)
        print(paddle.nn.functional.softmax(x,axis=1))

if __name__ == "__main__":
    unittest.main()

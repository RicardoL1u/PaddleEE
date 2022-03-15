import unittest
import sys
sys.path.append("..")

import paddle
import paddle.nn
class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        bsz = 2
        seq_len = 4
        x = paddle.randn([bsz*seq_len,2])
        y = paddle.randint(low=0,high=2,shape=[bsz*seq_len])
        print(x)
        print(y)
        selfc = paddle.nn.CrossEntropyLoss(weight=paddle.to_tensor([1.0,10.0],dtype='float32'), reduction="none")
        print(selfc(input=x,label=y))

if __name__ == "__main__":
    unittest.main()          
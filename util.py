# paddlepaddle >= 2.0
import paddle
# import torch


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

# def paddle2torch(x:paddle.Tensor)->torch.Tensor:
#     return torch.from_numpy(x.cpu().numpy())

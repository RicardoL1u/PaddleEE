# paddlepaddle >= 2.0
import paddle
from paddle import optimizer

# import torch


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

# def paddle2torch(x:paddle.Tensor)->torch.Tensor:
#     return torch.from_numpy(x.cpu().numpy())


# learning rate decay strategy
class WarmUp_LinearDecay:
    def __init__(self, optimizer: optimizer.AdamW, init_rate, warm_up_steps, decay_steps, min_lr_rate):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.warm_up_steps = warm_up_steps
        self.decay_steps = decay_steps
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= (self.warm_up_steps + self.decay_steps):
            rate = (1.0 - ((self.optimizer_step - self.warm_up_steps) / self.decay_steps)) * self.init_rate
        else:
            rate = self.min_lr_rate
        self.optimizer.set_lr(rate)
        self.optimizer.step()

def generate_query(arg,start,end,trigger)->str:
    if arg == "object":
        query = "处于位置&%d&和位置-%d-之间提到的*%s*事件的发起者(比如公司,政府党派,学校,政府,新闻机构,人名)为?" % (start, end, trigger)
    elif arg == "subject":
        query = "处于位置&%d&和位置-%d-之间提到的*%s*事件的行动对象(比如会议,活动,项目,计划,任务,以及组织,公司,人名)为?" % (start, end, trigger)
    elif arg == "time":
        query = "处于位置&%d&和位置-%d-之间提到的*%s*事件发生的时间(比如年,月,日,季度,时刻,节日)为?" % (start, end, trigger)
    elif arg == "location":
        query = "处于位置&%d&和位置-%d-之间提到的*%s*事件发生的地点(比如国家,城市,乡镇,大洲,公共场所,学校)为" % (start, end, trigger)

    return query
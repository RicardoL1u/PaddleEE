# 迁移代码-from pytorch to paddlepaddle

## Task

科大讯飞2020事件抽取比赛http://challenge.xfyun.cn/topic/info?type=hotspot

> 文本：过渡政府部队发言人说, 北约的战机 1 6 日在苏尔特附近击中了一座建筑，炸死大批卡扎菲部队士兵。抽取结果：![img](http://xfyun-doc.ufile.ucloud.com.cn/1590645969288906/3.png)



## Pipeline

### Find out Trigger word

#### encode part

1. 加上query并padding成相同长度

   [CLS] + query + [SEP] + context + [SEP] 

   (bsz, seq_len)

2. 通过bert来做encoding

   encode-rep (bsz,seq, bertdim)

3. 通过线性层来得到，每一个token的是否是（start/end）的表示

   也就是对每一个 **字** 做一次二分类 判断其是否为 start

   > 自然我们还要有另一个二分类，来判断其是否为 end
   >
   > 所以一个字其实是通过了两个二分类

   ```python
   start_logits = self.start_layer(encoder_rep)  # (bsz, seq, 2)
   end_logits = self.end_layer(encoder_rep)  # (bsz, seq, 2)
   start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=-1)  # (bsz, seq, 2)
   end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=-1)  # (bsz, seq, 2)
   ```

   > axis = -1 等价于与 axis = 2 及对最后一个维度，做softmax

4. 将start和end组合，**这里他是有bug的**

   ```python
   span1_logits = self.span1_layer(encoder_rep)  # (bsz, seq, 1)
   span2_logits = paddle.squeeze(self.span2_layer(encoder_rep))  # (bsz, seq)
   span_logits = paddle.tile(span1_logits,repeat_times=[1, 1, seq_len]) + paddle.tile(span2_logits[:, None, :],repeat_times=[1, seq_len, 1])
   # -1e30 是为了给在 span_mask 之外的猜测一个极小的 概率 （再通过softmax之后）
   span_logits = util.masked_fill(span_logits,span_mask==0,-1e30) # span_mask是一个上三角矩阵（上三角全为1）
   span_prob = paddle.nn.functional.softmax(span_logits,axis=1) # (bsz,seq,seq)
   ```

   也就是说他这里用 span_prob 的下标来存储 了 start index和 end index （有点像邻接矩阵）

#### decoder part

0. 输入是 context 以及 start_prob_seq, end_prob_seq (bsz,seq,2) 以及 span_prob

1. 先找出所有被判别为开始和结束的位置索引

2. 然后 遍历 每一个 判断为end的索引

   对于 每一个 end，找到所有在之前的S

3. 对于一个 end 和 所有在其的之前的 S，通过 span_prob 找出最佳的 S 

   ```python
   max_s = 0.0
   t = None
   # 找出其中能使得发生概率最大的 S
   for i in s:
       if p_seq[i, e] > max_s:
           t = (i, e)
           max_s = p_seq[i, e]
   cur_end = e
   if t is not None:
       ans_index.append(t)
   ```

   这个t就是找到的一个 trigger word了

## Find the Argument

#### Encoder Part

1. 加上query并padding成相同长度

   [CLS] + query + [SEP] + context + [SEP] 

   (bsz, seq_len)

   ```python
   query = "处于位置&%d&和位置-%d-之间的触发词*%s*的%s为?" % (start, end, trigger, self.arg_map[arg])
   ```

2. 通过bert来做encoding

   encode-rep (bsz,seq, bertdim) cls_rep (bsz, dim)

3. cls_rep 过一个线性层变成 cls_logits (bsz,2)

4. 让后通过两个 seq_len分类器，给每个token，分配其为start/end的概率

   ```python
   start_logits = self.start_layer(encoder_rep).squeeze(axis=-1)  # (bsz, seq)
   end_logits = self.end_layer(encoder_rep).squeeze(axis=-1)  # (bsz, seq)
   start_prob_seq = paddle.nn.functional.softmax(start_logits, axis=1)
   end_prob_seq = paddle.nn.functional.softmax(end_logits, axis=1)
   ```

   start_prob_seq, end_prob_seq (bsz,seq)

5. 最后返回 cls_logits (bsz,2), start_prob_seq, end_prob_seq  (bsz,seq)

#### Decoder Part

0. 输入 cls_logits, start_prob_seq, end_prob_seq
1. 通过 cls_logits 判断是否存在该类型轮元
2. 如果存在该类型argument 找到一组符合 s < e 且使得 s_prob[s] + e_prob[e]最大 的 s,e组合进行返回，作为论元

## Paddle VS PyTorch

transformer在paddle中是在 paddlenlp这个特殊的包中 https://paddlenlp.readthedocs.io/

而pytorch一般采用huggingface的 https://huggingface.co/docs/transformers/installation#install-with-condax



## Paddle Bug

RobertaModel 传参 attention_mask 后会报错 换成bert 即可避免

> 好像所有的都有这个问题，只能不传attention—mask了


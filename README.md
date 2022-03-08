# PaddleEE
Event Extraction Pipeline for PaddlePaddle translated from Pytorch
> We do not produce codes, we are the translators on GitHub

## Task Description
the task we faced was proposed in event extraction challenge 2020 by iFLYTEK 科大讯飞

the detailed description of the event extraction task could be checked from [the open platform of iFLYTEK](http://challenge.xfyun.cn/topic/info?type=hotspot)

## environments
this repo could be directly conducted on the notebook provided by [aistudio](https://aistudio.baidu.com/aistudio/usercenter)

## how to execute
- step1: run preprocess.py
- step2: 
    - run auxiliary_trigger.py;
    - run dominant_trigger.py;
    - run argument.py;
- step3: run joint_predict.py

核心思路: 以Bert作为Context Encoder, 将问题拆解为触发词识别和触发词对应四论元检测
触发词识别和论元识别基于MRC(Span Extraction类机器阅读理解)思路实现

>参考论文: ACL2020 《A Unified MRC Framework for Named Entity Recognition》

single model score: 0.78

## Pytorch Version
stored in the folder 'pytorch'



# QBQTC
QBQTC: QQ Browser Query Title Corpus


# 数据集介绍
TODO 这里是简要的数据集介绍

# baseline效果对比
TODO baseline效果对比
| 模型 | 训练集（train) | 验证集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: | :----: |
| <a href="https://huggingface.co/bert-base-chinese/tree/main">BERT-base</a> | F1:93.2  Acc:93.7 | F1: 64.1 Acc:66.6 | F1: 65.1 Acc:66.2 | batch=256, length=52, epoch=7, lr=2e-5, warmup=0.9 |
|<a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext"> RoBERTa-wwm-ext</a> | F1:89.0 Acc:89.8 | F1:62.3 Acc:64.8 | F1:65.1 Acc:66.2 | batch=256, length=52, epoch=7, lr=2e-5, warmup=0.9|
| <a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext-large">RoBERTa-wwm-large-ext</a> | F1:92.6 Acc:93.0 | F1:61.7 Acc:63.7 | F1:65.1 Acc:65.9 | batch=256, length=52, epoch=7, lr=2e-5, warmup=0.9|

# 一键运行baseline
---------------------------------------------------------------------
    使用方式：
    1、克隆项目 
       git clone https://github.com/CLUEbenchmark/QBQTC.git
    2、进入到相应的目录
      例如：cd QBQTC/baselines
    3、下载对应任务模型参数
    	QBQTC/weights/bert-base-chinese
    	QBQTC/weights/chinese-roberta-wwm-ext
    	QBQTC/weights/chinese-roberta-wwm-ext-large
    4、运行对应任务的模型(GPU方式): 
       python BERT.py --model_name_or_path ../weights/chinese-roberta-wwm-ext --max_seq_length 52 --batch_size 512 --num_epochs 7 --learning_rate 2e-5 --num_labels 3
       简化版：python BERT.py

# 数据集例子
TODO 数据集例子

# 提交样例
TODO 提交样例

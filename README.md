# QBQTC
QBQTC: QQ Browser Query Title Corpus


# 数据集介绍
TODO 这里是简要的数据集介绍

# baseline效果对比
TODO baseline效果对比
| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| BERT-base |F1:92.30 EM:86.60 | F1:91.46 EM:85.49 |  batch=32, length=512, epoch=2, lr=3e-5, warmup=0.1 |
| RoBERTa-wwm-ext |F1:94.26 EM:89.29 | F1:93.53 EM:88.12 |  batch=32, length=512, epoch=2, lr=3e-5, warmup=0.1|
| RoBERTa-wwm-large-ext |***F1:95.32 EM:90.54*** | ***F1:95.06 EM:90.70*** | batch=32, length=512, epoch=2, lr=2.5e-5, warmup=0.1 |

# 一键运行baseline
<a href="https://huggingface.co/bert-base-chinese/tree/main">BERT-base</a>
<a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext">RoBERTa-wwm-ext</a>
<a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext-large">RoBERTa-wwm-large</a>
---------------------------------------------------------------------
    使用方式：
    1、克隆项目 
       git clone https://github.com/CLUEbenchmark/QBQTC.git
    2、进入到相应的目录
      例如：
      cd QBQTC/baselines
    3、下载对应任务模型参数
    	
    4、运行对应任务的模型(GPU方式): 。
       python BERT.py --model_name_or_path ../weights/chinese-roberta-wwm-ext --max_seq_length 52 --batch_size 512 --num_epochs 7 --learning_rate 2e-5 --num_labels 3
       简化版：python BERT.py

# 数据集例子
TODO 数据集例子

# 提交样例
TODO 提交样例

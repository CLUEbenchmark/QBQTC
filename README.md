# QBQTC
QBQTC: QQ Browser Query Title Corpus


# 数据集介绍
TODO 这里是简要的数据集介绍

# baseline效果对比
TODO baseline效果对比


# 一键运行baseline
---------------------------------------------------------------------
    使用方式：
    1、克隆项目 
       git clone https://github.com/CLUEbenchmark/QBQTC.git
    2、进入到相应的目录
           例如：
           cd QBQTC/baselines
    4、下载对应任务模型参数
    	<a href="https://huggingface.co/bert-base-chinese/tree/main">BERT-base</a>
    	<a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext">RoBERTa-wwm-ext</a>
    	<a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext-large">RoBERTa-wwm-large</a>
    	
    3、运行对应任务的模型(GPU方式): 。
       python BERT.py --model_name_or_path ../weights/chinese-roberta-wwm-ext --max_seq_length 52 --batch_size 512 --num_epochs 7 --learning_rate 2e-5 --num_labels 3
       简化版：python BERT.py

# 数据集例子
TODO 数据集例子

# 提交样例
TODO 提交样例

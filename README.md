# QBQTC
QBQTC: QQ Browser Query Title Corpus

QQ浏览器搜索相关性数据集


# 数据集介绍
QQ浏览器搜索相关性数据集（QBQTC,QQ Browser Query Title Corpus），是QQ浏览器搜索引擎目前针对大搜场景构建的一个融合了相关性、权威性、内容质量、
时效性等维度标注的学习排序（LTR）数据集，广泛应用在搜索引擎业务场景中。

相关性的含义：0，相关程度差；1，有一定相关性；2，非常相关。数字越大相关性越高。

#### 数据量统计
 | 训练集（train) | 验证集（dev) | 公开测试集（test_public) | 私有测试集(test) |
| :----: | :----: | :----: | :----: |
| 180,000| 20,000| 5,000 | >=10,0000|

# baseline效果对比

| 模型 | 训练集（train) | 验证集（dev) | 公开测试集（test_public) | 训练参数 |
| :----:| :----: | :----: | :----: | :----: |
|<a href="https://huggingface.co/bert-base-chinese/tree/main">BERT-base</a> | F1:80.3  Acc:84.3 | F1: 64.9 Acc:72.4 | F1: 64.1 Acc:71.8 | batch=64, length=52, epoch=7, lr=2e-5, warmup=0.9 |
|<a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext"> RoBERTa-wwm-ext</a> | F1:67.9 Acc:76.2 | F1:64.9 Acc:71.5 | F1:64.0 Acc:71.0 | batch=64, length=52, epoch=7, lr=2e-5, warmup=0.9|
|<a href="https://huggingface.co/hfl/chinese-roberta-wwm-ext-large">RoBERTa-wwm-large-ext</a> | F1:79.8 Acc:84.2 | F1:65.1 Acc:72.4 | F1:66.3 Acc:73.1 | batch=64, length=52, epoch=7, lr=2e-5, warmup=0.9|

f1_score来自于sklearn.metrics，计算公式如下：
`F1 =  2 * (precision * recall) / (precision + recall)`

# 一键运行baseline

### 🏝运行环境：python 3.x/pytorch 1.7.1/cuda 11.0 + transformers 3.5.0

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
       python BERT.py --model_name_or_path ../weights/chinese-roberta-wwm-ext --max_seq_length 52 --batch_size 64 --num_epochs 7 --learning_rate 2e-5 --num_labels 3
       简化版：python BERT.py

# 数据集例子
    {"id": 0, "query": "小孩咳嗽感冒", "title": "小孩感冒过后久咳嗽该吃什么药育儿问答宝宝树", "label": "1"}
    {"id": 1, "query": "前列腺癌根治术后能活多久", "title": "前列腺癌转移能活多久前列腺癌治疗方法盘点-家庭医生在线肿瘤频道", "label": "1"}
    {"id": 3, "query": "如何将一个文件复制到另一个文件里", "title": "怎么把布局里的图纸复制到另外一个文件中去百度文库", "label": "0"}
    {"id": 214, "query": "免费观看电影速度与激情1", "title": "《速度与激情1》全集-高清电影完整版-在线观看", "label": "2"}
    {"id": 98, "query": "昆明公积金", "title": "昆明异地购房不能用住房公积金中新网", "label": "2"}
    {"id": 217, "query": "多张图片怎么排版好看", "title": "怎么排版图片", "label": "2"}


# 提交样例
<a href="./resources/qbqtc_submit_examples/">提交样例</a>

在测试集(test.json)上做测试预测，并提交到<a href="https://www.CLUEbenchmarks.com">测评系统</a>

# 问题反馈与交流
1) 可以提交issue反馈问题；或添加QQ交流群:836811304
2) 欢迎提交基线模型(baseline)代码。你可以通过提交一个pr，具体是在baselines目录下新建一个文件夹，我们会在24小时内处理。

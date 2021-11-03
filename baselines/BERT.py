# -*- coding: utf-8 -*-
"""
Created on Mon Nov 1 2021

@author: Taurids
"""
import time
import torch
import clue7
import argparse
from transformers import BertTokenizer, BertModel, BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_parser = argparse.ArgumentParser()
# 要求参数
my_parser.add_argument("--model_name_or_path", default="../weights/chinese-roberta-wwm-ext", type=str, required=False)
my_parser.add_argument("--max_seq_length", default=52, type=int, required=False)  # 文本截断长度
my_parser.add_argument("--batch_size", default=64, type=int, required=False)
my_parser.add_argument("--num_epochs", default=7, type=int, required=False)
my_parser.add_argument("--learning_rate", default=2e-5, type=float, required=False)
my_parser.add_argument("--warmup_proportion", default=0.9, type=int, required=False)
my_parser.add_argument("--warmup_step", default=2, type=int, required=False)
my_parser.add_argument("--num_labels", default=3, type=int, required=False)


args = my_parser.parse_args()

# 数据预处理
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path + '/vocab.txt')
train_set = clue7.CustomDataset('train.json', tokenizer, args.max_seq_length)
valid_set = clue7.CustomDataset('dev.json', tokenizer, args.max_seq_length)
test_set = clue7.CustomDataset('test_public.json', tokenizer, args.max_seq_length)

# DataLoader
train_params = {'batch_size': args.batch_size, 'shuffle': True}
valid_params = {'batch_size': args.batch_size, 'shuffle': True}
test_params = {'batch_size': args.batch_size, 'shuffle': False}
train_loader = torch.utils.data.DataLoader(train_set, **train_params)
valid_loader = torch.utils.data.DataLoader(valid_set, **valid_params)
test_loader = torch.utils.data.DataLoader(test_set, **test_params)


# 创建模型
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.config = BertConfig.from_pretrained(args.model_name_or_path + '/config.json')
        self.bert = BertModel.from_pretrained(args.model_name_or_path + '/pytorch_model.bin', config=self.config)
        self.linear = torch.nn.Linear(self.config.hidden_size, args.num_labels, bias=True)  # 分三类

    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooler_output = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        # [bs, 768]
        output = self.linear(pooler_output)
        return output


net = BERTClass()
net.to(device)

# 超参数设置
criterion = torch.nn.CrossEntropyLoss()  # 选择损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # 选择优化器

# 训练模型
clue7.train(net, train_loader, valid_loader, criterion, args.num_epochs, optimizer, device, args)


# 预测函数
def model_predict(pre_net, test_iter):
    # 预测模型
    print('加载最优模型')
    pre_net.load_state_dict(torch.load('../model/best.pth'))
    pre_net.to(device)
    print('inference测试集')
    with torch.no_grad():
        start = time.time()
        test_acc, test_f1 = clue7.evaluate_accuracy(test_iter, pre_net, device)
        print('test acc %.3f, test f1 %.3f, time %.1f sec'
              % (test_acc, test_f1, time.time() - start))


# 预测
model_predict(net, test_loader)

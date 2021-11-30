# -*- coding: utf-8 -*-
"""
Created on Mon Nov 1 2021

@author: Taurids
"""
import time
import torch
from clue import opt, loss
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
my_parser.add_argument("--cate_performance", default=True, type=bool, required=False)


args = my_parser.parse_args()

# 数据预处理
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path + '/vocab.txt')
train_set = opt.CustomDataset('train.json', tokenizer, args.max_seq_length)
valid_set = opt.CustomDataset('dev.json', tokenizer, args.max_seq_length)
test_set = opt.CustomDataset('test_public.json', tokenizer, args.max_seq_length)

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
        self.config = BertConfig.from_pretrained(args.model_name_or_path + '/config.json', output_hidden_states=True)
        self.bert = BertModel.from_pretrained(args.model_name_or_path + '/pytorch_model.bin', config=self.config)
        self.bi_lstm = torch.nn.LSTM(self.config.hidden_size*4, 128, 1, bidirectional=True)

        self.dropout = torch.nn.Dropout(0.25)
        self.softmax_d1 = torch.nn.Softmax(dim=1)
        self.atten_layer = torch.nn.Linear(256, 16)
        self.linear_layer = torch.nn.Linear(256, 16 * args.num_labels)

    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooler_output, hidden_states = self.bert(
            input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        # [bs, len, 768]  [bs, 768]
        h11_share = hidden_states[-1][:, 0].unsqueeze(0)
        h10_share = hidden_states[-2][:, 0].unsqueeze(0)
        h09_share = hidden_states[-3][:, 0].unsqueeze(0)
        h08_share = hidden_states[-4][:, 0].unsqueeze(0)
        concat_hidden_share = torch.cat((h11_share, h10_share, h09_share, h08_share), 2)  # [1, bs, 768*4]
        cls_emb = self.bi_lstm(concat_hidden_share)[0].squeeze(0)  # [bs, 256]

        attention_score = self.atten_layer(cls_emb)  # [bs, 16]
        attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))  # [bs, 1, 16]
        value = self.linear_layer(cls_emb).contiguous().view(-1, 16, args.num_labels)  # [bs, 16, 3]
        output = torch.matmul(attention_score, value).squeeze(1)  # [bs, 3]

        return output


net = BERTClass()
net.to(device)

# 超参数设置
criterion = loss.FocalLoss()  # 选择损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # 选择优化器

# 训练模型
opt.train(net, train_loader, valid_loader, criterion, args.num_epochs, optimizer, device, args)


# 预测函数
def model_predict(pre_net, test_iter):
    # 预测模型
    print('加载最优模型')
    pre_net.load_state_dict(torch.load('../model/best.pth'))
    pre_net.to(device)
    print('inference测试集')
    with torch.no_grad():
        start = time.time()
        test_acc, test_f1 = opt.evaluate_accuracy(args, test_iter, pre_net, device)
        print('test acc %.3f, test f1 %.3f, time %.1f sec'
              % (test_acc, test_f1, time.time() - start))


# 预测
model_predict(net, test_loader)

# coding=utf-8
import json
from tqdm import tqdm
from sklearn import metrics
import time
import torch
from collections import defaultdict


# 自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.ex_list = []
        with open('../dataset/' + file, "r", encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                query = sample["query"]
                title = sample["title"]
                relevant = int(sample["label"])
                self.ex_list.append((query, title, relevant))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, index):
        query, title, relevant = self.ex_list[index]

        inputs = self.tokenizer.encode_plus(
            query, title,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(relevant, dtype=torch.float)
        }


# 各个类别性能度量的函数
def category_performance_measure(labels_right, labels_pred, num_label=3):
    text_labels = [i for i in range(num_label)]
    # text_labels = list(set(labels_right))

    TP = dict.fromkeys(text_labels, 0)  # 预测正确的各个类的数目
    TP_FP = dict.fromkeys(text_labels, 0)  # 测试数据集中各个类的数目
    TP_FN = dict.fromkeys(text_labels, 0)  # 预测结果中各个类的数目

    label_dict = defaultdict(list)
    for num in range(num_label):
        label_dict[num].append(str(num))

    # 计算TP等数量
    for i in range(0, len(labels_right)):
        TP_FP[labels_right[i]] += 1
        TP_FN[labels_pred[i]] += 1
        if labels_right[i] == labels_pred[i]:
            TP[labels_right[i]] += 1
    # 计算准确率P，召回率R，F1值
    for key in TP_FP:
        P = float(TP[key]) / float(TP_FP[key] + 1e-9)
        R = float(TP[key]) / float(TP_FN[key] + 1e-9)
        F1 = P * R * 2 / (P + R) if (P + R) != 0 else 0
        print("%s:\t P:%f\t R:%f\t F1:%f" % (key, P, R, F1))


# 模型评估
def evaluate_accuracy(args, data_iter, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    y_pred_, y_true_ = [], []
    for data in tqdm(data_iter):
        # If device is the GPU, copy the data to the GPU.
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        net.eval()
        y_hat_ = net(ids, mask, token_type_ids)
        with torch.no_grad():
            targets = targets.long()
            # [[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            acc_sum += torch.sum((torch.argmax(y_hat_, dim=1) == targets))
            y_pred_.extend(torch.argmax(y_hat_, dim=1).cpu().numpy().tolist())
            y_true_.extend(targets.cpu().numpy().tolist())
            n += targets.shape[0]
    valid_f1 = metrics.f1_score(y_true_, y_pred_, average='macro')
    if args.cate_performance:
        category_performance_measure(y_true_, y_pred_, args.num_labels)
    return acc_sum.item()/n, valid_f1


# 模型训练
def train(net, train_iter, valid_iter, criterion, num_epochs, optimizer, device, args):
    print('training on', device)
    net.to(device)
    best_test_f1 = 0
    # 设置学习率下降策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.warmup_step, gamma=args.warmup_proportion)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=2e-06)  # 余弦退火
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        y_pred, y_true = [], []
        for data in tqdm(train_iter):
            net.train()
            optimizer.zero_grad()
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            y_hat = net(ids, mask, token_type_ids)
            loss = criterion(y_hat, targets.long())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                targets = targets.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == targets))).float()
                y_pred.extend(torch.argmax(y_hat, dim=1).cpu().numpy().tolist())
                y_true.extend(targets.cpu().numpy().tolist())
                n += targets.shape[0]
        valid_acc, valid_f1 = evaluate_accuracy(args, valid_iter, net, device)
        train_acc = train_acc_sum / n
        train_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        print('epoch %d, loss %.4f, train acc %.3f, valid acc %.3f, '
              'train f1 %.3f, valid f1 %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc, valid_acc,
                 train_f1, valid_f1, time.time() - start))
        if valid_f1 > best_test_f1:
            print('find best! save at model/best.pth')
            best_test_f1 = valid_f1
            torch.save(net.state_dict(), '../model/best.pth')
        scheduler.step()  # 更新学习率

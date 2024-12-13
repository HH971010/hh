from torch import nn
import torch as t
import torch.nn.functional as F

M=8

class MHA(nn.Module):
    def __init__(self,embedding_size,city_size):
        super(MHA, self).__init__()
        self.embedding_size=embedding_size
        self.dk = embedding_size / M
        self.city_size=city_size
        self.wq1 = nn.Linear(embedding_size, embedding_size)
        self.wk1 = nn.Linear(embedding_size, embedding_size)
        self.wv1 = nn.Linear(embedding_size, embedding_size)
        self.w1 = nn.Linear(embedding_size, embedding_size)
        # FF
        self.fw1 = nn.Linear(embedding_size, embedding_size * 4)
        self.fb1 = nn.Linear(embedding_size * 4, embedding_size)
        # Batch Normalization(BN)
        self.BN11 = nn.BatchNorm1d(embedding_size)
        self.BN12 = nn.BatchNorm1d(embedding_size)

    def forward(self,node,temp_len):

        query1 = self.wq1(node)
        query1 = t.unsqueeze(query1, dim=2)
        query1 = query1.expand(temp_len, self.city_size, self.city_size, self.embedding_size)
        key1 = self.wk1(node)
        key1 = t.unsqueeze(key1, dim=1)
        key1 = key1.expand(temp_len, self.city_size, self.city_size, self.embedding_size)
        value1 = self.wv1(node)
        value1 = t.unsqueeze(value1, dim=1)
        value1 = value1.expand(temp_len, self.city_size, self.city_size, self.embedding_size)
        x = query1 * key1
        x = x.view(temp_len, self.city_size, self.city_size, M, -1)
        x = t.sum(x, dim=4)  # u=q^T x k
        x = x / (self.dk ** 0.5)
        x = F.softmax(x, dim=2)
        x = t.unsqueeze(x, dim=4)
        x = x.expand(temp_len, self.city_size, self.city_size, M, 16)
        x = x.contiguous()
        x = x.view(temp_len, self.city_size, self.city_size, -1)
        x = x * value1
        x = t.sum(x, dim=2)  # MHA :(temp_len, city_size, embedding_size)
        x = self.w1(x)  # 得到一层MHA的结果

        x = x + node

        # 第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN11(x)
        x = x.permute(0, 2, 1)
        # x = t.tanh(x)

        # 第一层FF
        x1 = self.fw1(x)
        x1 = F.relu(x1)
        x1 = self.fb1(x1)

        x = x + x1

        # 第二个BN
        x = x.permute(0, 2, 1)
        x = self.BN12(x)
        x = x.permute(0, 2, 1)
        x1 = x  # h_i^(l) n=1
        return x1




import numpy as np
from torch import nn
import torch as t
import torch.nn.functional as F
from MHA import MHA

DEVICE= t.device("cuda" if t.cuda.is_available() else "cpu")
M=8
C = 10  # 做softmax得到选取每个点概率前，clip the result所使用的参数
bl_alpha = 0.05  # 做t-检验更新baseline时所设置的阈值
service_time_per_dem = 300  # 服务用户的时间
speed_kmph=10#车辆速度
penalty=10#超出时间窗的惩罚系数

def GetDis(s,temp_len,city_size):
    s1 = t.unsqueeze(s, dim=1)
    s1 = s1.expand(temp_len, city_size, city_size, 2)
    s2 = t.unsqueeze(s, dim=2)
    s2 = s2.expand(temp_len, city_size, city_size, 2)
    ss = s1 - s2
    dis = t.norm(ss, 2, dim=3, keepdim=True)
    # 计算两点之间在地球上的实际距离 代码来自CVRPTW-ortool
    for batch_idx in range(temp_len):
        s1 = s[batch_idx]
        for frm_idx in range(city_size):
            for to_idx in range(city_size):
                if frm_idx != to_idx:
                    lat1 = s1[frm_idx][0]
                    lon1 = s1[frm_idx][1]
                    lat2 = s1[to_idx][0]
                    lon2 = s1[to_idx][1]
                    lon1, lat1, lon2, lat2 = map(np.radians, [lon1.cpu(), lat1.cpu(), lon2.cpu(), lat2.cpu()])

                    # haversine formula
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = (
                            np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
                    c = 2 * np.arcsin(np.sqrt(a))

                    # 6367 km is the radius of the Earth
                    km = 6367 * c

                    dis[batch_idx][frm_idx][to_idx] = km
    return dis

    # dis表示任意两点间的距离 (temp_len, city_size, city_size, 1)

class Model(nn.Module):
    def __init__(self,embedding_size,city_size):
        super().__init__()
        self.dk=embedding_size/M
        self.city_size=city_size
        self.embedding_size=embedding_size
        self.embedding1 = nn.Linear(3, embedding_size)  # 用于客户点的坐标加容量需求的embedding
        self.embedding2=nn.Linear(2,embedding_size)
        self.embedding_p = nn.Linear(2, embedding_size)  # 用于仓库点的坐标embedding
        #encoder
        self.MHA1=MHA(embedding_size,city_size)
        self.MHA2=MHA(embedding_size,city_size)
        self.MHA3=MHA(embedding_size,city_size)
        self.MHA4 = MHA(embedding_size, city_size)
        self.MHA5 = MHA(embedding_size, city_size)
        self.MHA6 = MHA(embedding_size, city_size)
        #decoder
        self.wq = nn.Linear(embedding_size * 2 + 1, embedding_size)
        self.wk = nn.Linear(embedding_size, embedding_size)
        self.wv = nn.Linear(embedding_size, embedding_size)
        self.w = nn.Linear(embedding_size, embedding_size)

        self.q = nn.Linear(embedding_size, embedding_size)
        self.k = nn.Linear(embedding_size, embedding_size)


#每次同时训练batch个vrp问题
    def forward(self, data,l, train):  # s坐标，d需求，l初始容量, train==0表示无需计算梯度且使用greedy方法，train>0表示需要计算梯度且使用sampling方法
        # s :[temp_len x seq_len x 2]
        s, d, start_times_epoch, stop_times_epoch = data
        s = s.to(DEVICE)
        d = d.to(DEVICE)
        start_time = start_times_epoch.to(DEVICE)
        stop_time = stop_times_epoch.to(DEVICE)
        temp_len = len(s)  # 使用多GPU运行时数据集会被平均分给GPU，所以要重新确定数据集的维度
        temp_len = int(temp_len)
        mask_size = t.LongTensor(temp_len).to(DEVICE)  # 用于标号，方便后面两点间的距离计算
        for i in range(temp_len):
            mask_size[i] = self.city_size * i
        dis=GetDis(s,temp_len,self.city_size).to(DEVICE)
        pro = t.FloatTensor(temp_len, self.city_size * 2).to(DEVICE)  # 每个点被选取时的选取概率,将其连乘可得到选取整个路径的概率
        seq = t.LongTensor(temp_len, self.city_size * 2).to(DEVICE)  # 选取的序列
        index = t.LongTensor(temp_len).to(DEVICE)  # 当前车辆所在的点
        tag = t.ones(temp_len * self.city_size).to(DEVICE) #城市是否被访问
        distance = t.zeros(temp_len).to(DEVICE)  # 总距离
        rest = t.LongTensor(temp_len, 1, 1).to(DEVICE)  # 车的剩余容量
        dd = t.LongTensor(temp_len, self.city_size).to(DEVICE)  # 客户需求
        cur_time=t.zeros(temp_len).to(DEVICE)#记录当前时间
        total_cost=t.zeros(temp_len).to(DEVICE)#记录总cost，包括旅行距离和时间窗之外服务带来的惩罚
        rest[:, 0, 0] = l
        dd[:, :] = d[:, :, 0]  # 需求
        index[:] = 0
        start_time = t.unsqueeze(start_time, dim=2)
        stop_time = t.unsqueeze(stop_time, dim=2)
        #sss = t.cat([s, d.float(),start_time,stop_time], dim=2)  # [temp_len x seq_len x 3] 坐标与容量需求与时间窗拼接
        s_d=t.cat([s, d.float()], dim=2)#坐标与容量需求拼接
        TW=t.cat([start_time, stop_time], dim=2)

        # node为所有点的的初始embedding
        s_d_node = self.embedding1(s_d) # 客户点embedding坐标加容量需求加时间窗
        TW_node=self.embedding2(TW)
        s_d_node[:, 0, :] = self.embedding_p(s[:, 0, :])  # 仓库点只embedding坐标

        # encoder部分,node1（用户位置和需求）、node2（时间窗）分别经过三个M=8的MHA

        s_d_x1=self.MHA1(s_d_node,temp_len)
        s_d_x2=self.MHA2(s_d_x1,temp_len)
        s_d_x3=self.MHA3(s_d_x2,temp_len)
        s_d_x3 = s_d_x3.contiguous()

        TW_x1=self.MHA4(TW_node,temp_len)
        TW_x2 = self.MHA5(TW_x1, temp_len)
        TW_x3 = self.MHA6(TW_x2, temp_len)
        TW_x3 = TW_x3.contiguous()
        sum_node=s_d_x3+TW_x3
        avg = t.mean(sum_node, dim=1)  # 最后将所有节点的嵌入信息取平均得到整个图的嵌入信息，(temp_len, embedding_size)

        # decoder部分
        x=sum_node
        for i in range(self.city_size * 2):  # decoder输出序列的长度不超过city_size * 2
            flag = t.sum(dd, dim=1)  # dd:(temp_len, city_size)
            f1 = t.nonzero(flag > 0).view(-1)  # 取得需求不全为0的batch号
            f2 = t.nonzero(flag == 0).view(-1)  # 取得需求全为0的batch号

            if f1.size()[0] == 0:  # batch所有需求均为0
                pro[:, i:] = 1  # pro:(temp_len, city_size*2)
                seq[:, i:] = 0  # swq:(temp_len, city_size*2)
                temp = dis.view(-1, self.city_size, 1)[
                    index + mask_size]  # dis:任意两点间的距离 (temp_len, city_size, city_size, 1) temp:(temp_len, city_size,1)
                distance = distance + temp.view(-1)[mask_size]  # 加上当前点到仓库的距离
                break

            ind = index + mask_size
            tag[ind] = 0  # tag:(temp_len*city_size)
            start = x.view(-1, self.embedding_size)[ind]  # (temp_len, embedding_size)，每个batch中选出一个节点
            end = rest[:, :, 0]  # (temp_len, 1)
            end = end.float()  # 车上剩余容量

            graph = t.cat([avg, start, end], dim=1)  # 结合图embedding，当前点embedding，车剩余容量: (temp_len,embedding_size*2 + 1)_
            query = self.wq(graph)  # (temp_len, embedding_size)
            query = t.unsqueeze(query, dim=1)
            query = query.expand(temp_len, self.city_size, self.embedding_size)
            key = self.wk(x)
            value = self.wv(x)
            temp = query * key
            temp = temp.view(temp_len, self.city_size, M, -1)
            temp = t.sum(temp, dim=3)  # (temp_len, city_size, M)
            temp = temp / (self.dk ** 0.5)

            mask = tag.view(temp_len, -1, 1) < 0.5  # 访问过的点tag=0
            mask1 = dd.view(temp_len, self.city_size, 1) > rest.expand(temp_len, self.city_size, 1)  # 客户需求大于车剩余容量的点

            flag = t.nonzero(index).view(-1)  # 在batch中取得当前车不在仓库点的batch号
            mask = mask + mask1  # mask:(temp_len x city_size x 1)
            mask = mask > 0
            mask[f2, 0, 0] = 0  # 需求全为0则使车一直在仓库
            if flag.size()[0] > 0:  # 将有车不在仓库的batch的仓库点开放
                mask[flag, 0, 0] = 0

            mask = mask.expand(temp_len, self.city_size, M)
            temp.masked_fill_(mask, -float('inf'))
            temp = F.softmax(temp, dim=1)
            temp = t.unsqueeze(temp, dim=3)
            temp = temp.expand(temp_len, self.city_size, M, 16)
            temp = temp.contiguous()
            temp = temp.view(temp_len, self.city_size, -1)
            temp = temp * value
            temp = t.sum(temp, dim=1)
            temp = self.w(temp)  # hc,(temp_len,embedding_size)

            query = self.q(temp)
            key = self.k(x)  # (temp_len, city_size, embedding_size)
            query = t.unsqueeze(query, dim=1)  # (temp_len, 1 ,embedding_size)
            query = query.expand(temp_len, self.city_size, self.embedding_size)  # (temp_len, city_size, embedding_size)
            temp = query * key
            temp = t.sum(temp, dim=2)
            temp = temp / (self.dk ** 0.5)
            temp = t.tanh(temp) * C  # (temp_len, city_size)

            mask = mask[:, :, 0]
            temp.masked_fill_(mask, -float('inf'))
            p = F.softmax(temp, dim=1)  # 得到选取每个点时所有点可能被选择的概率

            indexx = t.LongTensor(temp_len).to(DEVICE)
            if train != 0:
                indexx[f1] = t.multinomial(p[f1], 1)[:, 0]  # 按sampling策略选点
            else:
                indexx[f1] = (t.max(p[f1], dim=1)[1])  # 按greedy策略选点

            indexx[f2] = 0
            p = p.view(-1)
            pro[:, i] = p[indexx + mask_size]
            pro[f2, i] = 1
            rest = rest - (dd.view(-1)[indexx + mask_size]).view(temp_len, 1, 1)  # 车的剩余容量
            service_time=t.zeros(temp_len).to(DEVICE)
            dd = dd.view(-1)
            for i1 in range(temp_len):
                service_time[i1]=dd[indexx[i1]+mask_size[i1]]*service_time_per_dem#计算服务时间
            dd[indexx + mask_size] = 0
            dd = dd.view(temp_len, self.city_size)#用户需求更新

            temp = dis.view(-1, self.city_size, 1)[index + mask_size]
            total_cost = total_cost + temp.view(-1)[indexx + mask_size]
            cur_time=cur_time+(temp.view(-1)[indexx + mask_size]/ (speed_kmph * 1.0 / 60**2))

            #计算早到迟到惩罚
            for i2 in range(temp_len):
                if start_time[i2,indexx[i2]]>cur_time[i2]:
                    total_cost[i2]=total_cost[i2]+t.floor((start_time[i2,indexx[i2]]-cur_time[i2])*0.2)
                if stop_time[i2,indexx[i2]]<cur_time[i2]:
                    total_cost[i2] = total_cost[i2] + t.floor((cur_time[i2]-stop_time[i2, indexx[i2]] ))
            cur_time=cur_time+service_time#服务结束，更新时间

            mask3 = indexx == 0
            mask3 = mask3.view(temp_len, 1, 1)
            rest.masked_fill_(mask3, l)  # 车回到仓库将容量设为初始值

            index = indexx
            seq[:, i] = index[:]

        if train == 0:
            seq = seq.detach()
            pro = pro.detach()
            total_cost = total_cost.detach()

        return seq, pro, total_cost  # 被选取的点序列,每个点被选取时的选取概率,这些序列的总路径长度


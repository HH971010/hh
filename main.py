#将时间单独提取出来经过MHA网络的模型

import random
import torch as t
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import time
import numpy as np
import os
from scipy.stats import ttest_rel
from torch.utils.data import Dataset, DataLoader
from CVRPTW import CVRPTW
from Model import Model
from tensorboard_logger import Logger as TbLogger
DEVICE= t.device("cuda" if t.cuda.is_available() else "cpu")
t.manual_seed(111)
random.seed(111)
np.random.seed(111)
output_dir = 'output'
save_dir = os.path.join(os.getcwd(), output_dir)
bl_alpha = 0.05  # 做t-检验更新baseline时所设置的阈值
embedding_size = 128
city_size = 20  # 节点总数
tb_logger = None
tb_logger = TbLogger(os.path.join("logs", "{}_{}".format(CVRPTW, 20), "CVRPTW"))
batch = 64  # 每个batch的算例数
times = 70  # 训练中每个epoch所需的训练batch数
epochs = 50  # 训练的epoch总数
test2save_times = 20  # 训练过程中每次保存模型所需的测试batch数
min = 1000000000 # 当前已保存的所有模型中测试用例cost的最小值
l=50

is_train = True  # 是否训练
# 测试
test_times = 100  # 测试时所需的batch总数
test_is_sample = False  # 测试时是否要使用sampling方法，否则使用greedy方法

#数据可视化
def log_values(cost,bs_cost, epoch, step, tb_logger):
    avg_cost = t.mean(cost)
    avg_bs_cost=t.mean(bs_cost)
    # Log values to tensorboard
    step=epoch*times+step
    tb_logger.log_value('avg_cost', avg_cost, step)
    tb_logger.log_value('avg_bs_cost', avg_bs_cost, step)


net1 = Model(embedding_size,city_size).to(DEVICE)
net2 = Model(embedding_size,city_size).to(DEVICE)
if t.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    print("use GPU")
    net1 = nn.DataParallel(net1)
    net2=nn.DataParallel(net2)

#net2.load_state_dict(net1.state_dict())
# 训练部分
if is_train is True:
    opt = optim.Adam(net1.parameters(), 0.0001)

    bs_Dataset = DataLoader(dataset=CVRPTW(batch * test2save_times, city_size),
                             batch_size=batch, shuffle=True,num_workers=3)
    train_Dataset=DataLoader(dataset=CVRPTW(batch * times,city_size),
                             batch_size=batch,shuffle=True,num_workers=3)

    for epoch in range(epochs):
        i=0
        for data in train_Dataset:
            clock1=time.time()
            seq2, pro2, cost2 = net2(data,l, 0)  # baseline return seq, pro, distance
            seq1, pro1, cost1 = net1(data,l, 2)
            clock2=time.time()
            clock=clock2-clock1
            log_values(cost2,cost1,epoch,i, tb_logger)
            # print('nn_output_time={}'.format(t2 - t1))
            ###################################################################
            # 带baseline的策略梯度训练算法,cost2作为baseline
            pro = t.log(pro1)
            loss = t.sum(pro, dim=1)
            score = cost1 - cost2  # advantage reward(优势函数)

            score = score.detach()
            loss = score * loss
            loss = t.sum(loss) / batch  # 最终损失函数

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(net1.parameters(), 1)
            opt.step()
            print('epoch={},i={},mean_cost1={},mean_cost2={},time={}'.format(epoch, i, t.mean(cost1), t.mean(
                cost2),clock))  # ,'disloss:',t.mean((dis1-dis2)*(dis1-dis2)), t.mean(t.abs(dis1-dis2)), nan)

            # OneSidedPairedTTest(做t-检验看当前Sampling的解效果是否显著好于greedy的解效果,如果是则更新使用greedy策略作为baseline的net2参数)
            if (cost1.mean() - cost2.mean()) < 0:
                tt, pp = ttest_rel(cost1.cpu().numpy(), cost2.cpu().numpy())
                p_val = pp / 2
                assert tt < 0, "T-statistic should be negative"
                if p_val < bl_alpha:
                    print('Update baseline')
                    net2.load_state_dict(net1.state_dict())

            # 每隔xxx步做测试判断结果有没有改进，如果改进了则把当前模型保存下来
            if (i + 1) % 10 == 0:
                cost = t.zeros(1).to(DEVICE)

                for bs_data in bs_Dataset:
                    b_seq, b_pro2, b_cost = net1(bs_data,l,0)
                    cost = cost + t.mean(b_cost)
                cost = cost / test2save_times
                if cost < min:
                    t.save(net1.state_dict(), os.path.join(save_dir,
                                                           'epoch{}-i{}-cost_{:.5f}.pt'.format(
                                                               epoch, i, cost.item())))
                    min = cost.int()
                print('mincost=', cost.item())
            i=i+1
# 测试部分
else:
    # 按照greedy策略测试
    if test_is_sample is False:
        test_Dataset = DataLoader(dataset=CVRPTW(batch * test_times, city_size),
                                batch_size=batch, shuffle=True, num_workers=3)
        sum_dis = t.zeros(1).to(DEVICE)
        sum_clock = 0  # 记录生成解的总时间
        net1.load_state_dict(t.load('output/epoch13-i29-cost_2134579.50000.pt'))
        net1.eval()
        for test_data in test_Dataset:

            t.cuda.empty_cache()
            clock1 = time.time()
            seq, pro, cost = net1(test_data, l, 0)
            clock2 = time.time()
            deta_clock = clock2 - clock1
            sum_clock = sum_clock + deta_clock

            print("dis:{},deta_clock:{}".format(t.mean(cost), deta_clock))
            sum_dis = sum_dis + t.mean(cost)
        mean_dis = sum_dis / test_times
        mean_clock = sum_clock / test_times
        print("mean_dis:{},mean_clock:{}".format(mean_dis, mean_clock))

    # 按照sampling策略测试
    else:
        test_Dataset = DataLoader(dataset=CVRPTW(test_times, city_size),
                                  batch_size=batch, shuffle=True, num_workers=3)
        all_repeat_size = 1280
        num_batch_repeat = all_repeat_size // batch

        sum_dis = t.zeros(1).to(DEVICE)
        sum_clock = 0  # 记录生成解的总时间
        for test_data in test_Dataset:
            t.cuda.empty_cache()
            available_seq = []
            available_dis = []
            deta_clock = 0
            for _ in range(num_batch_repeat):
                clock1 = time.time()
                seq, pro, dis = net1(test_data, l, 1)
                clock2 = time.time()

                mini_deta_clock = clock2 - clock1
                deta_clock = deta_clock + mini_deta_clock
                for j in range(batch):
                    available_seq.append(seq[j])
                    available_dis.append(dis[j])

            available_seq = t.stack(available_seq)
            available_dis = t.stack(available_dis)
            mincost, mincost_index = t.min(available_dis, 0)
            sum_cost = sum_cost + mincost
            sum_clock = sum_clock + deta_clock
            print("mincost:{},deta_clock:{}".format(mincost, deta_clock))
        mean_cost = sum_cost / test_times
        mean_clock = sum_clock / test_times
        print("mean_cost:{},mean_clock:{}".format(mean_cost.item(), mean_clock))

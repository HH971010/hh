import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

#创建CVRPTW问题数据集
time_horizon = 24 * 60 ** 2  # A 24 hour period.
min_tw = 2
max_tw = 6
def make_time_windows(num=100,city_size=10):
    # 加时间窗 代码来自CVRPTW-ortool

    # The customers demand min_tw to max_tw hour time window for each
    # delivery
    def time_windows(n,city_size):
        start_times = torch.rand(n, city_size)
        stop_times = torch.rand(n, city_size)
        for i in range(n):
            time_windows_temp = np.random.random_integers(min_tw * 3600, max_tw * 3600,
                                                          city_size)  # 秒为单位
            # The last time a delivery window can start
            latest_time = time_horizon - time_windows_temp
            start_times_temp = [None for o in time_windows_temp]
            stop_times_temp = [None for o in time_windows_temp]
            # Make random timedeltas, nominaly from the start of the day.
            for idx in range(city_size):
                stime = int(np.random.random_integers(latest_time[idx]))
                start_times_temp[idx] = stime
                stop_times_temp[idx] = (
                        start_times_temp[idx] + int(time_windows_temp[idx]))
            start_times[i, :] = torch.tensor(start_times_temp)
            stop_times[i, :] = torch.tensor(stop_times_temp)
        return start_times,stop_times

    return time_windows(num,city_size)

class CVRPTW(Dataset):

    def __init__(self, size, city_size):  # RandomDataset类的构造器
        self.len=size
        self.S = torch.rand(size, city_size, 2)  # 坐标0~1之间
        self.D = np.random.randint(1, 10, size=(size, city_size, 1))  # 所有客户的需求
        self.D = torch.LongTensor(self.D)
        self.D[:, 0, 0] = 0  # 仓库点的需求为0
        self.t_start_times,self.t_stop_times = make_time_windows(size,city_size)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        S= self.S[index]
        D=self.D[index]
        start_times=self.t_start_times[index]
        stop_times=self.t_stop_times[index]
        return S,D,start_times,stop_times



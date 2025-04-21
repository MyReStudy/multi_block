import time

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from torch.utils.data import Dataset
import torch
import os
import pickle
import random
import torch.nn.functional as F
from options import get_options
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)



class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"
        order_after_delete = dataset['order_after_delete']
        orders_coords_std = dataset['orders_coords_std']
        d = order_after_delete.gather(1, pi)
        next_d = d[:, 1:]
        prev_d = d[:, :-1]
        distance_matrix = np.load('files/distance_matrix.npy', allow_pickle=True)
        # required_info = np.load('files/required_info.npy', allow_pickle=True)
        cost = []
        batch_size = pi.size(0)
        for b in range(batch_size):
            cost_tmp = 0
            prev_d_item = prev_d[b].tolist()
            next_d_item = next_d[b].tolist()
            for i in range(len(prev_d_item)):
                distance = distance_matrix[prev_d_item[i]][next_d_item[i]]
                cost_tmp += distance
            cost_tmp += distance_matrix[next_d_item[-1]][prev_d_item[0]]
            cost.append(float(cost_tmp))
        cost = torch.tensor(cost, device='cuda')
        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            require_info = np.load('files/required_info.npy', allow_pickle=True)  # 剩余部分
            opts = get_options()
            aisle_num, cross_num = opts.aisle_num, opts.cross_num
            time1 = time.time()
            self.data = [self.generate_data(require_info, aisle_num, cross_num) for i in range(num_samples)]
            time2 = time.time()
            print(f'生成数据时间:{time2-time1}')
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def generate_data(self, require_info, aisle_num, cross_num):
        opts = get_options()
        goods_low = opts.goods_low
        goods_high = opts.goods_high
        # time3 = time.time()
        num_points = random.randint(goods_low, goods_high)
        orders = [0] + random.sample(require_info[7][1:].tolist(), num_points)  # 从require_info[7]里面随机抽取一些点
        orders_info = require_info[:, orders]  # 把这些点对应的信息取出来
        # time4 = time.time()
        # print(f'生成预处理前的数据耗时{time4-time3}')
        time5 = time.time()
        group_keys = orders_info[3]  # 取出来所在巷道
        group_values = orders_info[7]  # 取出来他们的index
        sorted_indices = np.argsort(group_values)  # 按 require_info[7] 升序排列 获得对应的index
        group_keys_sorted = group_keys[sorted_indices]  # 获取他们所在巷道
        group_values_sorted = group_values[sorted_indices]  # 获取他们的index对应的require_info[7]取值

        _, group_start = np.unique(group_keys_sorted, return_index=True)  # 获取unique的巷道在group_keys_sorted开始的位置
        group_sizes = np.diff(np.append(group_start, len(group_keys_sorted)))  # 看看每个巷道里有几个点

        order_after_delete = np.array([])  # 存最终删减完的数据

        for i, (start, size) in enumerate(zip(group_start, group_sizes)):
            if i==len(group_start)-1:  # i是最后一个
                goods_in_aisle = group_values_sorted[group_start[i]:]  # 取出来这个巷道里所有的货物
            else:
                goods_in_aisle = group_values_sorted[group_start[i]:group_start[i + 1]]
            if size > 3:
                min_index = np.argmin(goods_in_aisle)  # 最小的idx
                max_index = np.argmax(goods_in_aisle)  # 最大的idx
                diffs = np.abs(np.diff(goods_in_aisle))  # 巷道里不同idx的差值
                max_diff_idx = np.argmax(diffs)  # 最大的差值所在的位置
                max_diff_indices = [max_diff_idx, max_diff_idx + 1]  # 最大差值的两个idx
                result_indices = np.unique([min_index, max_index] + max_diff_indices)
                order_after_delete = np.append(order_after_delete, goods_in_aisle[result_indices]).astype(int) # 把删完的点加进去
            else:
                order_after_delete = np.append(order_after_delete, goods_in_aisle).astype(int)

        # locations = np.argwhere(require_info[7]==order_after_delete)
        orders_coords_std = require_info[6][order_after_delete]  # 把这些点对应的坐标取出来
        padding_length = 4 * aisle_num * (cross_num-1) - len(orders_coords_std) + 1
        orders_coords_std = torch.tensor(orders_coords_std.tolist(), device='cuda')
        orders_coords_std = F.pad(orders_coords_std, (0, 0, 0, padding_length), "constant", 0)
        order_after_delete = torch.tensor(order_after_delete.tolist(), device='cuda')
        order_after_delete = F.pad(order_after_delete, (0, padding_length), "constant", 0)


        # time6 = time.time()
        # print(f'生成数据耗时{time6-time3}')
        return {'order_after_delete': order_after_delete,
                'orders_coords_std': orders_coords_std}

    def delete_order(self, orders_info):
        indices = orders_info.index.tolist()
        if len(indices) <= 3:
            return indices  # <=3 的情况直接返回
        else:
            # 找最小索引和最大索引
            min_index = min(indices)
            max_index = max(indices)

            # 找出与相邻索引的差最大的索引
            diffs = [abs(indices[i] - indices[i + 1]) for i in range(len(indices) - 1)]
            max_diff_idx = diffs.index(max(diffs))
            max_diff_indices = [indices[max_diff_idx], indices[max_diff_idx + 1]]

            # 合并所有需要保留的索引，并去重
            result = list(set([min_index, max_index] + max_diff_indices))
            return result
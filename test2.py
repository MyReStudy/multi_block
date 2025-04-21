import random
import time

import pandas as pd
import torch
import torch.nn.functional as F
from gurobipy import *
import numpy as np
from matplotlib import pyplot as plt

from options import get_options
from simple_tsp import tsp_solver
from utils import load_model
from 精确算法 import exact_solver
from 蚁群算法 import ACO_solver

random.seed(0)


deep_l1=[]
deep_l2 = []
exact_l = []
seed = []
aco_l = []
ins = []
model, _ = load_model('pretrained/warehouse_tsp/')

# opts = get_options()
df = pd.DataFrame(columns=['aisle_num','corss_num','avg_time_deep', 'avg_time_exact','avg_time_aco','status'])


def generate_data(require_info,goods_low, goods_high):
    # time3 = time.time()
    num_points = random.randint(goods_low, goods_high)
    orders = random.sample(require_info[7][1:].tolist(), num_points)  # 从require_info[7]里面随机抽取一些点
    ins.append(orders)
    orders_info = require_info[:, orders]  # 把这些点对应的信息取出来
    group_keys = orders_info[3]  # 取出来所在巷道
    group_values = orders_info[7]  # 取出来他们的index
    sorted_indices = np.argsort(group_values)  # 按 require_info[7] 升序排列 返回索引
    group_keys_sorted = group_keys[sorted_indices]  # 获取排序后所在巷道
    group_values_sorted = group_values[sorted_indices]  # 获取排序后的index orders_info[7]升序排序
    all_loc_before_delete = orders_info[6][sorted_indices]  # 排序后的坐标
    print(all_loc_before_delete)

    _, group_start = np.unique(group_keys_sorted, return_index=True)  # 获取unique的巷道在group_keys_sorted开始的位置 每个巷道开始的位置
    group_sizes = np.diff(np.append(group_start, len(group_keys_sorted)))  # 看看每个巷道里有几个点

    order_after_delete = np.array([])  # 存最终删减完的数据

    neighbor_nodes = {}
    for i, (start, size) in enumerate(zip(group_start, group_sizes)):
        if i == len(group_start) - 1:  # i是最后一个
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
            order_after_delete = np.append(order_after_delete, goods_in_aisle[result_indices]).astype(int)  # 把删完的点加进去
            if (goods_in_aisle[min_index]!=goods_in_aisle[max_diff_idx]):
                    # and (goods_in_aisle[min_index]!=0 and goods_in_aisle[max_diff_idx]!=0)):
                if goods_in_aisle[min_index]==0:
                    # neighbor_nodes[goods_in_aisle[min_index]] = goods_in_aisle[max_diff_idx]
                    pass
                else:
                    neighbor_nodes[goods_in_aisle[min_index]] = goods_in_aisle[max_diff_idx]
                    neighbor_nodes[goods_in_aisle[max_diff_idx]] = goods_in_aisle[min_index]
            if goods_in_aisle[max_index]!=goods_in_aisle[max_diff_idx+1]:
                # and (goods_in_aisle[max_index]!=0 and goods_in_aisle[max_diff_idx+1]!=0):
                neighbor_nodes[goods_in_aisle[max_index]] = goods_in_aisle[max_diff_idx+1]
                neighbor_nodes[goods_in_aisle[max_diff_idx+1]] = goods_in_aisle[max_index]
        else:
            order_after_delete = np.append(order_after_delete, goods_in_aisle).astype(int)

    order_after_delete = np.insert(order_after_delete,0,0)
    # locations = np.argwhere(require_info[7]==order_after_delete)
    orders_coords_std = require_info[6][order_after_delete]  # 把这些点对应的坐标取出来
    orders_coords_std = torch.tensor(orders_coords_std.tolist(), device='cuda')
    order_after_delete = torch.tensor(order_after_delete.tolist(), device='cuda')
    coords_orig = orders_coords_std
    padding_length = 4 * aisle_num * (cross_num-1) - len(orders_coords_std)+1
    orders_coords_std = F.pad(orders_coords_std, (0, 0, 0, padding_length), "constant", 0)
    order_after_delete = F.pad(order_after_delete, (0, padding_length), "constant", 0)

    # time6 = time.time()
    # print(f'生成数据耗时{time6-time3}')
    return {'order_after_delete': order_after_delete,
            'orders_coords_std': orders_coords_std,
            'coords_orig':coords_orig,
            'neighbor_nodes':neighbor_nodes,
            'all_loc_before_delete':all_loc_before_delete,
            'orders_idx':orders}

def get_info_aisle_vertex(coords_all_after_delete, require_len_for_batch, steiner_len_for_batch, aisle_num, cross_num):
    '''
    Args:
        coords_all_after_delete: 所有点的坐标 包括depot
        require_len_for_batch:
        steiner_len_for_batch:
        aisle_num:
        cross_num:
        aisle_length:

    Returns: 所有batch每个巷道内点idx，所有batch每个点存在哪个巷道

    '''

    aisle_vertex_info = []  # 存每个巷道内有哪些点 batch,巷道编号,点的idx [[[7,3,4,0][2,5,8,9]]]
    where_is_vertex = []  # 存每个点在哪个巷道里面 batch,点的idx,存在哪个巷道 [[0,1,2,1],[2,1,1]]

    require_len = require_len_for_batch

    for i in range(coords_all_after_delete.size(0)):
        aisle_vertex_info.append([])
        vertex_poition_batch = []
        required_len_for_instancei = require_len[i]  # 这个instance的必访点数量
        steiner_len_for_steineri = steiner_len_for_batch[i]  # 这个instance Steiner点数量 为了不对后面padding的内容进行距离矩阵和邻居的计算

        for _ in range(required_len_for_instancei+steiner_len_for_steineri):
            vertex_poition_batch.append(-1)

        coords = coords_all_after_delete[i].tolist()[0:required_len_for_instancei+steiner_len_for_steineri]

        points_dict_x = {}
        for point_index, (x, y) in enumerate(coords):
            x_val = x  # x的取值
            if x_val in points_dict_x:
                points_dict_x[x_val].append((point_index, y))  # [x]: 点的序号, y
            else:
                points_dict_x[x_val] = [(point_index, y)]  # points_dict_x[x坐标] = [(点的idx, y坐标)]
        # 对字典中每个键对应的值按照元组的第二个元素升序排列
        sorted_dict = {key: sorted(value, key=lambda x: x[1]) for key, value in points_dict_x.items()}  # x取值相同，y升序排列
        # 将排序后每个键对应值的元组的第一个元素取出来，以列表形式存储
        first_elements = {key: [item[0] for item in value] for key, value in sorted_dict.items()}  # 对应的点的idx
        for j in range(0, aisle_num * (cross_num - 1)):  # 遍历所有的巷道
            bottom_steiner = j  # 巷道下面的steiner点
            top_steiner = j + aisle_num  # 巷道上面的Steiner点

            x_axis = coords[bottom_steiner][0]  # 获取下面steiner点的x坐标
            vertex_in_aisle = first_elements[x_axis]  # 把巷道内所有的点取出来
            bottom_steiner_idx = vertex_in_aisle.index(bottom_steiner)  # 下面的Steiner点的idx 不是在全部坐标里的idx 而是在当前aisle的Idx
            top_steiner_idx = vertex_in_aisle.index(top_steiner)  # 上面的Steiner点的Idx 不是在全部坐标里的idx 而是在当前aisle的Idx
            require_in_aisle = vertex_in_aisle[bottom_steiner_idx+1:top_steiner_idx]  # 巷道内所有必须访问点的idx
            steiner_require_in_aisle = vertex_in_aisle[bottom_steiner_idx:top_steiner_idx+1]

            aisle_vertex_info[i].append(steiner_require_in_aisle)
            for v in require_in_aisle:
                vertex_poition_batch[v] = j  # 没必要知道Steiner点在哪
        where_is_vertex.append(vertex_poition_batch)
    return aisle_vertex_info, where_is_vertex

def exact_solver1(aisle_vertex_info,coords, aisle_num, corss_num):
    v_aisle = aisle_num
    c_aisle = corss_num

    neighbors = {}
    for x in aisle_vertex_info:
        for j in x:
            neighbors[j] = []
    for x in aisle_vertex_info:
        for j in range(1, len(x)):
            neighbors[x[j]].append(x[j - 1])
            neighbors[x[j - 1]].append(x[j])
    aisles_num_in_block = v_aisle
    for cross_aisle in range(0, c_aisle):
        for aisle in range(0, v_aisle - 1):
            neighbors[aisle + aisles_num_in_block * cross_aisle].append(aisle + aisles_num_in_block * cross_aisle + 1)

    for cross_aisle in range(0, c_aisle):
        for aisle in range(0, v_aisle - 1):
            neighbors[aisle + aisles_num_in_block * cross_aisle + 1].append(aisle + aisles_num_in_block * cross_aisle)

    edge_dict = {}
    for x in neighbors:
        for y in neighbors[x]:
            edge_dict[(x, y)] = abs(coords[x][0] - coords[y][0]) + abs(coords[x][1] - coords[y][1])

    # required_vertex = [i for i in range(12,36)]
    # for i in range
    # edge_dict = 1
    # neighbors = 2
    steiner_vertex = [i for i in range(0, v_aisle * c_aisle)]
    required_vertex = [i for i in range(v_aisle * c_aisle, len(coords))]
    all_vertex = steiner_vertex + required_vertex
    time1 = time.time()
    mdl = Model('STSP')
    mdl.Params.TimeLimit = 300
    mdl.setParam('OutputFlag', 0)

    edge = list(edge_dict.keys())  # 所有的边
    N = len(required_vertex)  # 必访点数量

    x = {}
    y = {}

    for e in edge:
        x[e] = mdl.addVar(vtype=GRB.BINARY, name='x_' + str(e[0]) + '_' + str(e[1]))
        y[e] = mdl.addVar(vtype=GRB.CONTINUOUS, name='y_' + str(e[0]) + '_' + str(e[1]))

    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(x[e] * edge_dict[e] for e in edge))

    mdl.addConstrs(quicksum(x[i, j] for j in neighbors[i]) >= 1 for i in required_vertex)

    for i in all_vertex:
        expr1 = LinExpr(0)
        expr2 = LinExpr(0)
        for j in neighbors[i]:
            expr1.addTerms(1, x[i, j])
            expr2.addTerms(1, x[j, i])
        mdl.addConstr(expr1 == expr2)
        expr1.clear()
        expr2.clear()

    for i in required_vertex[1:]:
        expr3 = LinExpr(0)
        expr4 = LinExpr(0)
        for j in neighbors[i]:
            expr3.addTerms(1, y[i, j])
            expr4.addTerms(1, y[j, i])
        mdl.addConstr(-expr3 - 1 == -expr4)
        expr3.clear()
        expr4.clear()

    for i in steiner_vertex:
        expr5 = LinExpr(0)
        expr6 = LinExpr(0)
        for j in neighbors[i]:
            expr5.addTerms(1, y[i, j])
            expr6.addTerms(1, y[j, i])
        mdl.addConstr(expr5 == expr6)
        expr5.clear()
        expr6.clear()

    for e in edge:
        mdl.addConstr(y[e[0], e[1]] <= N * x[e[0], e[1]])

    mdl.optimize()  # 优化
    obj_res = mdl.getObjective().getValue()
    time2 = time.time()
    # # 下面开始画图
    # gurobi_solution = {}
    # # 遍历所有变量，找出名字以 'x' 开头并且值为1的变量
    # for var in mdl.getVars():
    #     if var.varName.startswith('x') and var.x > 0.5:  # 判断是否是路径，并且值为 1
    #         # 解析出节点 i 和 j
    #         var_name = var.varName  # 例如 x_1_3
    #         _, i, j = var_name.split('_')  # 解析出 '1' 和 '3'
    #         i, j = int(i), int(j)  # 转化为整数
    #         gurobi_solution[(i, j)] = 1  # 将结果存储到字典中
    #
    # # 创建绘图
    # kuoda = 80
    # plt.figure(figsize=(8, 8))
    #
    # # 绘制所有节点
    # for i, (x, y) in enumerate(coords):
    #     plt.scatter(x*kuoda, y*kuoda, color='blue')
    #     plt.text(x*kuoda, y*kuoda, f'{i}', fontsize=12, ha='right')  # 显示节点编号
    #
    #
    # for (i, j), val in gurobi_solution.items():
    #     if val == 1:
    #         plt.plot([coords[i][0]*kuoda, coords[j][0]*kuoda], [coords[i][1]*kuoda, coords[j][1]*kuoda],
    #                  color='red', linewidth=2, label='Line between Points')
    #
    # plt.title('ex')
    # plt.show()
    # # print(mdl.PoolObjBound)
    time_e_tmp = time2-time1
    return obj_res, time_e_tmp, mdl.status


for aisle_num in range(2,11):
    for cross_num in range(2,11):
        print(f'现在运行到aisle{aisle_num},cross{cross_num}')
        time_e = []
        time_d = []
        time_a = []
        file = f'{aisle_num}_{cross_num}'
        require_info = np.load(f'files/speed/{file}/required_info.npy', allow_pickle=True)
        steiner_info = np.load(f'files/speed/{file}/steiner_info.npy', allow_pickle=True)
        distance_matrix = np.load(f'files/speed/{file}/distance_matrix.npy', allow_pickle=True)

        for i in range(0, 500):
            min_ = max(4, aisle_num*(cross_num-1)*4-20)
            max_ = aisle_num*(cross_num-1)*4+20
            info_stsp = generate_data(require_info,min_,max_)  # 随机抽取的点的idx(pad)、坐标(pad)、坐标(没有pad)

            required_len = len(info_stsp['all_loc_before_delete'].tolist())
            steiner_len = len(steiner_info[0])

            all_loc = torch.tensor(steiner_info[3].tolist()+info_stsp['all_loc_before_delete'].tolist(), device='cuda').unsqueeze(0)
            aisle_vertex_info, where_is_vertex = get_info_aisle_vertex(all_loc, [required_len], [steiner_len], aisle_num, cross_num)
            coords = all_loc.squeeze(0).tolist()

            ex_length, time_e_tmp, status = exact_solver1(aisle_vertex_info[0],coords, aisle_num, cross_num)
            time_e.append(time_e_tmp)
            exact_l.append(ex_length)
            print(f'jq:{ex_length}')

            time3 = time.time()
            length = tsp_solver(model, info_stsp, f'speed/{file}', distance_matrix)
            time4 = time.time()
            deep_l1.append(length)
            time_d.append(time4-time3)

            aco_length, time_aco_tmp = ACO_solver(info_stsp['orders_idx'], distance_matrix)
            time_a.append(time_aco_tmp)
            aco_l.append(aco_length)

        avg_time_deep = np.mean(time_d)
        avg_time_exact = np.mean(time_e)
        avg_time_aco = np.mean(time_a)
        print(f'现在运行到aisle{aisle_num},cross{cross_num},avg_time_deep={avg_time_deep},avg_time_exact={avg_time_exact},avg_time_aco={avg_time_aco}')

        df = df._append(pd.DataFrame([[aisle_num, cross_num, avg_time_deep, avg_time_exact,avg_time_aco,status]], columns=df.columns))

# 保存到Excel文件
df.to_excel('速度对比.xlsx', index=False)

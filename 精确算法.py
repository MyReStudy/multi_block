import itertools
import math

import numpy as np
import pandas as pd
from gurobipy import LinExpr, quicksum, GRB, Model, tuplelist
from matplotlib import pyplot as plt

def exact_solver(order_after_delete):
    # 模型构建
    m = Model('tsp')
    order_after_delete = order_after_delete.tolist()
    node_num = len(order_after_delete)
    # 添加变量
    x = m.addVars(node_num, node_num, vtype=GRB.BINARY, name='x')
    u = m.addVars(node_num, vtype=GRB.CONTINUOUS, name='u')
    distmat = np.load('files/56/distance_matrix.npy', allow_pickle=True)
    # 设置目标函数
    m.modelSense = GRB.MINIMIZE
    m.setObjective(sum(distmat[order_after_delete[i]][order_after_delete[j]] * x[i, j] for i in range(node_num) for j in range(node_num)), GRB.MINIMIZE)
    # 添加限制
    m.addConstrs(sum(x[i, j] for i in range(node_num) if i != j) == 1 for j in range(node_num))
    m.addConstrs(sum(x[i, j] for j in range(node_num) if i != j) == 1 for i in range(node_num))
    m.addConstrs((u[i] - u[j] + node_num * x[i, j]) <= node_num - 1 for i in range(1, node_num) for j in range(1, node_num) if i != j)

    m.write('tsp.lp')
    m.optimize()

    print(m.ObjVal)

    # 找到那些是1的变量
    result = 0
    route = []
    for i in range(node_num):
        for j in range(node_num):
            if x[i, j].X == 1:
                result+=distmat[order_after_delete[i]][order_after_delete[j]]
                route.append([order_after_delete[i], order_after_delete[j]])
    print(route)

    # # 下面开始画图
    # gurobi_solution = {}
    # # 遍历所有变量，找出名字以 'x' 开头并且值为1的变量
    # for var in m.getVars():
    #     if var.varName.startswith('x') and var.x > 0.5:  # 判断是否是路径，并且值为 1
    #         # 解析出节点 i 和 j
    #         var_name = var.varName  # 例如 x_1_3
    #         _, i, j = var_name.split('_')  # 解析出 '1' 和 '3'
    #         i, j = int(i), int(j)  # 转化为整数
    #         gurobi_solution[(i, j)] = 1  # 将结果存储到字典中
    #
    # # # 创建绘图
    # plt.figure(figsize=(8, 8))
    #
    # # 绘制所有节点
    # for i, (x, y) in enumerate(order_after_delete):
    #     plt.scatter(x, y, color='blue')
    #     plt.text(x, y, f'{i}', fontsize=12, ha='right')  # 显示节点编号
    #
    #
    # for (i, j), val in gurobi_solution.items():
    #     if val == 1:
    #         plt.plot([order_after_delete[i][0], order_after_delete[j][0]], [order_after_delete[i][1], order_after_delete[j][1]],
    #                  color='red', linewidth=2, label='Line between Points')
    #
    # plt.title('ex')
    # plt.show()

    return result
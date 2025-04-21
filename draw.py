from matplotlib import pyplot as plt

def draw_exact(mdl, coords):
    # 下面开始画图
    gurobi_solution = {}
    # 遍历所有变量，找出名字以 'x' 开头并且值为1的变量
    for var in mdl.getVars():
        if var.varName.startswith('x') and var.x > 0.5:  # 判断是否是路径，并且值为 1
            # 解析出节点 i 和 j
            var_name = var.varName  # 例如 x_1_3
            _, i, j = var_name.split('_')  # 解析出 '1' 和 '3'
            i, j = int(i), int(j)  # 转化为整数
            gurobi_solution[(i, j)] = 1  # 将结果存储到字典中

    # 创建绘图
    kuoda = 100

    # 绘制所有节点
    for i, (x, y) in enumerate(coords):
        plt.scatter(x * kuoda, y * kuoda, color='blue')
        plt.text(x * kuoda, y * kuoda, f'{i}', fontsize=12, ha='right')  # 显示节点编号

    for (i, j), val in gurobi_solution.items():
        if val == 1:
            # 这里可以稍微偏移一下连线，例如在y方向偏移0.1
            offset_y = 0.0001
            plt.plot([coords[i][0] * kuoda, coords[j][0] * kuoda],
                     [coords[i][1] * kuoda + offset_y, coords[j][1] * kuoda + offset_y],
                     color='red', linewidth=2)


def draw_deep(x_coords, y_coords):
    coords_list = [(a, b) for a, b in zip(x_coords, y_coords)]
    # 这里可以稍微偏移一下连线，例如在y方向偏移-0.1
    offset_y = -0.1
    plt.scatter(x_coords, y_coords)
    plt.plot(x_coords, y_coords, '-o', color='green')
    # 标注每个点的坐标
    for i, coord in enumerate(coords_list):
        plt.annotate(f"({coord[0]}, {coord[1]})", xy=coord, xytext=(5, 5), textcoords='offset points')

def draw(mdl, coords, x_coords, y_coords):
    plt.figure(figsize=(8, 8))
    draw_exact(mdl, coords)
    draw_deep(x_coords, y_coords)

    plt.title('Comparison of Two Algorithms')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

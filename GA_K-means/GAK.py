import numpy as np
import matplotlib.pyplot as plt

people_num = 1000
K = 8  # 划分区域数
# 范围0~1000 均匀分布
X_aix = np.random.randint(0, 10001, people_num)
Y_aix = np.random.randint(0, 10001, people_num)
# X_aix = np.random.randn(people_num)
# Y_aix = np.random.randn(people_num)
bias_x = np.random.random(people_num)
bias_y = np.random.random(people_num)
X_aix = X_aix[:] + bias_x[:]
Y_aix = Y_aix[:] + bias_y[:]


# 二进制转成下标
def biToNum(arr, i):
    s = 1
    res = 0.
    for j in range(chromosome_size - 1, -1, -1):  # 2进制转10进制
        if arr[i][j] == 1:
            res += s
        s *= 2
    res = lower_bound + res * (upper_bound - lower_bound) / (np.power(2, chromosome_size) - 1)  # 解码
    return res


# 初始化种群
def init():
    for i in range(population_size):
        for j in range(chromosome_size):
            population_x[i][j] = np.random.randint(2)
            population_y[i][j] = np.random.randint(2)


# 两点之间的距离公式
def distance(x1, y1, x2, y2):
    d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return np.power(d, 0.5)


# 适应度
def fitness():
    # fit = np.zeros((population_size,), dtype=np.float32)
    # 适应度函数为：f(x) = -x-10*sin(5*x)-7*cos(4*x);
    for i in range(population_size):
        s_x = 1
        r_x = 0.
        for j in range(chromosome_size - 1, -1, -1):  # 2进制转10进制
            if population_x[i][j] == 1:
                r_x += s_x
            s_x *= 2
        s_y = 1
        r_y = 0.
        for j in range(chromosome_size - 1, -1, -1):  # 2进制转10进制
            if population_y[i][j] == 1:
                r_y += s_y
                s_y *= 2
        r_x = lower_bound + r_x * (upper_bound - lower_bound) / (np.power(2, chromosome_size) - 1)  # 解码
        r_y = lower_bound + r_y * (upper_bound - lower_bound) / (np.power(2, chromosome_size) - 1)
        d = 0.
        for k in range(people_num):
            d += distance(r_x, r_y, X_aix[k], Y_aix[k])
        fitness_value[i] = d  # 计算自变量xi的适应度函数值,点的距离之和


# 对个体按适应度大小进行排序，并且保存最佳个体
def rank(G, best_fitness, best_generation):
    temp_chromosome_x = np.zeros((chromosome_size,))
    temp_chromosome_y = np.zeros((chromosome_size,))
    # 遍历种群
    # 选择排序
    # 最后population(i)的适应度随i递增而递增，population(0)最小，population(population_size-1)最大
    for i in range(population_size):
        max_index = i
        for j in range(i + 1, population_size, 1):
            max_index = j if fitness_value[j] > fitness_value[max_index] else max_index
        if max_index != i:
            # 交换 fitness_value(i) 和 fitness_value(max_index) 的值
            temp = fitness_value[i]
            fitness_value[i] = fitness_value[max_index]
            fitness_value[max_index] = temp
            # 交换 population(i) 和 population(max_index) 的染色体串
            for k in range(chromosome_size):
                temp_chromosome_x[k] = population_x[i][k]
                population_x[i][k] = population_x[max_index][k]
                population_x[max_index][k] = temp_chromosome_x[k]
                temp_chromosome_y[k] = population_y[i][k]
                population_y[i][k] = population_y[max_index][k]
                population_y[max_index][k] = temp_chromosome_y[k]
    # fitness_sum(i) = 前i个个体的适应度之和
    fitness_sum[0] = fitness_value[0]
    for i in range(1, population_size, 1):
        fitness_sum[i] = fitness_sum[i - 1] + fitness_value[i]
    # 更新最大适应度和对应的迭代次数，保存最佳个体(最佳个体的适应度最大)
    if fitness_value[population_size - 1] > best_fitness:
        best_fitness = fitness_value[population_size - 1]
        best_generation = G + 1
        fitness_average[G] = fitness_sum[population_size - 1] / population_size  # 第G次迭代的平均适应度
        for j in range(chromosome_size):
            best_individual_x[j] = population_x[population_size - 1][j]
            best_individual_y[j] = population_y[population_size - 1][j]
    return best_fitness, best_generation


# 选择
def selection():
    population_new_x = np.zeros((population_size, chromosome_size))
    population_new_y = np.zeros((population_size, chromosome_size))
    for i in range(population_size):
        r = np.random.random() * fitness_sum[population_size - 1]  # 产生一个随机适应度[0...max]
        first = 0
        last = population_size - 1
        mid = int(round((last + first) / 2))  # 向下取整
        index = -1
        # 排中法选择个体
        while (first <= last) and (index == -1):
            if r > fitness_sum[mid]:
                first = mid
            elif r < fitness_sum[mid]:
                last = mid
            else:
                index = mid
                break
            if last - first == 1:
                index = first
                break
            mid = int(round((first + last) / 2))
        # 产生新一代个体
        for j in range(chromosome_size):
            population_new_x[i][j] = population_x[index][j]
            population_new_y[i][j] = population_y[index][j]
    # 是否精英选择
    if elitism:
        population_x[0:population_size - 1, :] = population_new_x[0:population_size - 1, :]
        population_y[0:population_size - 1, :] = population_new_y[0:population_size - 1, :]
    else:
        population_x[:, :] = population_new_x[:, :]
        population_y[:, :] = population_new_y[:, :]


# 交叉
def crossover():
    for i in range(0, population_size, 2):
        if np.random.random() < cross_rate:
            cross_position = round(np.random.random() * chromosome_size)
            if cross_position == 0 or cross_position == 1:
                continue
            for j in range(cross_position, chromosome_size, 1):
                temp_x = population_x[i, j]
                population_x[i, j] = population_x[i + 1, j]
                population_x[i + 1, j] = temp_x
                temp_y = population_y[i, j]
                population_y[i, j] = population_y[i + 1, j]
                population_y[i + 1, j] = temp_y


# 变异
def mutation():
    for i in range(population_size):
        if np.random.random() < mutate_rate:
            mutate_position = round(np.random.random() * (chromosome_size - 1))  # 变异位置
            if mutate_position != 0:
                population_x[i, mutate_position] = 1 - population_x[i, mutate_position]
                population_y[i, mutate_position] = 1 - population_y[i, mutate_position]


def genetic_algorithm():
    n = 0.  # 历代最佳适应值
    p = 0  # 最佳个体出现代
    init()  # 初始化
    for g in range(generation_size):
        fitness()  # 计算适应度
        n, p = rank(g, n, p)  # 对个体按适应度大小进行排序
        selection()  # 选择操作
        crossover()  # 交叉操作
        mutation()  # 变异操作
    # 获得最佳个体变量值，对不同的优化目标，这里需要修改
    s_x = 1
    r_x = 0
    for j in range(chromosome_size - 1, -1, -1):  # 2进制转10进制
        if best_individual_x[j] == 1:
            r_x += s_x
        s_x *= 2
    r_X = lower_bound + r_x * (upper_bound - lower_bound) / (np.power(2, chromosome_size) - 1)  # 解码
    s_y = 1
    r_y = 0
    for j in range(chromosome_size - 1, -1, -1):  # 2进制转10进制
        if best_individual_y[j] == 1:
            r_y += s_y
        s_y *= 2
    r_Y = lower_bound + r_y * (upper_bound - lower_bound) / (np.power(2, chromosome_size) - 1)  # 解码
    return n, p, np.float32(r_X), np.float32(r_Y)


def getInitialPoint():
    point = np.zeros((K, 2), dtype=np.float32)
    for i in range(K):
        best_fitness, iterations, x, y = genetic_algorithm()
        initial_x.append(x)
        initial_y.append(y)
        point[i][0] = x
        point[i][1] = y
    return point


##############################--------------K-means 算法-------------

# 聚类
def cluster(center):
    for i in range(people_num):
        minDis = 1e8
        resClass = 0
        for k in range(K):
            dis = distance(X_aix[i], Y_aix[i], center[k][0], center[k][1])
            if dis < minDis:
                minDis = dis
                resClass = k
        labels[i] = resClass + 1


# 重新计算质心
def calculCenter():
    center = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        center_x = 0.
        center_y = 0.
        num = 0
        for i in range(people_num):
            if labels[i] == k + 1:
                num += 1
                center_x += X_aix[i]
                center_y += Y_aix[i]
        if num == 0: num = 1
        center[k][0] = center_x / num
        center[k][1] = center_y / num
    return center


# GK-means
def GKmeans():
    center = getInitialPoint()
    cluster(center)
    for i in range(clusterNum):
        plotGA(center, i)  # 打印
        center = calculCenter()
        cluster(center)
    return center


# K-means
def Kmeans():
    center = np.zeros((K, 2), dtype=np.float32)
    for i in range(K):
        center[i][0] = np.random.random() * 1000
        center[i][1] = np.random.random() * 1000
    cluster(center)
    for i in range(clusterNum):
        plotGA(center, i)  # 打印
        center = calculCenter()
        cluster(center)
    return center


def plotGA(center, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.cla()
    ax.scatter(center[:, 0], center[:, 1])
    ax.scatter(X_aix, Y_aix, 3.0 * labels, 3.0 * labels)
    plt.axis('off')
    plt.show()
    if i == (clusterNum - 1):
        print("暂停15s~~~")
        plt.pause(15)
    else: plt.pause(0.5)
    plt.close()
    plt.ioff()


# 评估性能
def calculationFit(center):
    Dis = np.zeros((K,))
    Num = np.zeros((K,))
    for k in range(K):
        num = 0
        dis = 0.
        for i in range(people_num):
            if labels[i] == k + 1:
                num += 1
                dis += distance(center[k][0], center[k][1], X_aix[i], Y_aix[i])
        Dis[k] = dis / num
        Num[k] = num
    return Dis, Num


# 算方差
def calcuVariance(arr, mid):
    temp = 0.
    for i in range(arr.size):
        temp += (arr[i] - mid) * (arr[i] - mid)
    return np.power(temp, 0.5)


if __name__ == "__main__":
    # 设置遗传算法的参数，测试效果
    # 设定求解精度为小数点后4位
    clusterNum = 10  # kmeans聚类次数
    elitism = True  # 选择精英操作
    population_size = 40  # 种群大小
    chromosome_size = 16  # 染色体长度
    generation_size = 20  # 最大迭代次数
    cross_rate = 0.6  # 交叉概率
    mutate_rate = 0.01  # 变异概率
    population_x = np.zeros((population_size, chromosome_size))  # 种群
    population_y = np.zeros((population_size, chromosome_size))
    best_individual_x = np.zeros((chromosome_size,))  # 历代最佳个体
    best_individual_y = np.zeros((chromosome_size,))
    fitness_value = np.zeros((population_size,), dtype=np.float32)  # 当前代适应度矩阵
    fitness_sum = np.zeros((population_size,), dtype=np.float32)  # 种群累计适应度矩阵
    fitness_average = np.zeros((generation_size,), dtype=np.float32)  # 历代平均适应值矩阵
    labels = np.zeros((people_num,), dtype=int)
    initial_x = []
    initial_y = []
    upper_bound = 10000  # 自变量的区间上限
    lower_bound = 0  # 自变量的区间下限
    finalCenter1 = GKmeans()
    Dis1, Num1 = calculationFit(finalCenter1)
    print("############GKmeans#########质心：")
    print(finalCenter1)
    finalCenter2 = Kmeans()
    Dis2, Num2 = calculationFit(finalCenter2)
    print("############Kmeans#########质心：")
    print(finalCenter2)
    print("############GKmeans#########各区域用户数：")
    print(Num1)
    print("############Kmeans#########各区域用户数：")
    print(Num2)
    print("############GKmeans#########距离质心平均距离：")
    print(Dis1)
    print("方差：", calcuVariance(Dis1, sum(Dis1)/K))
    print("############Kmeans#########距离质心平均距离：")
    print(Dis2)
    print("方差：", calcuVariance(Dis2, sum(Dis2)/K))

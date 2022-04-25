import numpy as np
import matplotlib.pyplot as plt


# 初始化种群
def init():
    for i in range(population_size):
        for j in range(chromosome_size):
            population[i][j] = np.random.randint(2)


# 适应度
def fitness():
    fit = np.zeros((population_size,), dtype=np.float32)
    # 适应度函数为：f(x) = -x-10*sin(5*x)-7*cos(4*x);
    for i in range(population_size):
        s = 1
        for j in range(chromosome_size - 1, -1, -1):  # 2进制转10进制
            if population[i][j] == 1:
                fit[i] += s
            s *= 2
        fit[i] = lower_bound + fit[i] * (upper_bound - lower_bound) / (np.power(2, chromosome_size) - 1)  # 解码
        fitness_value[i] = fit[i] + 10 * np.sin(5 * fit[i]) + 7 * np.cos(4 * fit[i])  # 计算自变量xi的适应度函数值


# 对个体按适应度大小进行排序，并且保存最佳个体
def rank(G, best_fitness, best_generation):
    temp_chromosome = np.zeros((chromosome_size,))
    # 遍历种群
    # 选择排序
    # 最后population(i)的适应度随i递增而递增，population(0)最小，population(population_size-1)最大
    for i in range(population_size):
        min_index = i
        for j in range(i + 1, population_size, 1):
            min_index = j if fitness_value[j] < fitness_value[min_index] else min_index
        if min_index != i:
            # 交换 fitness_value(i) 和 fitness_value(min_index) 的值
            temp = fitness_value[i]
            fitness_value[i] = fitness_value[min_index]
            fitness_value[min_index] = temp
            # 交换 population(i) 和 population(min_index) 的染色体串
            for k in range(chromosome_size):
                temp_chromosome[k] = population[i][k]
                population[i][k] = population[min_index][k]
                population[min_index][k] = temp_chromosome[k]
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
            best_individual[j] = population[population_size-1][j]
    return best_fitness, best_generation


# 选择
def selection():
    population_new = np.zeros((population_size, chromosome_size))
    for i in range(population_size):
        r = np.random.random() * fitness_sum[population_size-1]  # 产生一个随机适应度[0...max]
        first = 0
        last = population_size - 1
        mid = int(round((last+first)/2))  # 向上取整
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
                index = last
                break
            mid = int(round((first+last)/2))
        # 产生新一代个体
        for j in range(chromosome_size):
            population_new[i][j] = population[index][j]
    # 是否精英选择
    if elitism:
        population[0:population_size-1, :] = population_new[0:population_size-1, :]
    else:
        population[:, :] = population_new[:, :]
    # p = population_size-1 if elitism else population_size
    # for i in range(p):
    #     for j in range(chromosome_size):
    #         population


# 交叉
def crossover():
    for i in range(0, population_size, 2):
        if np.random.random() < cross_rate:
            cross_position = round(np.random.random()*chromosome_size)
            if cross_position == 0 or cross_position == 1:
                continue
            for j in range(cross_position,chromosome_size,1):
                temp = population[i, j]
                population[i, j] = population[i+1, j]
                population[i+1, j] = temp


# 变异
def mutation():
    for i in range(population_size):
        if np.random.random() < mutate_rate:
            mutate_position = round(np.random.random()*(chromosome_size-1))  # 变异位置
            if mutate_position != 0:
                population[i, mutate_position] = 1 - population[i, mutate_position]


def plotGA():
    Y = fitness_average
    plt.plot(Y, color="red", linewidth=1.0, linestyle="-")  # 将100个散点连在一起
    plt.show()


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
    s = 1
    q = 0
    for j in range(chromosome_size - 1, -1, -1):  # 2进制转10进制
        if best_individual[j] == 1:
            q += s
        s *= 2
    q = lower_bound + q * (upper_bound - lower_bound) / (np.power(2, chromosome_size) - 1)  # 解码
    return n, p, q


if __name__ == "__main__":
    # 设置遗传算法的参数，测试效果
    # 设定求解精度为小数点后4位
    elitism = True  # 选择精英操作
    population_size = 100  # 种群大小
    chromosome_size = 15  # 染色体长度
    generation_size = 15  # 最大迭代次数
    cross_rate = 0.6  # 交叉概率
    mutate_rate = 0.01  # 变异概率
    population = np.zeros((population_size, chromosome_size))  # 种群
    best_individual = np.zeros((chromosome_size,))  # 历代最佳个体
    fitness_value = np.zeros((population_size,), dtype=np.float32)  # 当前代适应度矩阵
    fitness_sum = np.zeros((population_size,), dtype=np.float32)  # 种群累计适应度矩阵
    fitness_average = np.zeros((generation_size,), dtype=np.float32)  # 历代平均适应值矩阵
    upper_bound = 9  # 自变量的区间上限
    lower_bound = 0  # 自变量的区间下限
    best_fitness, iterations, x = genetic_algorithm()
    print(x)
    print(x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x))
    print(best_fitness)
    print(iterations)
    plotGA()  # 打印算法迭代过程

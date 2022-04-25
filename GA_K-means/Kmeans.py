import numpy as np
import matplotlib.pyplot as plt
import time

people_num = 200
K = 4  # 划分区域数
# 范围0~1000 均匀分布
X_aix = np.random.randint(0, 1001, people_num)
Y_aix = np.random.randint(0, 1001, people_num)
# X_aix = np.random.randn(people_num)
# Y_aix = np.random.randn(people_num)
bias_x = np.random.random(people_num)
bias_y = np.random.random(people_num)
X_aix = X_aix[:] + bias_x[:]
Y_aix = Y_aix[:] + bias_y[:]


# 两点之间的距离公式
def distance(x1, y1, x2, y2):
    d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return np.power(d, 0.5)


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
        labels2[i] = resClass + 1


# 重新计算质心
def calculCenter():
    center = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        center_x = 0.
        center_y = 0.
        num = 0
        for i in range(people_num):
            if labels2[i] == k + 1:
                num += 1
                center_x += X_aix[i]
                center_y += Y_aix[i]
        center[k][0] = center_x / num
        center[k][1] = center_y / num
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
    ax.scatter(X_aix, Y_aix, 3.0 * labels2, 3.0 * labels2)
    plt.axis('off')
    plt.show()
    plt.pause(4)
    if i != clusterNum - 2:
        plt.close()
        plt.ioff()


if __name__ == "__main__":
    # 设置遗传算法的参数，测试效果
    # 设定求解精度为小数点后4位
    clusterNum = 5  # kmeans聚类次数
    labels2 = np.zeros((people_num,), dtype=int)
    finalCenter2 = Kmeans()
    print(finalCenter2)

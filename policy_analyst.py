from database_player import HMDplayer
import matplotlib.pyplot as plt  # plt 用于显示图片
from collections import deque
from pylab import mpl
from predictpolicy import *
import saliency_policy
import numpy as np
import video
import csv
import os
# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 初始化路径
path = video.Video().test_set_path
out_name = video.Video().video_name + '.png'
out_path = os.path.join('out', out_name)

class Policy:
    def __init__(self, predict_policy):
        self.policy = predict_policy
        self.name = self.policy().name
        print(self.name)
        self.accuracy = deque()
        self.tile_accuracy = deque()
        self.player_no = deque()
        self.execute()


    def execute(self):
        files = os.listdir(path)
        for file in files:
            print('当前文件为：', file)
            csv_path = os.path.join(path, file)
            play = HMDplayer(csv_path, self.policy)
            self.accuracy.append(play.accuracy)
            self.tile_accuracy.append(play.Mean_accuracy)

class Saliency_Policy():
    def __init__(self):
        self.accuracy = deque()
        self.tile_accuracy = deque()
        self.execute()

    def execute(self):
        files = os.listdir(path)
        for file in files:
            print('当前数据为：', file)
            csv_path = os.path.join(path, file)
            play = saliency_policy.Player(csv_path)
            self.accuracy.append(play.accuracy)
            self.tile_accuracy.append(play.Mean_accuracy)


'''
    def write(self):
        print("正在写入...")
        csv_name = "out_" + str(Configuration().predict_time) + ".csv"
        out = open(csv_name, 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(['', '精确度', 'tile精确度'])
        csv_write.writerow([self.policy().name, self.accuracy, self.tile_accuracy])
        out.close()
        print("写入成功！")
'''

class Analyst:
    def __init__(self):
        self.result = deque()
        self.colors = ['red', 'orange', 'green', 'purple', 'yellow']  # 设置颜色
        self.label = []
        self.run()

    def run(self):
        # 策略设置
        self.policy = [Policy(LSRpolicy), Policy(Ridgepolicy)]
        # 统计
        for i in range(len(self.policy)):
            self.label.append(self.policy[i].name)
            self.result.append(self.policy[i].tile_accuracy)

    def average(self):
        print("正在写入...")
        csv_name = "out_" + video.Video().video_name + '_' + str(video.Configuration().predict_time) + ".csv"
        csv_path = os.path.join('out', csv_name)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        out = open(csv_path, 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(['策略名', '精确度'])
        for i in range(len(self.result)):
            queue = self.result[i]
            result = np.mean(queue)
            print(self.label[i], '的平均精确度为：', result)
            csv_write.writerow([self.label[i], result])
        out.close()
        print("写入成功！")

    def map(self):
        # 一些初始化
        plt.figure()
        plt.xlabel("编号")
        plt.ylabel("正确率")
        # plt.xlim(0.8, 1) # X轴上下界必须再商量
        #plt.ylim(0, 1)
        for i in range(len(self.result)):
            data = self.result[i]
            # L0-L1层统计
            player = []
            length = len(data)
            for j in range(length):
                player.append(j+1)
            # 设置颜色和标签
            plt.plot(player, data, '-', color=self.colors[i], label=self.label[i])

        plt.legend(loc="best")

        plt.grid(linestyle='--')  # 设置网格
        plt.savefig('正确率统计.svg')
        plt.show()

    def cdf(self):
        # 一些初始化
        plt.figure()
        plt.xlabel("正确率")
        plt.ylabel("CDF")
        # plt.xlim(0.8, 1) # X轴上下界必须再商量
        plt.ylim(0, 1)
        for i in range(len(self.result)):
            queue = self.result[i]
            # L0-L1层统计
            result = []
            data = sorted(queue)
            length = len(data)
            for j in range(length):
                temp = (j + 1) / length
                result.append(temp)
            # 设置颜色和标签
            plt.plot(data, result, '-', color=self.colors[i], label=self.label[i])

        plt.legend(loc="best")

        plt.grid(linestyle='--')  # 设置网格
        plt.savefig(out_path)
        plt.show()


if __name__ == '__main__':
    a = Analyst()
    a.average()
    a.map()




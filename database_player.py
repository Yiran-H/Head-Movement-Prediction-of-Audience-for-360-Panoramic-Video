from video import *
from collections import deque  # 双端队列
from transformtile import Transform_tile
from pylab import mpl
import matplotlib.pyplot as plt  # plt 用于显示图片
import predictpolicy as pp
import pandas as pd
import numpy as np
import math
import os

# 设置字体（不然显示不出中文）
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
out_path = "out"
# 户号
no_user = 1

# 初始化路径
file_path = Video().test_set_path
video_name = Video().video_name
file_name = 'user_' + "%03d" % no_user + '_' + video_name + '.csv'
file_in = os.path.join(file_path, file_name)
# 预设参数：



class HMDplayer:
    def __init__(self, path_in, policy):
        # 接口
        self.file_in = path_in
        self.predict_policy = policy()
        # 初始化
        self.video = Video()
        self.config = Configuration()
        self.transform_tile = Transform_tile()
        self.sliding_windows = int(self.video.video_fps * self.config.training_time)
        self.predict_windows = int(self.video.video_fps * self.config.predict_time)
        file = pd.read_csv(self.file_in, usecols=[0, 1, 2])
        df = pd.DataFrame(file)
        self.real_coordinates = deque()
        self.pre_coordinates = deque()
        self.accuracy = deque()
        # 速度
        self.yaw_rate = deque()
        self.pitch_rate = deque()
        # 开始播放
        self.play(df)
        self.Mean_accuracy = np.mean(np.asarray(self.accuracy))
        print('平均精确度为:', self.Mean_accuracy)

    '''
    def play(self, df):
        training_set = deque(maxlen=self.sliding_windows)
        data = np.array(df)
        for i in range(len(data)):
            frame = i + 1
            # print("当前帧为", frame)
            current_coordinates = self.transform_tile.corrected_coordinates([data[i][0], data[i][1], data[i][2]])
            self.real_coordinates.append(current_coordinates)
            print("当前时间帧为", frame, "坐标为", current_coordinates)
            if frame <= self.video.video_frames - self.predict_windows:
                next_frame = float(frame + self.predict_windows)
                training_set.append(current_coordinates)
                predict_coordinates = self.transform_tile.corrected_coordinates(self.predict_policy.predict(training_set, next_frame))
                print("未来时间预测坐标", next_frame, "坐标为", predict_coordinates)
                self.pre_coordinates.append(predict_coordinates)

            if frame > self.predict_windows:
                current_prediction = self.pre_coordinates[i - self.predict_windows]
                print('当前时间预测坐标', current_prediction)
                current_tile_seq = self.transform_tile.tile_counter(current_coordinates[1], current_coordinates[2])
                predict_tile_seq = self.transform_tile.tile_counter(current_prediction[1], current_prediction[2])
                accuracy = self.cal_accuracy(predict_tile_seq, current_tile_seq)
                # print(current_tile_seq, predict_tile_seq, '当前精确度为', accuracy)
                self.accuracy.append(accuracy)

    '''
    def play(self, df):
        training_set = deque(maxlen=self.sliding_windows)
        data = np.array(df)
        for i in range(len(data)):
            frame = i + 1
            #print("当前帧为", frame)
            current_coordinates = [data[i][0], data[i][1], data[i][2]]
            self.real_coordinates.append(current_coordinates)
            print("当前时间帧为", frame, "坐标为", current_coordinates)
            if frame <= self.video.video_frames - self.predict_windows:
                next_frame = float(frame + self.predict_windows)
                training_set.append(current_coordinates)
                predict_coordinates = self.predict_policy.predict(training_set, next_frame)
                print("未来时间预测坐标", next_frame, "坐标为", predict_coordinates)
                self.pre_coordinates.append(predict_coordinates)

            if frame > self.predict_windows:
                current_prediction = self.pre_coordinates[i - self.predict_windows]
                print('当前时间预测坐标', current_prediction)
                current_tile_seq = self.transform_tile.tile_counter(current_coordinates[1], current_coordinates[2])
                predict_tile_seq = self.transform_tile.tile_counter(current_prediction[1], current_prediction[2])
                accuracy = self.cal_accuracy(predict_tile_seq, current_tile_seq)
                #print(current_tile_seq, predict_tile_seq, '当前精确度为', accuracy)
                self.accuracy.append(accuracy)





    def cal_accuracy(self, pred, real):
        true = len(list(set(pred) & set(real)))
        accuracy = true / len(real)
        return accuracy


    def draw(self):
        plt.figure(1)
        yaw = plt.subplot(2, 1, 1)
        pitch = plt.subplot(2, 1, 2)
        colors = ['black', 'red']  # 设置颜色
        label = ['实际点', '预测点']  # 标签题目

        plt.sca(yaw)  # 选中yaw图
        plt.title("yaw的运动轨迹")
        plt.xlim(0, self.video.video_frames)
        plt.ylim(-180, 180)
        real_frame = np.asarray(self.real_coordinates)[:, 0]
        real_yaw = np.asarray(self.real_coordinates)[:, 1]
        plt.scatter(real_frame, real_yaw, s=1, color=colors[0], label=label[0])
        pre_frame = np.asarray(self.pre_coordinates)[:, 0]
        pre_yaw = np.asarray(self.pre_coordinates)[:, 1]
        plt.scatter(pre_frame, pre_yaw, s=1, color=colors[1], label=label[1])  # alpha=：设置透明度（0-1）
        plt.xlabel("帧")
        plt.ylabel("角度")
        plt.legend(loc="best")

        plt.sca(pitch)  # 选中pitch图
        plt.title("pitch的运动轨迹")
        plt.xlim(0, self.video.video_frames)
        plt.ylim(-90, 90)
        real_frame = np.asarray(self.real_coordinates)[:, 0]
        real_pitch = np.asarray(self.real_coordinates)[:, 2]
        plt.scatter(real_frame, real_pitch, s=1, color=colors[0], label=label[0])
        pre_frame = np.asarray(self.pre_coordinates)[:, 0]
        pre_pitch = np.asarray(self.pre_coordinates)[:, 2]
        plt.scatter(pre_frame, pre_pitch, s=1, color=colors[1], label=label[1])
        plt.xlabel("帧")
        plt.ylabel("角度")
        plt.legend(loc="best")
        plt.show()


if __name__ == '__main__':
    # PredictPolicy.LSRpolicy
    a = HMDplayer(file_in, pp.LRpolicy)
    a.draw()

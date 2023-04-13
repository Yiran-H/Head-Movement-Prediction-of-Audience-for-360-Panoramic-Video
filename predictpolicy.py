#Python本身不提供抽象类和接口机制，要想实现抽象类，可以借助abc模块。ABC是Abstract Base Class的缩写。
from abc import ABC, abstractmethod
from prediction import Predict
from collections import deque  # 双端队列
from video import *
from transformtile import Transform_tile
import configparser
import math
import numpy as np
import pandas as pd

class Policy(ABC):
    @abstractmethod
    def predict(self, train, next_x):
        pass

# 最后样本复制策略-Last Sample Replication
class LSRpolicy(Policy):
    def __init__(self):
        self.name = "LSR"
        self.video = Video()
        self.config = Configuration()
        path = os.path.join('parameter', self.video.video_name)
        para_name = self.video.video_name + '_' + self.name + '_' + str(self.config.predict_time) + '.ini'
        para_path = os.path.join(path, para_name)
        config = configparser.ConfigParser()

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).LSR()
        next_pitch = Predict(train_pitch, next_x).LSR()
        result = [next_x, next_yaw, next_pitch]
        return result

# 线性回归-Linear regression
class LRpolicy(Policy):
    def __init__(self):
        self.name = "LR"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).LR()
        next_pitch = Predict(train_pitch, next_x).LR()
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result

# 局部加权线性回归-Linear regression
class WLRpolicy(Policy):
    def __init__(self):
        self.name = "WLR"
        self.video = Video()
        self.config = Configuration()
        path = os.path.join('parameter', self.video.video_name)
        para_name = self.video.video_name + '_' + self.name + '_' + str(self.config.predict_time) + '.ini'
        para_path = os.path.join(path, para_name)
        config = configparser.ConfigParser()

        self.k = 13  # 这是默认值，后面修改下

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).WLR(self.k)
        next_pitch = Predict(train_pitch, next_x).WLR(self.k)
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result


# 支持向量回归
class SVRpolicy(Policy):
    def __init__(self):
        self.name = "SVR"
        self.video = Video()
        self.config = Configuration()
        path = os.path.join('parameter', self.video.video_name)
        para_name = self.video.video_name + '_' + self.name + '_' + str(self.config.predict_time) + '.ini'
        para_path = os.path.join(path, para_name)
        parameter = configparser.ConfigParser()
        parameter.read(para_path)
        self.gamma = parameter.getfloat("parameter", 'gamma')
        self.C = parameter.getfloat('parameter', 'C')

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).SVR(self.gamma, self.C)
        next_pitch = Predict(train_pitch, next_x).SVR(self.gamma, self.C)
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result

class Ridgepolicy(Policy):
    def __init__(self):
        self.name = "Ridge"
        self.video = Video()
        self.config = Configuration()
        path = os.path.join('parameter', self.video.video_name)
        para_name = self.video.video_name + '_' + self.name + '_' + str(self.config.predict_time) + '.ini'
        para_path = os.path.join(path, para_name)
        parameter = configparser.ConfigParser()
        parameter.read(para_path)
        self.alpha = parameter.getfloat("parameter", 'alpha')

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).Ridge(self.alpha)
        next_pitch = Predict(train_pitch, next_x).Ridge(self.alpha)
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result

class Adaboostpolicy(Policy):
    def __init__(self):
        self.name = "Adaboost"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).Adaboost()
        next_pitch = Predict(train_pitch, next_x).Adaboost()
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result


class RandomForestpolicy(Policy):
    def __init__(self):
        self.name = "RandomForest"
        self.video = Video()
        self.config = Configuration()
        path = os.path.join('parameter', self.video.video_name)
        para_name = self.video.video_name + '_' + self.name + '_' + str(self.config.predict_time) + '.ini'
        para_path = os.path.join(path, para_name)
        parameter = configparser.ConfigParser()
        parameter.read(para_path)
        self.n_estimators = parameter.getint("parameter", 'n_estimators')  # 修改
        self.max_depth = parameter.getint("parameter", 'max_depth')  # 修改

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).RandomForest(self.n_estimators, self.max_depth)
        next_pitch = Predict(train_pitch, next_x).RandomForest(self.n_estimators, self.max_depth)
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result

class Votingpolicy(Policy):
    def __init__(self):
        self.name = "Voting"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).Voting()
        next_pitch = Predict(train_pitch, next_x).Voting()
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result

# 截断的线性预测-truncated linear prediction
class TLPpolicy(Policy):
    def __init__(self):
        self.name = "TLP"
    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).TLP()
        next_pitch = Predict(train_pitch, next_x).TLP()
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        result = [next_x, next_yaw, next_pitch]
        return result

class Cube360policy(Policy):
    def __init__(self):
        self.name = "Cube360"
        self.video = Video()
        self.config = Configuration()
        path = os.path.join('parameter', self.video.video_name)
        para_name = self.video.video_name + '_' + self.name + '_' + str(self.config.predict_time) + '.ini'
        para_path = os.path.join(path, para_name)
        parameter = configparser.ConfigParser()
        parameter.read(para_path)
        self.k = parameter.getint("parameter", 'k')  # 修改

        frames = self.video.video_frames
        self.saliency_set = deque(maxlen=frames)
        current_frame_history = deque()
        for i in range(frames):
            elem = [i + 1, current_frame_history]
            self.saliency_set.append(elem)
        # print(self.saliency_set)
        path = self.video.train_set_path
        files = os.listdir(path)
        for file in files:
            # print('正在读取：', file)
            csv_path = os.path.join(path, file)
            file = pd.read_csv(csv_path, usecols=[0, 1, 2])
            data = np.array(pd.DataFrame(file))
            for i in range(len(data)):
                coordinate = (data[i][1], data[i][2])
                # print(coordinate)
                temp = deque(self.saliency_set[i][1])
                temp.append(coordinate)
                self.saliency_set[i][1] = temp


    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).LR()
        next_pitch = Predict(train_pitch, next_x).LR()
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        history = deque(self.saliency_set[int(next_x - 1)][1])  # 找到当前帧的显著性坐标
        #print(history)
        history_distance = deque()
        for i in range(len(history)):
            temp_yaw = history[i][0]
            temp_pitch = history[i][1]
            distance = math.sqrt(pow(next_yaw - temp_yaw, 2) + pow(next_pitch - temp_pitch, 2))
            history_distance.append([distance, i])
        history_distance = np.array(history_distance)
        history_distance = history_distance[np.argsort(history_distance[:, 0])]
        k_elem = []
        for i in range(self.k):
            index = int(history_distance[i][1])
            k_elem.append(history[index])
        k_elem = np.array(k_elem)
        center = k_elem.mean(axis=0)
        next_yaw = center[0]
        next_pitch = center[1]
        result = [next_x, next_yaw, next_pitch]
        return result

class cube360_LSRpolicy(Policy):
    def __init__(self):
        self.name = "cube360_LSR"
        self.k = 10
        frames = Video().video_frames
        self.saliency_set = deque(maxlen=frames)
        current_frame_history = deque()
        for i in range(frames):
            elem = [i + 1, current_frame_history]
            self.saliency_set.append(elem)
        # print(self.saliency_set)
        path = Video().train_set_path
        files = os.listdir(path)
        for file in files:
            # print('正在读取：', file)
            csv_path = os.path.join(path, file)
            file = pd.read_csv(csv_path, usecols=[0, 1, 2])
            data = np.array(pd.DataFrame(file))
            for i in range(len(data)):
                coordinate = (data[i][1], data[i][2])
                # print(coordinate)
                temp = deque(self.saliency_set[i][1])
                temp.append(coordinate)
                self.saliency_set[i][1] = temp


    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        current_yaw = sample[-1][1]
        current_pitch = sample[-1][2]

        history = deque(self.saliency_set[int(next_x - 1)][1])  # 找到当前帧的显著性坐标
        #print(history)
        history_distance = deque()
        for i in range(len(history)):
            temp_yaw = history[i][0]
            temp_pitch = history[i][1]
            distance = math.sqrt(pow(current_yaw - temp_yaw, 2) + pow(current_pitch - temp_pitch, 2))
            history_distance.append([distance, i])
        history_distance = np.array(history_distance)
        history_distance = history_distance[np.argsort(history_distance[:, 0])]
        k_elem = []
        for i in range(self.k):
            index = int(history_distance[i][1])
            k_elem.append(history[index])
        k_elem = np.array(k_elem)
        center = k_elem.mean(axis=0)
        next_yaw = center[0]
        next_pitch = center[1]
        result = [next_x, next_yaw, next_pitch]
        return result


class MCpolicy(Policy):
    def __init__(self):
        self.name = "MC"
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.predict_time = self.configure.predict_time
        np_name = self.video_name + '_' + str(self.predict_time) + 's.npy'
        np_path = os.path.join('markovchain', np_name)
        if os.path.exists(np_path):
            self.markovchain = np.load(np_path)
        else:
            print('马尔科夫链不存在！请生成！')
            os._exit(0)

    def predict(self, train, next_x):
        sample = deque(train)
        current_coordinates = sample.pop()
        current_frame = int(current_coordinates[0])
        current_yaw = current_coordinates[1]
        current_pitch = current_coordinates[2]
        current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
        current_markovchain = self.markovchain[current_frame - 1][current_tile_no - 1]
        # 为了防止有重复最大值，这里有待改进
        max_value = np.max(current_markovchain)
        if max_value != 0:
            max_tile_no = np.argmax(current_markovchain) + 1  # 索引和tile_id相差1
        else:
            max_tile_no = current_tile_no
        next_yaw, next_pitch = self.tf.cal_tile_center(max_tile_no)
        #print(current_tile_no, max_tile_no)
        result = [next_x, next_yaw, next_pitch]
        return result

class MCswitchpolicy(Policy):
    def __init__(self):
        self.name = "MCswitch"
        self.video = Video()
        self.config = Configuration()
        path = os.path.join('parameter', self.video.video_name)
        para_name = self.video.video_name + '_' + self.name + '_' + str(self.config.predict_time) + '.ini'
        para_path = os.path.join(path, para_name)
        #parameter = configparser.ConfigParser()
        #parameter.read(para_path)
        #self.Ridge_alpha = parameter.getint("parameter", 'Ridge_alpha')  # 修改


        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.predict_time = self.config.predict_time
        # 修改岭回归参数
        #Ridge = Ridgepolicy()
        #Ridge.alpha = self.Ridge_alpha

        self.policys = [LSRpolicy(), TLPpolicy(), LRpolicy(), Ridgepolicy()]
        np_name = self.video_name + '_' + str(self.predict_time) + 's.npy'
        np_path = os.path.join('markovchain', np_name)
        if os.path.exists(np_path):
            self.markovchain = np.load(np_path)
        else:
            print('马尔科夫链不存在！请生成！')
            os._exit(0)

    def correct_coordinates(self, yaw_in, pitch_in):
        corrected_pitch = None
        corrected_yaw = (yaw_in + 180) % 360 - 180
        if pitch_in > 90:
            corrected_pitch = 90
        elif pitch_in < -90:
            corrected_pitch = -90
        return corrected_yaw, corrected_pitch

    def predict(self, train, next_x):
        # 寻找转移矩阵
        current_coordinates = train[-1]
        current_frame = int(current_coordinates[0])
        current_yaw = current_coordinates[1]
        current_pitch = current_coordinates[2]
        current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
        current_markovchain = self.markovchain[current_frame - 1][current_tile_no - 1]
        # 线性回归预测的位置
        prediction = []
        probability = []
        for elem in self.policys:
            elem_predict = elem.predict(train, next_x)
            elem_yaw = elem_predict[1]
            elem_pitch = elem_predict[2]
            elem_tile_no = self.tf.transform_coordinates(elem_yaw, elem_pitch)
            elem_probability = current_markovchain[elem_tile_no - 1]
            prediction.append([elem_predict[1], elem_predict[2]])
            probability.append(elem_probability)
        max_index = int(np.argmax(np.array(probability)))
        next_yaw = prediction[max_index][0]
        next_pitch = prediction[max_index][1]
        result = [next_x, next_yaw, next_pitch]
        #print(result)
        return result

class MCswitch_votepolicy(Policy):
    def __init__(self):
        self.name = "MCswitch_vote"
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.predict_time = self.configure.predict_time
        np_name = self.video_name + '_' + str(self.predict_time) + 's.npy'
        np_path = os.path.join('markovchain', np_name)
        if os.path.exists(np_path):
            self.markovchain = np.load(np_path)
        else:
            print('马尔科夫链不存在！请生成！')
            os._exit(0)

    def correct_coordinates(self, yaw_in, pitch_in):
        corrected_pitch = None
        corrected_yaw = (yaw_in + 180) % 360 - 180
        if pitch_in > 90:
            corrected_pitch = 90
        elif pitch_in < -90:
            corrected_pitch = -90
        return corrected_yaw, corrected_pitch

    def predict(self, train, next_x):
        current_coordinates = train[-1]
        current_frame = int(current_coordinates[0])
        current_yaw = current_coordinates[1]
        current_pitch = current_coordinates[2]
        current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
        current_markovchain = self.markovchain[current_frame - 1][current_tile_no - 1]
        # 线性回归预测的位置
        policys = [LSRpolicy(), Ridgepolicy(), LRpolicy(), TLPpolicy()]
        prediction = []
        probability = []
        key = []
        for elem in policys:
            elem_predict = elem.predict(train, next_x)
            elem_yaw = elem_predict[1]
            elem_pitch = elem_predict[2]
            elem_tile_no = self.tf.transform_coordinates(elem_yaw, elem_pitch)
            elem_probability = current_markovchain[elem_tile_no - 1]
            prediction.append([elem_predict[1], elem_predict[2]])
            probability.append(elem_probability)
            if elem_tile_no == current_tile_no:
                key.append(True)
            else:
                key.append(False)

        print(probability, key)

        if key.count(False) > key.count(True):
            count = 0
            for i in range(len(key)):
                if key[i - count] is True:
                    del key[i - count]
                    del probability[i - count]
                    del prediction[i - count]
                    count += 1
        else:
            count = 0
            for i in range(len(key)):
                if key[i - count] is False:
                    del key[i - count]
                    del probability[i - count]
                    del prediction[i - count]
                    count += 1
        print(key)

        max_index = int(np.argmax(np.array(probability)))
        next_yaw = prediction[max_index][0]
        next_pitch = prediction[max_index][1]
        result = [next_x, next_yaw, next_pitch]
        return result

class MCswitch_TLPpolicy(Policy):
    def __init__(self):
        self.name = "MCswitch_TLP"
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.predict_time = self.configure.predict_time
        np_name = self.video_name + '_' + str(self.predict_time) + 's.npy'
        np_path = os.path.join('markovchain', np_name)
        if os.path.exists(np_path):
            self.markovchain = np.load(np_path)
        else:
            print('马尔科夫链不存在！请生成！')
            os._exit(0)

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        # 寻找转移矩阵
        current_coordinates = sample[-1]
        current_frame = int(current_coordinates[0])
        current_yaw = current_coordinates[1]
        current_pitch = current_coordinates[2]
        current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
        current_markovchain = self.markovchain[current_frame - 1][current_tile_no - 1]
        # 线性回归预测的位置S
        TLP_yaw = Predict(train_yaw, next_x).TLP()
        TLP_pitch = Predict(train_pitch, next_x).TLP()
        TLP_yaw = (TLP_yaw + 180) % 360 - 180
        if TLP_pitch > 90:
            TLP_pitch = 90
        elif TLP_pitch < -90:
            TLP_pitch = -90
        TLP_tile_no = self.tf.transform_coordinates(TLP_yaw, TLP_pitch)

        P_TLP = current_markovchain[TLP_tile_no - 1]
        P_LSR = current_markovchain[current_tile_no - 1]
        if P_LSR >= P_TLP:
            next_yaw = current_yaw
            next_pitch = current_pitch
        else:
            next_yaw = TLP_yaw
            next_pitch = TLP_pitch
        # print(current_tile_no, max_tile_no)
        result = [next_x, next_yaw, next_pitch]
        return result

class MCpropolicy(Policy):
    def __init__(self):
        self.name = "MCpro"
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.predict_time = self.configure.predict_time
        np_name = self.video_name + '_' + str(self.predict_time) + 's.npy'
        np_path = os.path.join('markovchain', np_name)
        if os.path.exists(np_path):
            self.markovchain = np.load(np_path)
        else:
            print('马尔科夫链不存在！请生成！')
            os._exit(0)


    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        # 寻找转移矩阵
        current_coordinates = sample[-1]
        current_frame = int(current_coordinates[0])
        current_yaw = current_coordinates[1]
        current_pitch = current_coordinates[2]
        current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
        current_markovchain = self.markovchain[current_frame - 1][current_tile_no - 1]
        # 线性回归预测的位置S
        next_yaw = Predict(train_yaw, next_x).TLP()
        next_pitch = Predict(train_pitch, next_x).TLP()
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        next_tile_no = self.tf.transform_coordinates(next_yaw, next_pitch)
        center_yaw, center_pitch = self.tf.cal_tile_center(next_tile_no)
        neighbors = self.tf.tile_counter(center_yaw, current_pitch)
        probability = []
        for elem in neighbors:
            probability.append(current_markovchain[elem - 1])
        if sum(probability) == 0:
            max_tile_no = next_tile_no
        else:
            max_id = np.argmax(probability)
            max_tile_no = neighbors[max_id]
        next_yaw, next_pitch = self.tf.cal_tile_center(max_tile_no)
        #print(current_tile_no, max_tile_no)
        result = [next_x, next_yaw, next_pitch]
        return result

class MCpro2policy(Policy):
    def __init__(self):
        self.name = "MCpro2"
        self.video = Video()
        self.configure = Configuration()
        self.tf = Transform_tile()
        self.video_name = self.video.video_name
        self.predict_time = self.configure.predict_time
        np_name = self.video_name + '_' + str(self.predict_time) + 's.npy'
        np_path = os.path.join('markovchain', np_name)
        if os.path.exists(np_path):
            self.markovchain = np.load(np_path)
        else:
            print('马尔科夫链不存在！请生成！')
            os._exit(0)


    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        # 寻找转移矩阵
        current_coordinates = sample[-1]
        current_frame = int(current_coordinates[0])
        current_yaw = current_coordinates[1]
        current_pitch = current_coordinates[2]
        current_tile_no = self.tf.transform_coordinates(current_yaw, current_pitch)
        current_markovchain = self.markovchain[current_frame - 1][current_tile_no - 1]
        # 线性回归预测的位置
        next_yaw = Predict(train_yaw, next_x).WLR()
        next_pitch = Predict(train_pitch, next_x).WLR()
        next_yaw = (next_yaw + 180) % 360 - 180
        if next_pitch > 90:
            next_pitch = 90
        elif next_pitch < -90:
            next_pitch = -90
        next_tile_no = self.tf.transform_coordinates(next_yaw, next_pitch)
        center_yaw, center_pitch = self.tf.cal_tile_center(next_tile_no)
        neighbors = self.tf.tile_counter(center_yaw, current_pitch)
        probability = []
        for elem in neighbors:
            probability.append(current_markovchain[elem - 1])
        if sum(probability) == 0:
            max_tile_no = current_tile_no
        else:
            max_id = np.argmax(probability)
            max_tile_no = neighbors[max_id]
        next_yaw, next_pitch = self.tf.cal_tile_center(max_tile_no)
        #print(current_tile_no, max_tile_no)
        result = [next_x, next_yaw, next_pitch]
        return result

class RATEpolicy(Policy):
    def __init__(self):
        self.name = "RATE"
        self.max_yaw_rate = 0.0
        self.min_yaw_rate = 100.0
        self.max_pitch_rate = 0.0
        self.min_pitch_rate = 100.0

    def yaw_rate(self, rate):
        if rate > self.max_yaw_rate:
            self.max_yaw_rate = rate
        if rate < self.min_yaw_rate:
            self.min_yaw_rate = rate
    def pitch_rate(self, rate):
        if rate > self.max_pitch_rate:
            self.max_pitch_rate = rate
        if rate < self.min_pitch_rate:
            self.min_pitch_rate = rate


    def predict(self, train, next_x):
        #threshold_yaw_rate = (self.max_yaw_rate + self.min_yaw_rate) / 2
        #threshold_pitch_rate = (self.max_pitch_rate + self.min_pitch_rate) / 2
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        if len(sample) > 1:
            # yaw
            yaw_rate = abs((train_yaw[-1][1] - train_yaw[-2][1]) * Video().video_fps)
            self.yaw_rate(yaw_rate)
            threshold_yaw_rate = (self.max_yaw_rate + self.min_yaw_rate) / 2
            print('最大yaw速度为：', self.max_yaw_rate, '最小yaw速度为：', self.min_yaw_rate, '阈值为：', threshold_yaw_rate)
            if yaw_rate > threshold_yaw_rate:
                next_yaw = Predict(train_yaw, next_x).TLP()
            else:
                next_yaw = Predict(train_yaw, next_x).LSR()
            # pitch
            pitch_rate = abs((train_pitch[-1][1] - train_pitch[-2][1]) * Video().video_fps)
            self.pitch_rate(pitch_rate)
            threshold_pitch_rate = (self.max_pitch_rate + self.min_pitch_rate) / 2
            print('最大yaw速度为：', self.max_pitch_rate, '最小yaw速度为：', self.min_pitch_rate, '阈值为：', threshold_pitch_rate)
            if pitch_rate > threshold_pitch_rate:
                next_pitch = Predict(train_pitch, next_x).TLP()
            else:
                next_pitch = Predict(train_pitch, next_x).LSR()
        else:
            next_yaw = Predict(train_yaw, next_x).LSR()
            next_pitch = Predict(train_pitch, next_x).LSR()

        result = [next_x, next_yaw, next_pitch]
        return result

class RATE2policy(Policy):
    def __init__(self):
        self.name = "RATE2"

    def predict(self, train, next_x):
        threshold = 20
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        if len(sample) > 1:
            # yaw
            yaw_rate = abs((train_yaw[-1][1] - train_yaw[-2][1]) * Video().video_fps)
            if yaw_rate > threshold:
                next_yaw = Predict(train_yaw, next_x).TLP()
            else:
                next_yaw = Predict(train_yaw, next_x).LSR()
            # pitch
            pitch_rate = abs((train_pitch[-1][1] - train_pitch[-2][1]) * Video().video_fps)
            if pitch_rate > threshold:
                next_pitch = Predict(train_pitch, next_x).TLP()
            else:
                next_pitch = Predict(train_pitch, next_x).LSR()
        else:
            next_yaw = Predict(train_yaw, next_x).LSR()
            next_pitch = Predict(train_pitch, next_x).LSR()

        result = [next_x, next_yaw, next_pitch]
        return result


# 优化的算法-Optimal prediction algorithm
class OPpolicy(Policy):
    def __init__(self):
        self.name = "OP"
        self.sliding_windows = int(Video().video_fps * Configuration().training_time)
        # 存放预测和实际的举例的队列
        self.yaw_dist = deque(maxlen=self.sliding_windows)
        self.pitch_dist = deque(maxlen=self.sliding_windows)

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        Pre_yaw = Predict(train_yaw, next_x)
        Pre_pitch = Predict(train_pitch, next_x)
        #加策略
        Pre_yaw_list = np.array([Pre_yaw.LSR(), Pre_yaw.LR, Pre_yaw.AVG()])
        Pre_pitch_list = np.array([Pre_pitch.LSR(), Pre_pitch.LR, Pre_pitch.AVG()])


        next_yaw = Predict(train_yaw, next_x).LR()
        next_pitch = Predict(train_pitch, next_x).LR()
        result = [next_x, next_yaw, next_pitch]
        return result

class AVGpolicy(Policy):
    def __init__(self):
        self.name = "AVG"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).AVG()
        next_pitch = Predict(train_pitch, next_x).AVG()
        result = [next_x, next_yaw, next_pitch]
        return result

class WAVGpolicy(Policy):
    def __init__(self):
        self.name = "WAVG"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).WAVG()
        next_pitch = Predict(train_pitch, next_x).WAVG()
        result = [next_x, next_yaw, next_pitch]
        return result

# 指数加权平均数
class IWAVGpolicy(Policy):
    def __init__(self):
        self.name = "IWAVG"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = Predict(train_yaw, next_x).IWAVG()
        next_pitch = Predict(train_pitch, next_x).IWAVG()
        result = [next_x, next_yaw, next_pitch]
        return result

class Speed(Policy):
    def __init__(self):
        self.name = "Speed"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        threshold = 20
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]

        if len(sample) <= 1:
            next_yaw = Predict(train_yaw, next_x).LSR()
            next_pitch = Predict(train_pitch, next_x).LSR()
        else:
            yaw_speed = (train_yaw[-1][1] - train_yaw[-2][1]) * 29
            if abs(yaw_speed) >= threshold:
                next_yaw = Predict(train_yaw, next_x).LR()
            else:
                next_yaw = Predict(train_yaw, next_x).IWAVG()

            pitch_speed = (train_pitch[-1][1] - train_pitch[-2][1]) * 29
            if abs(pitch_speed) >= threshold:
                next_pitch = Predict(train_pitch, next_x).LR()
            else:
                next_pitch = Predict(train_pitch, next_x).IWAVG()

        result = [next_x, next_yaw, next_pitch]

        return result


class AVGTLPpolicy(Policy):
    def __init__(self):
        self.name = "AVGTLP"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = (Predict(train_yaw, next_x).AVG() + Predict(train_yaw, next_x).TLP())/2
        next_pitch = (Predict(train_pitch, next_x).AVG() + + Predict(train_pitch, next_x).TLP())/2
        result = [next_x, next_yaw, next_pitch]
        return result


class WAVGTLPpolicy(Policy):
    def __init__(self):
        self.name = "WAVGTLP"

    def predict(self, train, next_x):
        sample = np.array(train, dtype=float)
        train_yaw = sample[:, [0, 1]]
        train_pitch = sample[:, [0, 2]]
        next_yaw = (Predict(train_yaw, next_x).WAVG() + Predict(train_yaw, next_x).TLP())/2
        next_pitch = (Predict(train_pitch, next_x).WAVG() + + Predict(train_pitch, next_x).TLP())/2
        result = [next_x, next_yaw, next_pitch]
        return result



if __name__ == '__main__':
    #mylist = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mylist = [[5326.0, -12.3890748658605, 11.523517568790721], [5327.0, -10.384070201245247, 11.45449598353632], [5328.0, -8.608059533955629, 11.394406305778014], [5329.0, -7.097359189519622, 11.301050841766155], [5330.0, -6.223229960068978, 11.254861712255824], [5331.0, -5.597096860132327, 11.218834186379418], [5332.0, -5.293508535643829, 11.212512215546546], [5333.0, -5.0836695272001675, 11.206169646071539], [5334.0, -4.967410681003645, 11.178228563910112], [5335.0, -4.804377309862987, 11.105575913366344], [5336.0, -4.660038893840626, 11.03940647807385], [5337.0, -4.463713906708367, 10.89286943908784], [5338.0, -4.299891492503118, 10.489622674007633], [5339.0, -4.1467942699527365, 9.502468567875649], [5340.0, -3.886883388561857, 7.274153127393775], [5341.0, -3.511287087903121, 4.641880688450106], [5342.0, -2.9797257139421274, 1.7611360682818151], [5343.0, -2.27062617302063, -0.8465477515328703], [5344.0, -1.432374387817458, -3.343006973773896], [5345.0, -0.638420813776495, -5.367816419421864], [5346.0, 0.2585896774629268, -7.326178259878411], [5347.0, 0.9531802052284672, -9.08398325172042], [5348.0, 1.5700208029399545, -10.939127070911486], [5349.0, 2.12310938035086, -12.570274209770249], [5350.0, 2.648490090299449, -14.228010337098825], [5351.0, 3.0070703017087164, -15.631560504135964], [5352.0, 3.2696714375802998, -16.995518860011433], [5353.0, 3.4843868881074482, -18.583742556391986], [5354.0, 3.6977559162906704, -20.184206582349667], [5355.0, 4.042100359523031, -22.178881135214386], [5356.0, 4.632505750783632, -25.155382713115877], [5357.0, 5.339009862628743, -28.73321270089561], [5358.0, 6.002252897617649, -31.996971085312374], [5359.0, 6.592726256064153, -35.00454135696438], [5360.0, 6.88138232583071, -36.67244139413293], [5361.0, 7.007907025717512, -37.6261562038946], [5362.0, 6.976455029795067, -38.98814150945812], [5363.0, 6.7646514775717135, -40.68412393856859], [5364.0, 6.285748846925106, -42.86111173366025], [5365.0, 5.64720903447988, -44.318764249838885], [5366.0, 4.856834264125983, -45.67588067276769], [5367.0, 3.995245755428891, -47.50338128345992], [5368.0, 3.104145467179012, -49.683516546865256], [5369.0, 2.2839238747382393, -51.91541640363288], [5370.0, 1.6139027401774306, -53.65297706600833], [5371.0, 1.1019642500864328, -55.04637439178447], [5372.0, 0.8290050200904628, -55.72735187852634], [5373.0, 0.6331130631793992, -56.22056673740016], [5374.0, 0.4585084577165318, -57.04774259496589], [5375.0, 0.3289340081200875, -58.15297764634156], [5376.0, 0.22540157533696456, -59.543770825244295], [5377.0, 0.05221778927252686, -61.487919193354294], [5378.0, 0.05579298970830392, -63.415047369305576], [5379.0, 0.32588739695011604, -64.7183403944649], [5380.0, 0.9722972641002321, -65.72534847916221], [5381.0, 1.7632368008703054, -66.63540623446706], [5382.0, 2.6408159668093787, -67.54820058566521], [98.0, 3.5675894893170383, -68.48220892925157], [99.0, 4.206109153811407, -69.04583826284667], [100.0, 4.724055738725603, -69.38705019374113]]
    print(cube360policy().predict(mylist, 100))


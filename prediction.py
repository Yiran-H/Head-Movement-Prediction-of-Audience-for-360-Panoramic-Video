# 单独对yaw或者pitch的预测
#一些必要的包
from collections import deque
import numpy as np
import math
#机器学习包
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

from sklearn import svm
#深度学习包
#import tensorflow as tf



class Predict:
    def __init__(self, train, next_x):
        self.train = train
        self.next_x = float(next_x)

    # 最后样本复制-Last Sample Replication
    def LSR(self):
        sample = deque(self.train)
        next_y = sample.pop()[1]
        return next_y

    # 线性回归-Linear regression
    def LR(self):
        sample = np.asarray(self.train, dtype=float)
        train_x = np.transpose(sample[:, 0])[:, np.newaxis]
        train_y = np.transpose(sample[:, 1])[:, np.newaxis]
        model = LinearRegression()
        model.fit(train_x, train_y)
        next_y = np.transpose(model.predict([[self.next_x]]))[0][0]
        return next_y

    # 加权线性回归
    def WLR(self, k=13):
        sample = np.asarray(self.train, dtype=float)
        featureArr = []
        labelArr = []
        for i in range(len(sample)):
            featureArr.append([1.0, sample[i][0]])
            labelArr.append([sample[i][1]])
        featureArr = np.array(featureArr)
        labelArr = np.array(labelArr)
        w = np.mat(np.eye(np.size(labelArr)))
        # print(featureArr)
        for i in range(np.size(labelArr)):
            x = self.next_x - featureArr[i][1]
            # print(x, x.T)
            # w[i, i] = i*i*i + 1
            w[i, i] = math.exp(abs(x * x.T) / (-2 * k * k))
            # w[i, i] = i + 1
        # print(featureArr.T)
        # print(w)
        theta = np.linalg.pinv(featureArr.T @ w @ featureArr) @ featureArr.T @ w @ labelArr
        # print(theta.shape)
        # print('ans=', theta)
        theta = theta.tolist()
        next_y = theta[0][0] + theta[1][0] * self.next_x
        # print(featureArr, '\n', labelArr)
        return next_y

    # 支持向量回归
    def SVR(self, gamma, C):
        sample = np.asarray(self.train, dtype=float)
        train_x = np.transpose(sample[:, 0])[:, np.newaxis]
        train_y = sample[:, 1]
        model = svm.SVR(gamma=gamma, C=C)  # gamma='scale'或‘auto'
        model.fit(train_x, train_y)
        next_y = np.transpose(model.predict([[self.next_x]]))[0]
        return next_y

    # 岭回归
    def Ridge(self, alpha):
        sample = np.asarray(self.train, dtype=float)
        train_x = np.transpose(sample[:, 0])[:, np.newaxis]
        train_y = np.transpose(sample[:, 1])[:, np.newaxis]
        model = Ridge(alpha=alpha)
        model.fit(train_x, train_y)
        next_y = np.transpose(model.predict([[self.next_x]]))[0][0]
        return next_y

    def Adaboost(self):
        sample = np.asarray(self.train, dtype=float)
        train_x = np.transpose(sample[:, 0])[:, np.newaxis]
        train_y = sample[:, 1]
        model = ensemble.AdaBoostRegressor(random_state=100, n_estimators=10)
        model.fit(train_x, train_y)
        next_y = np.transpose(model.predict([[self.next_x]]))[0]
        return next_y

    def RandomForest(self, n_estimators, max_depth):
        sample = np.asarray(self.train, dtype=float)
        train_x = np.transpose(sample[:, 0])[:, np.newaxis]
        train_y = sample[:, 1]
        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        reg.fit(train_x, train_y)
        next_y = np.transpose(reg.predict([[self.next_x]]))[0]
        return next_y

    def Voting(self):
        sample = np.asarray(self.train, dtype=float)
        train_x = np.transpose(sample[:, 0])[:, np.newaxis]
        train_y = np.transpose(sample[:, 1])
        #r1 = LinearRegression()
        r2 = Ridge(150, fit_intercept=True, max_iter=1000, random_state=10)
        r3 = ensemble.RandomForestRegressor(max_depth=100, random_state=100, n_estimators=10)
        er = VotingRegressor([('Ridge', r2), ('RandomForest', r3)])
        er.fit(train_x, train_y)
        next_y = np.transpose(er.predict([[self.next_x]]))[0]
        return next_y


    '''
    # 定义一个添加层的函数
    def add_layer(self, input_, in_size, out_size, activation_funtion=None):
        
        :param input_: 输入的tensor
        :param in_size: 输入的维度，即上一层的神经元个数
        :param out_size: 输出的维度，即当前层的神经元个数即当前层的
        :param activation_funtion: 激活函数
        :return: 返回一个tensor
        
        weight = tf.Variable(tf.random_normal([in_size, out_size]))  # 权重，随机的in_size*out_size大小的权重矩阵
        biase = tf.Variable(tf.zeros([1, out_size]) + 0.01)  # 偏置，1*out_size大小的0.01矩阵，不用0矩阵避免计算出错
        if not activation_funtion:  # 根据是否有激活函数决定输出
            output = tf.matmul(input_, weight) + biase
        else:
            output = activation_funtion(tf.matmul(input_, weight) + biase)
        return output
    
    # 对数回归
    def NN(self):
        sample = np.asarray(self.train, dtype=float)
        x_data = np.array(sample[:, 0])[:, np.newaxis]
        #print(type(x_data), x_data)
        y_data = np.transpose(sample[:, 1])[:, np.newaxis]
        #print(type(y_data), y_data)
        next_data = np.asarray([[self.next_x]])
        #print(type(next_data))

        # 创建占位符用于minibatch的梯度下降训练,建议数据类型使用tf.float32、tf.float64等浮点型数据
        x_in = tf.placeholder(tf.float32, [None, 1])
        y_in = tf.placeholder(tf.float32, [None, 1])

        # 定义隐藏层,输入为原始数据，特征为1，所以输入为1个神经元，输出为4个神经元
        layer1 = self.add_layer(x_in, 1, 4, tf.nn.relu)

        # 定义输出层,输入为layer1返回的tensor，输入为4个神经元，输出为1个神经元，激活函数为ReLU
        predict = self.add_layer(layer1, 4, 1)
        # 定义损失函数
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_in - predict), axis=[1]))  # tf.reduce_sum的axis=[1]表示按列求和

        # 定义训练的优化方式为梯度下降
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 学习率为0.1

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # 训练1000次
            for step in range(1000):
                # 执行训练,因为有占位符所以要传入字典，占位符的好处是可以用来做minibatch训练，这里数据量小，直接传入全部数据来训练
                sess.run(train, feed_dict={x_in: x_data, y_in: y_data})
                # 每50步输出一次loss
                if step % 49 == 0:
                    print(sess.run(loss, feed_dict={x_in: x_data, y_in: y_data}))
            predict_value = sess.run(predict, feed_dict={x_in: x_data})  # 先要获得预测值
        print(predict_value)
        return predict_value

    '''


    # 截断的线性预测-truncated linear prediction
    def TLP(self):
        mon_window = 2
        sample = deque(self.train)
        #print("顺序", sample)
        sample.reverse()
        token = deque()
        #print("逆序", sample)
        for i in range(len(sample) - 1):
            comparison = sample[i][1] - sample[i+1][1]
            if comparison > 0:
                token.append(1)
            elif comparison < 0:
                token.append(-1)
            else:
                token.append(0)
        #print(token)
        key = 0
        for i in range(len(token)):
            if token[i] is not 0:
                key = token[i]
                break
        result = deque()
        temp = sample.popleft()
        result.appendleft(temp)
        for i in range(len(token)):
            temp = sample.popleft()
            temp_key = token.popleft()
            if temp_key == key or temp_key == 0:
                result.appendleft(temp)
            else:
                break
        #print(token)
        #print(result, '\n', len(result))
        token.clear()
        sample.clear()
        # 线性回归部分
        if len(result) >= mon_window:
            train = np.asarray(result, dtype=float)
            result.clear()
            train_x = np.transpose(train[:, 0])[:, np.newaxis]
            train_y = np.transpose(train[:, 1])[:, np.newaxis]
            model = LinearRegression()
            model.fit(train_x, train_y)
            next_y = np.transpose(model.predict([[self.next_x]]))[0][0]
            return next_y
        # 不满足单调窗口，回退到最后样本
        else:
            next_y = result.pop()[1]
            result.clear()
            return next_y


    # 平均值
    def AVG(self):
        sample = np.asarray(self.train, dtype=float)
        train_y = np.transpose(sample[:, 1])
        result = sum(train_y, 0)/len(train_y)
        return result

    # 加权平均数
    def WAVG(self):
        sample = np.asarray(self.train, dtype=float)
        train_y = deque(np.transpose(sample[:, 1]))
        length = len(train_y)
        sum = (1 + length) * length / 2
        result = 0
        for i in range(length):
            temp = train_y.popleft()
            result = result + temp * (i + 1) / sum
        return result

    # 指数加权平均数
    def IWAVG(self):
        sample = np.asarray(self.train, dtype=float)
        train_y = deque(np.transpose(sample[:, 1]))
        length = len(train_y)
        sum = (1 - math.pow(2, length)) / (1 - 2)
        result = 0
        for i in range(length):
            temp = train_y.popleft()
            result = result + temp * math.pow(2, i) / sum
        return result





if __name__ == '__main__':
    my_list = [[1, 1], [2, 2], [3, 4], [4, 8], [5, 16], [6, 32], [7, 64], [8, 128], [9, 256], [10, 512]]
    a = [1, 2, 4, 8, 16, 32, 7, 64, 128, 256, 512]
    print('平均', sum(a)/len(a))
    print(Predict(my_list, 11).Voting())

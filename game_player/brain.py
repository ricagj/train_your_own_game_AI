# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb
# https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DQN3

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import os
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(tf.config.experimental.get_device_details(gpus[0])['device_name'])

# ---------- 以下根据 control_keyboard_keys.py 里定义的函数来导入 ----------
from game_player.control_keyboard_keys import W, S, A, D
# ---------- 以上根据 control_keyboard_keys.py 里定义的函数来导入 ----------

# ---*---

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['observation', 'action', 'reward', 'next_observation']
        )
        self.i = 0    # 行索引
        self.count = 0    # 经验存储数量
        self.capacity = capacity    # 经验容量

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity    # 更新行索引
        self.count = min(self.count + 1, self.capacity)    # 保证数量不会超过经验容量

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

# ---*---

# DoubleDQN
class DoubleDQN:
    def __init__(
        self,
        in_height,      # 图像高度
        in_width,       # 图像宽度
        in_channels,    # 颜色通道数量
        outputs,        # 动作数量
        lr,             # 学习率
        gamma,    # 奖励衰减
        replay_memory_size,     # 记忆容量
        replay_start_size,      # 开始经验回放时存储的记忆量，到达最终探索率后才开始
        batch_size,             # 样本抽取数量
        update_freq,                   # 训练评估网络的频率
        target_network_update_freq,    # 更新目标网络的频率
        save_weights_path = None,    # 指定模型权数保存的路径。默认为None，不保存。
        load_weights_path = None     # 指定模型权重加载的路径。默认为None，不加载。
    ):
        self.in_height   = in_height
        self.in_width    = in_width
        self.in_channels = in_channels
        self.outputs     = outputs
        self.lr          = lr,

        self.gamma = gamma

        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.update_freq = update_freq
        self.target_network_update_freq = target_network_update_freq

        self.save_weights_path = save_weights_path
        self.load_weights_path = load_weights_path

        self.min_epsilon = 0.1    # 最终探索率

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.replayer = DQNReplayer(self.replay_memory_size)    # 经验回放

        self.step = 0    # 计步

    # 评估网络和目标网络的构建方法
    def build_network(self):
        Input = tf.keras.Input(shape=[self.in_height, self.in_width, self.in_channels])

        # -------------------- 以下务必自己修改 --------------------
        # 第 1 层 卷积层和最大池化层
        conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(Input)
        pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_1)
        
        # 第 2 层 卷积层和最大池化层
        conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(pool_1)
        pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_2)

        # 你要是觉得不够，可以自己增加卷积层和池化层，觉得太多了就删掉
        
        # 扁平化层
        flat = tf.keras.layers.Flatten()(pool_2)

        # 第 1 层 全连接层
        dense_1 = tf.keras.layers.Dense(1, activation=tf.nn.relu)(flat)
        dense_1 = tf.keras.layers.BatchNormalization()(dense_1)

        # 第 2 层 全连接层
        dense_2 = tf.keras.layers.Dense(1, activation=tf.nn.relu)(dense_1)
        dense_2 = tf.keras.layers.BatchNormalization()(dense_2)

        output = dense_2
        # -------------------- 以上务必自己修改 --------------------

        # 输出层
        output = tf.keras.layers.Dense(self.outputs, activation=tf.nn.softmax)(output)

        model = tf.keras.Model(inputs=Input, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(self.lr),    # 你觉得有更好的可以自己改
            loss=tf.keras.losses.MeanSquaredError(),    # 你觉得有更好的可以自己改
        )

        if self.load_weights_path:
            if os.path.exists(self.load_weights_path):
                model.load_weights(self.load_weights_path)
                print('Load ' + self.load_weights_path)
            else:
                print('Nothing to load')

        return model

    # 行为选择方法
    def choose_action(self, observation):

        # 先看运行的步数(self.step)有没有达到开始回放经验的要求(self.replay_start_size)，没有就随机探索
                                                  # 如果已经达到了，就再看随机数在不在最终探索率范围内，在的话也是随机探索
        if self.step <= self.replay_start_size or np.random.rand() < self.min_epsilon:
            q_values = np.random.rand(self.outputs)
            self.who_play = '随机探索'
        else:
            observation = observation.reshape(-1, self.in_height, self.in_width, self.in_channels)
            q_values = self.evaluate_net.predict(observation)[0]
            self.who_play = '模型预测'

        action = np.argmax(q_values)

        # ---------- 以下根据 control_keyboard_keys.py 里定义的函数来修改 ----------

        """
        将所有的动作都用编码成数字，并且数字满足从零开始和正整数的要求。
        例如
            W 移动 前 0
            S 移动 后 1
            A 移动 左 2
            D 移动 右 3
        """

        # 执行动作
        if   action == 0:
            W()
        elif action == 1:
            S()
        elif action == 2:
            A()
        elif action == 3:
            D()
        elif action == 4:    # 等你添加，不需要可以删除
            pass
        # 不够可以添加，注意，一定要是正整数，还要和上一个相邻
        # ---------- 以上根据 control_keyboard_keys.py 里定义的函数来修改 ----------

        return action

    # 学习方法
    def learn(self, verbose=0):

        self.step += 1

        # 当前步数满足更新评估网络的要求
        if self.step % self.update_freq == 0:

            # 当前步数满足更新目标网络的要求
            if self.step % self.target_network_update_freq == 0:
                self.update_target_network() 

            # 经验回放
            observations, actions, rewards, next_observations = self.replayer.sample(self.batch_size)

            # 数据预处理
            observations = observations.reshape(-1, self.in_height, self.in_width, self.in_channels)
            actions = actions.astype(np.int8)
            next_observations = next_observations.reshape(-1, self.in_height, self.in_width, self.in_channels)


            next_eval_qs = self.evaluate_net.predict(next_observations)
            next_actions = next_eval_qs.argmax(axis=-1)

            next_qs = self.target_net.predict(next_observations)
            next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]

            us = rewards + self.gamma * next_max_qs
            targets = self.evaluate_net.predict(observations)
            targets[np.arange(us.shape[0]), actions] = us


            self.evaluate_net.fit(observations, targets, batch_size=1, verbose=verbose)

            self.save_evaluate_network()

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_weights_path)
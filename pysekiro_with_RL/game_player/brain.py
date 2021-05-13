# https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb
# https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DQN3

import threading
import os

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True) if gpus else -1

from game_player.control_keyboard_keys import Waiting_to_learn, J, K, LSHIFT

# ---*---

class DQNReplayer:
    def __init__(self, capacity=1000):
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
        save_weights_path,    # 指定模型权数保存的路径。
        load_weights_path     # 指定模型权重加载的路径。
    ):
        self.in_height   = 224     # 图像高度
        self.in_width    = 224     # 图像宽度
        self.in_channels = 3       # 颜色通道数量
        self.outputs     = 3       # 动作数量
        self.lr          = 0.01    # 学习率

        self.gamma = 0.9    # 奖励衰减

        self.batch_size = 128    # 样本抽取数量

        self.update_freq = 200                   # 训练评估网络的频率
        self.target_network_update_freq = 500    # 更新目标网络的频率

        self.save_weights_path = save_weights_path
        self.load_weights_path = load_weights_path

        self.evaluate_net = self.build_network()    # 评估网络
        self.target_net = self.build_network()      # 目标网络
        self.replayer = DQNReplayer()    # 经验回放

    # 评估网络和目标网络的构建方法
    def build_network(self):
        
        input_shape = [self.in_height, self.in_width, self.in_channels]

        inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        x = tf.cast(inputs, tf.float32)
        outputs = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, weights=None, classes=self.outputs)(x)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        model.compile(
            optimizer=tf.keras.optimizers.Nadam(self.lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
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

        if np.random.rand() < 0.15:
            q_values = np.random.rand(self.outputs)
        else:
            q_values = self.evaluate_net.predict(np.expand_dims(observation, 0))[0]

        action = np.argmax(q_values)

        # 执行动作
        if   action == 0:
            act = J
        elif action == 1:
            act = K
        elif action == 2:
            act = LSHIFT

        self.act_process = threading.Thread(target=act)
        self.act_process.start()

        return action

    # 学习方法
    @Waiting_to_learn
    def learn(self):

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


        self.evaluate_net.fit(observations, targets, verbose=0)

        self.save_evaluate_network()

    # 更新目标网络权重方法
    def update_target_network(self):
        self.target_net.set_weights(self.evaluate_net.get_weights())

    # 保存评估网络权重方法
    def save_evaluate_network(self):
        self.evaluate_net.save_weights(self.save_weights_path)
from collections import deque
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from game_player.brain import DoubleDQN
from game_player.others import get_status, roi
from game_player.grab_screen import get_game_screen
from game_player.detect_keyboard_keys import key_check

# ---*---

class RewardSystem:
    def __init__(self):
        self.total_reward = 0    # 当前积累的 reward
        self.reward_history = list()    # reward 的积累过程

    # 获取奖励
    def get_reward(self, cur_status, next_status):
        """
        cur_status 和 next_status 都是存放状态信息的列表，内容：[状态1, 状态2, 状态3, 状态4]
        cur_status  表示当前的人物状态
        next_status 表示未来的人物状态
        """
        if sum(next_status) == 0:

            reward = 0
        else:
            # ---------- 以下根据 others.py 里定义的函数来修改 ----------
            # 通过列表索引的方式，取出相应的信息，用未来的状态信息减去当前的状态信息，得到状态变化值
            s1 = next_status[0] - cur_status[0]
            s2 = next_status[1] - cur_status[1]
            s3 = next_status[2] - cur_status[2]
            s4 = next_status[3] - cur_status[3]

            """
            注意，未来 - 现在
            假如你现在生命值 130（现在），过了一会生命值变成 63（未来）
            计算：s = 63 - 130, s = -67, 生命值降低了67，生命值减低应该惩罚，那么s完全可以当成得分，得到 -67 分。
            
            再假如Boss 现在生命值 112（现在）， 过了一会生命值变成 102（未来）
            计算：s = 102 - 112, s = -10, Boss生命值降低了10，应该奖励才对，但是s为负值，所以要乘上 -1 ，这样才能得到正常的分数。
            
            请根据具体的游戏来定义，不要生搬硬套，别搞得HP掉了还加分
            """
            # 示例 定义得分
            s1 *=  1    # 与 奖励 呈正相关，所以 +
            s2 *= -1    # 与 惩罚 呈正相关，所以 -
            s3 *= -1    # 与 惩罚 呈正相关，所以 -
            s4 *=  1    # 与 奖励 呈正相关，所以 +

            reward = s1 + s2 + s3 +s4
            # ---------- 以上根据 others.py 里定义的函数来修改 ----------

        self.total_reward += reward
        self.reward_history.append(self.total_reward)

        return reward

    def save_reward_curve(self, save_path='reward.png'):
        total = len(self.reward_history)
        if total > 100:
            plt.rcParams['figure.figsize'] = 100, 15
            plt.plot(np.arange(total), self.reward_history)
            plt.ylabel('reward')
            plt.xlabel('training steps')
            plt.xticks(np.arange(0, total, int(total/100)))
            plt.savefig(save_path)
            plt.show()

# -------------------- 一些参数，根据实际情况修改 --------------------

x   = 250
x_w = 550
y   = 75
y_h = 375

in_depth    = 1
in_height   = 50    # 图像高度
in_width    = 50    # 图像宽度
in_channels = 1     # 颜色通道数量
outputs = 5     # 动作数量
lr = 0.001      # 学习率
gamma = 0.99    # 奖励衰减
replay_memory_size = 10000    # 记忆容量
replay_start_size = 500       # 开始经验回放时存储的记忆量，到达最终探索率后才开始
batch_size = 16               # 样本抽取数量
update_freq = 200                   # 训练评估网络的频率
target_network_update_freq = 500    # 更新目标网络的频率

# -------------------- 一些参数，根据实际情况修改 --------------------

class Agent:
    def __init__(
        self,
        save_memory_path=None,
        load_memory_path=None,
        save_weights_path=None,
        load_weights_path=None
    ):
        self.save_memory_path = save_memory_path     # 指定记忆/经验保存的路径。默认为None，不保存。
        self.load_memory_path = load_memory_path     # 指定记忆/经验加载的路径。默认为None，不加载。
        self.brain = DoubleDQN(
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
            save_weights_path = save_weights_path,    # 指定模型权重保存的路径。默认为None，不保存。
            load_weights_path = load_weights_path     # 指定模型权重加载的路径。默认为None，不加载。
        )
        if not save_weights_path:    # 注：默认也是测试模式，若设置该参数，就会开启训练模式
            self.train = False
            self.brain.step = self.brain.replay_start_size + 1
        else:
            self.train = True

        self.reward_system = RewardSystem()

        self.i = 0    # 计步器

        self.screens = deque(maxlen = in_depth * 2)    # 用双端队列存放图像

        if self.load_memory_path:
            self.load_memory()    # 加载记忆/经验

    def load_memory(self):
        if os.path.exists(self.load_memory_path):
            last_time = time.time()
            self.brain.replayer.memory = pd.read_json(self.load_memory_path)    # 从json文件加载记忆/经验。 
            print(f'Load {self.load_memory_path}. Took {round(time.time()-last_time, 3):>5} seconds.')

            i = self.brain.replayer.memory.action.count()
            self.brain.replayer.i = i
            self.brain.replayer.count = i
            self.brain.step = i

        else:
            print('No memory to load.')

    def get_S(self):

        for _ in range(in_depth):
            self.screens.append(get_game_screen())    # 先进先出，右进左出

    def img_processing(self, screens):
        return np.array([cv2.resize(roi(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), x, x_w, y, y_h), (in_height, in_width)) for screen in screens])

    def round(self):

        observation = self.img_processing(list(self.screens)[:in_depth])    # S

        action = self.action = self.brain.choose_action(observation)    # A

        self.get_S()    # 观测

        reward = self.reward_system.get_reward(
            cur_status=get_status(list(self.screens)[in_depth - 1]),
            next_status=get_status(list(self.screens)[in_depth * 2 - 1])
        )    # R

        next_observation = self.img_processing(list(self.screens)[in_depth:])    # S'

        if self.train:

            self.brain.replayer.store(
                observation,
                action,
                reward,
                next_observation
            )

            if self.brain.replayer.count >= self.brain.replay_start_size:
                self.brain.learn()

    def run(self):

        paused = True
        print("Ready!")

        while True:

            last_time = time.time()
            
            keys = key_check()
            
            if paused:
                if 'T' in keys:
                    self.get_S()
                    paused = False
                    print('\nStarting!')

            else:    # 按 'T' 之后，马上下一轮就进入这里

                self.i += 1

                self.round()

                print(f'\r {self.brain.who_play:>4} , step: {self.i:>6} . Loop took {round(time.time()-last_time, 3):>5} seconds. action {self.action:>1} , total_reward: {self.reward_system.total_reward:>10.3f} , memory: {self.brain.replayer.count:7>} .', end='')
 
                if 'P' in keys:
                    if self.train:
                        self.brain.save_evaluate_network()    # 学习完毕，保存网络权重
                        self.brain.replayer.memory.to_json(self.save_memory_path)    # 保存经验
                    self.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
                    break

        print('\nDone!')
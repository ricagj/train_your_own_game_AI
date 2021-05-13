from collections import deque
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd

from game_player.brain import DoubleDQN
from game_player.control_keyboard_keys import Lock_On, Reset_Self_HP
from game_player.detect_keyboard_keys import key_check
from game_player.grab_screen import get_game_screen
from game_player.others import get_status, roi

# ---*---

class RewardSystem:
    def __init__(self):
        self.total_reward = 0    # 当前积累的 reward
        self.reward_history = list()    # reward 的积累过程

    # 获取奖励
    def get_reward(self, cur_status, next_status):
        if sum(next_status) == 0:
            reward = 0
        else:
            reward = np.sum((next_status - cur_status) * [1/257, -1/90, -1/168*10, 1/155*10])
        if next_status[0] < 10:
            Reset_Self_HP()    # 重置自身生命值。注：先把修改器开着，不然这一步无效
            time.sleep(0.5)
            Lock_On()    # 重置视角/固定目标
        self.total_reward += reward
        self.reward_history.append(self.total_reward)
        return reward

    def save_reward_curve(self):
        plt.rcParams['figure.figsize'] = 100, 15
        plt.plot(np.arange(len(self.reward_history)), self.reward_history)
        plt.ylabel('reward')
        plt.xlabel('training steps')
        plt.savefig('reward.png')
        plt.show()

class Agent:
    def __init__(self, train, save_weights_path=None, load_weights_path=None):
        self.train = train
        if self.train:
            self.reward_system = RewardSystem()
        self.brain = DoubleDQN(save_weights_path, load_weights_path)
        self.screens = deque(maxlen = 2)    # 用双端队列存放图像
        self.i = 1    # 计步器

        self.run()

    def round(self):
        observation = roi(self.screens[0], 288, 512, 113, 337)    # S
        action = self.brain.choose_action(observation)    # A
        time.sleep(0.2)
        self.screens.append(get_game_screen())    # 先进先出，右进左出    # 观测
        if self.train:
            self.brain.replayer.store(
                observation,
                action,
                self.reward_system.get_reward(get_status(self.screens[0]), get_status(self.screens[-1])),    # R,
                roi(self.screens[-1], 288, 512, 113, 337)    # S'
            )
            if self.i % self.brain.update_freq == 0:    # 当前步数满足更新评估网络的要求
                self.brain.learn()
                if self.i % self.brain.target_network_update_freq == 0:    # 当前步数满足更新目标网络的要求
                    self.brain.update_target_network()

    def run(self):
        paused = True
        print("Ready!", end='')
        while True:
            last_time = time.time()
            keys = key_check()
            if paused:
                if 'T' in keys:
                    self.screens.append(get_game_screen())    # 先进先出，右进左出
                    paused = False
                    print('Starting!')
            else:    # 按 'T' 之后，马上下一轮就进入这里
                self.round()
                print(f'\rstep: {self.i:>4} , Loop took {round(time.time()-last_time, 3):>5} seconds.', end='')
                self.i += 1
                if 'P' in keys or max([np.sum(screen == 0) for screen in self.screens]) >= 224 * 224 * 3 * 0.1:
                    if self.train:
                        self.brain.save_evaluate_network()    # 学习完毕，保存网络权重
                        self.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
                    print('\nDone!')
                    break
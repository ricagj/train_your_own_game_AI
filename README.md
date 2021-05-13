## 训练属于自己的游戏AI v0.3.0

## 本次更新
只狼专用代码更新，通用代码未更新

## 第 1 个实例 《只狼：影逝二度》

[v0.2.0 代码](https://github.com/ricagj/train_your_own_game_AI/tree/d0a553b53559ff7cf6decea9252a139013a70746/pysekiro_with_RL)  
[v0.2.0 教程](https://github.com/ricagj/train_your_own_game_AI/blob/d0a553b53559ff7cf6decea9252a139013a70746/pysekiro_with_RL/sekiro.ipynb)

## 第 2 个实例 《Muse Dash》

正在进行

## 项目结构

- game_player
    - \__init__.py
    - brain.py
    - control_keyboard_keys.py
    - detect_keyboard_keys.py
    - grab_screen.py
    - others.py
    - run.py

## 安装

#### 安装 Anaconda3

https://www.anaconda.com/  

#### 创建虚拟环境和安装依赖

~~~shell
conda create -n game_AI python=3.8
conda activate game_AI
conda install pandas
conda install matplotlib
conda install pywin32
pip install opencv-python>=4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow>=2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c conda-forge jupyterlab
~~~
## 训练属于自己的游戏AI

# 正在调试中...
# 正在调试中...
# 正在调试中...

# 正在更新中...
# 正在修改 pysekiro_with_RL 的代码，使之成为本项目的第一个实例，也是第一个示例
# 下一个实例是：Muse Dash

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
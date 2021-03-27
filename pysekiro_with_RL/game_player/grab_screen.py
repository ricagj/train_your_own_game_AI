import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

# ---*---

def grab_screen(x, x_w, y, y_h):

    # 获取桌面
    hwin = win32gui.GetDesktopWindow()

    w = x_w - x
    h = y_h - y

    # 返回句柄窗口的设备环境、覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    hwindc = win32gui.GetWindowDC(hwin)

    # 创建设备描述表
    srcdc = win32ui.CreateDCFromHandle(hwindc)

    # 创建一个内存设备描述表
    memdc = srcdc.CreateCompatibleDC()

    # 创建位图对象
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, w, h)
    memdc.SelectObject(bmp)
    
    # 截图至内存设备描述表
    memdc.BitBlt((0, 0), (w, h), srcdc, (x, y), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)

    # 内存释放
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# ---------- 注意：以下需要设置 ----------

GAME_WIDTH   = 800    # 游戏窗口宽度
GAME_HEIGHT  = 450    # 游戏窗口高度
white_border = 31     # 游戏边框

# ---------- 注意：以上需要设置 ----------

def get_game_screen():
    return grab_screen(
        x = 0,
        x_w = GAME_WIDTH,
        y = white_border,
        y_h = white_border+GAME_HEIGHT)

# 全屏
FULL_WIDTH = 1920
FULL_HEIGHT = 1080

def get_full_screen():
    return grab_screen(
        x = 0,
        x_w = FULL_WIDTH,
        y = 0,
        y_h = FULL_HEIGHT)
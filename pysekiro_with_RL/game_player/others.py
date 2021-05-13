import cv2
import numpy as np

# ---*---

def roi(img, x, x_w, y, y_h):
    return img[y:y_h, x:x_w]

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append([x, y])
        try:
            cv2.imshow("window", img)
        except NameError:
            pass
    return vertices

def get_xywh(img):
    global vertices
    vertices = []

    print('Press "ESC" to quit. ')
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("window", on_EVENT_LBUTTONDOWN)
    while True:
        cv2.imshow("window", img)
        if cv2.waitKey(0)&0xFF==27:
            break
    cv2.destroyAllWindows()

    if len(vertices) != 4:
        print("vertices number not match")
        return -1

    x = min(vertices[0][0], vertices[1][0])
    x_w = max(vertices[2][0], vertices[3][0])
    y = min(vertices[1][1], vertices[2][1])
    y_h = max(vertices[0][1], vertices[3][1])

    cv2.imshow('img', img)
    cv2.imshow('roi(img)', roi(img, x, x_w, y, y_h))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'\n x={x}, x_w={x_w}, y={y}, y_h={y_h}\n')

# ---*---

# debug

# import cv2
# from game_player.grab_screen import get_game_screen
# from game_player.others import roi
# from game_player.detect_keyboard_keys import key_check
# import numpy as np
# Count = dict()
# paused = True
# print("Ready!")
# while True:
#     keys = key_check()
#     if paused:
#         if 'T' in keys:
#             paused = False
#     else:
#         img = get_game_screen()
#         Self_Posture = get_Self_Posture(img)
#         print(f'\r {str(Self_Posture):<10}', end='')
#         if 'P' in keys:
#             break
# sorted(Count.items(), key = lambda kv:(kv[1], kv[0]))

# ---*---

# 获取自身生命
def get_Self_HP(img):
    img_roi = roi(img, x=48, x_w=305, y=409, y_h=409+1)

    b, g ,r =cv2.split(img_roi)    # 颜色通道分离

    retval, img_th = cv2.threshold(g, 50, 255, cv2.THRESH_TOZERO)             # 图像阈值处理，像素点的值低于50的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0

    target_img = img_th[0]
    if 0 in target_img:
        Self_HP = np.argmin(target_img)
    else:
        Self_HP = len(target_img)

    return Self_HP

# 获取自身架势
def get_Self_Posture(img):
    # global Count
    img_roi = roi(img, x=401, x_w=491, y=389, y_h=389+1)
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离
    
    key_pixel = r[0][0]
    if key_pixel in [159, 160, 161, 162, 163, 254, 255]:
        # Count[key_pixel] = Count.get(key_pixel, 0) + 1
        canny = cv2.Canny(cv2.GaussianBlur(r,(3,3),0), 0, 100)
        Self_Posture = np.argmax(canny)
    else:
    	Self_Posture = 0

    return Self_Posture

# 获取目标生命
def get_Target_HP(img):
    img_roi = roi(img, x=48, x_w=216, y=41, y_h=41+1)

    b, g ,r =cv2.split(img_roi)    # 颜色通道分离

    retval, img_th = cv2.threshold(g, 25, 255, cv2.THRESH_TOZERO)             # 图像阈值处理，像素点的值低于25的设置为0
    retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0

    target_img = img_th[0]
    if 0 in target_img:
        Target_HP = np.argmin(target_img)
    else:
        Target_HP = len(target_img)
    
    return Target_HP

# 获取目标架势
def get_Target_Posture(img):
#     global Count
    img_roi = roi(img, x=401, x_w=556, y=29, y_h=29+1)
    b, g ,r =cv2.split(img_roi)    # 颜色通道分离

    key_pixel = r[0][0]
    if key_pixel in [190, 255] + list(range(197, 220+1)):
#         Count[key_pixel] = Count.get(key_pixel, 0) + 1
        canny = cv2.Canny(cv2.GaussianBlur(r,(3,3),0), 0, 100)
        Target_Posture = np.argmax(canny)
    else:
        Target_Posture = 0

    return Target_Posture

# ---*---

def get_status(img):
    return np.array([get_Self_HP(img), get_Self_Posture(img), get_Target_HP(img), get_Target_Posture(img)])
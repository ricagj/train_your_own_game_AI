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

# def get_P(img):
#     img = roi(img, x=402, x_w=484, y=388, y_h=390)
#     canny = cv2.Canny(cv2.GaussianBlur(img,(3,3),0), 0, 100)
#     value = canny.argmax(axis=-1)
#     return np.median(value)

def get_state_1(img):    # 自己改
    return 0

def get_state_2(img):    # 自己改
    return 0

def get_state_3(img):    # 自己改
    return 0

def get_state_4(img):    # 自己改
    return 0

# 不够就自己添加，多了就自己删除

def get_status(img):
    return get_state_1(img), get_state_2(img), get_state_3(img), get_state_4(img)    # 这里也要改成相应的函数名

# ---*---
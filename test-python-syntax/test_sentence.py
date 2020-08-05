import numpy as np  # 提供矩阵运算功能的库

base_anchor = np.array([1, 1, 16, 16]) - 1


# 枚举一个anchor的各种宽高比，anchor[0 0 15 15],ratios[0.5,1,2]
def _ratio_enum(anchor, ratios):
    """
    :param anchor: 输入参数为一个anchor(四个坐标值表示)
    :param ratios: 和三种宽高比例（0.5,1,2）
    :return:
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)  # 返回宽高和中心坐标，w:16,h:16,x_ctr:7.5,y_ctr:7.5
    size = w * h  # size:16*16=256
    size_ratios = size / ratios  # 256/ratios[0.5,1,2]=[512,256,128]
    # round()方法返回x的四舍五入的数字，sqrt()方法返回数字x的平方根
    ws = np.round(np.sqrt(size_ratios))  # ws:[23 16 11]
    hs = np.round(ws * ratios)  # hs:[12 16 22],ws和hs一一对应。as:23&12
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)  # 给定一组宽高向量，输出各个预测窗口
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    # anchor:存储了窗口左上角，右下角的坐标
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)  # anchor中心点坐标
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


# 给定一组宽高向量，输出各个anchor，即预测窗口，**输出anchor的面积相等，只是宽高比不同**
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    :param ws:
    :param hs:
    :param x_ctr:
    :param y_ctr:
    :return:
    """

    # ws:[23 16 11]，hs:[12 16 22],ws和hs一一对应。
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]  # newaxis:将数组转置
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),  # hstack、vstack:合并数组
                         y_ctr - 0.5 * (hs - 1),  # anchor：[[-3.5 2 18.5 13]
                         x_ctr + 0.5 * (ws - 1),  # [0  0  15  15]
                         y_ctr + 0.5 * (hs - 1)))  # [2.5 -3 12.5 18]]
    return anchors

print(base_anchor)
w, h, x_ctr, y_ctr = _whctrs(base_anchor)  # 返回宽高和中心坐标，w:16,h:16,x_ctr:7.5,y_ctr:7.5
ratios=[0.5, 1, 2]

ratio_anchors = _ratio_enum(base_anchor, ratios)  # 枚举各种宽高比
print(ratio_anchors)

w, h, x_ctr, y_ctr = _whctrs([-3.5, 2, 18.5,13])  # 返回宽高和中心坐标，w:16,h:16,x_ctr:7.5,y_ctr:7.5
size = w * h
print(size)

print(np.arange(3, 6))
print(2 ** np.arange(3, 6))
#
# print('w, h, x_ctr, y_ctr = ',w, h, x_ctr, y_ctr)
# size = w * h  # size:16*16=256
# print('size= ',size)
# size_ratios = size / ratios  # 256/ratios[0.5,1,2]=[512,256,128]
# print(' ratios =',ratios)
# print(' size / ratios =',size_ratios)
# sqrt_size_ratios = np.sqrt(size_ratios)
# print('np.sqrt(size_ratios) = ',sqrt_size_ratios)
# ws = np.round(sqrt_size_ratios)
# print('np.round(sqrt_size_ratios) =ws ',ws)
# hs = np.round(ws * ratios)
# print('np.round(ws * ratios = hs',hs)
# ws = ws[:, np.newaxis]  # newaxis:将数组转置
# print('对ws进行np.newaxis',ws)
# hs = hs[:, np.newaxis]
# print('对hs进行np.newaxis',hs)
# first = (x_ctr - 0.5 * (ws - 1),  # hstack、vstack:合并数组
#                      y_ctr - 0.5 * (hs - 1),  # anchor：[[-3.5 2 18.5 13]
#                      x_ctr + 0.5 * (ws - 1),  # [0  0  15  15]
#                      y_ctr + 0.5 * (hs - 1))
# print(first)
# anchors = np.hstack(first)  # [2.5 -3 12.5 18]]
# print('anchors = ',anchors)

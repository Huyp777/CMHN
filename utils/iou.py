import numpy as np


def ointer(s0, t0, s1, t1):
    the_inter = min(t0, t1) - max(s0, s1)
    the_inter = max(0, the_inter)
    return the_inter


def ounion(s0, t0, s1, t1):
    return max(t0, t1) - min(s0, s1)


def oiou(s0, t0, s1, t1):
    i = ointer(s0, t0, s1, t1)
    u = ounion(s0, t0, s1, t1)
    if i == 0:
        return 0
    return i / u


# def viou(s0, t0, s1, t1):
#     i = inter(s0, t0, s1, t1)
#     u = t0 - s0
#     if i == 0:
#         return 0
#     return i / u


def inter(s0, t0, s1, t1):
    the_inter = np.minimum(t0, t1) - np.maximum(s0, s1)
    the_inter = np.maximum(0, the_inter)
    return the_inter


def union(s0, t0, s1, t1):
    return np.maximum(t0, t1) - np.minimum(s0, s1)


def iou(s0, t0, s1, t1):
    i = inter(s0, t0, s1, t1)
    u = union(s0, t0, s1, t1) + 1e-7
    return i / u


def viou(s0, t0, s1, t1):
    i = inter(s0, t0, s1, t1)
    u = t0 - s0 + 1e-7
    return i / u


def giou(s0, t0, s1, t1):
    the_iou = iou(s0, t0, s1, t1)
    c = union(s0, t0, s1, t1) + 1e-7
    u = c.copy()
    i = inter(s0, t0, s1, t1)
    index = i == 0
    u[index] = (u - np.maximum(s0, s1) + np.minimum(t0, t1))[index]
    return the_iou - 1 + u / c

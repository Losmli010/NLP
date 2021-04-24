# coding: utf-8
import numpy as np


def LaplaceSmoothing(arr):
    """
    拉普拉斯平滑法
    对所有统计次数（add-one）+1
    """
    if arr.size == len(arr):
        smooth_arr = np.array([0] * len(arr), np.int)
        for i in range(len(arr)):
            smooth_arr[i] = arr[i]+1
    else:
        (row, column) = arr.shape
        smooth_arr = np.zeros((row, column), np.int)
        for i in range(row):
            for j in range(column):
                smooth_arr[i][j] = arr[i][j] + 1
    return smooth_arr


def LidstoneSmoothing(arr, λ):
    """
    Lidstone平滑法
     对所有统计次数（add-λ）+λ
    """
    if arr.size == len(arr):
        smooth_arr = np.array([0.0] * len(arr), np.float)
        for i in range(len(arr)):
            smooth_arr[i] = arr[i] + λ
    else:
        (row, column) = arr.shape
        smooth_arr = np.zeros((row, column), np.float)
        for i in range(row):
            for j in range(column):
                smooth_arr[i][j] = arr[i][j] + λ
    return smooth_arr


def SimpleGoodTuring(arr):
    (row, column) = arr.shape
    smooth_arr = np.zeros((row, column), np.float)
    for i in range(row):
        N_zero = list(arr[i]).count(0)
        N_one = list(arr[i]).count(1)
        N_two = list(arr[i]).count(2)
        for j in range(column):
            if arr[i][j] == 0:
                smooth_arr[i][j] = N_one / N_zero
            elif arr[i][j] == 1:
#                Nc=((c+1)*(list(arr[i]).count(c+1))/(list(arr[i]).count(c))-c*(k+1)*N_k/N_one)/(1-(k+1)*N_k/N_one)
                smooth_arr[i][j] = N_two / (N_one - 2 * N_two)
            else:
                smooth_arr[i][j] = arr[i][j]
    return smooth_arr


def AbsoluteDiscounting(arr, discount=0.75):
    """
    绝对折扣平滑法
    对所有统计次数大于0的项减去-discount
    """
    if arr.size == len(arr):
        smooth_arr = np.array([0.0] * len(arr), np.float)
        N_zero = list(arr).count(0)
        for i in range(len(arr)):
            if arr[i] == 0:
                smooth_arr[i] = (len(arr) - N_zero) * discount / N_zero
            else:
                smooth_arr[i] = arr[i] - discount
    else:
        (row, column) = arr.shape
        smooth_arr = np.zeros((row, column), np.float)
        for i in range(row):
            N_zero = list(arr[i]).count(0)
            for j in range(column):
                if arr[i][j] == 0:
                    smooth_arr[i][j] = (column - N_zero) * discount / N_zero
                else:
                    smooth_arr[i][j] = arr[i][j] - discount
    return smooth_arr


def KneserNeySmoothing(arr):
    pass

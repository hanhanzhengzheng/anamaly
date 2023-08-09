import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *

def calculate_difference_matrix(real_matrix, predicted_matrix):
    '''
    计算实际矩阵和预测矩阵之间的差异矩阵。
    它接受两个参数，real_matrix表示实际矩阵的序列，predicted_matrix表示预测矩阵的序列。
    '''

    # 计算real_matrix和predicted_matrix之间的差异矩阵,创建一个空的列表difference_matrix，用来存储每个时间点的差异矩阵
    difference_matrix = []
    for t in range(len(real_matrix)):
        diff_matrix_t = np.linalg.norm(predicted_matrix[t] - real_matrix[t], axis=1)
        difference_matrix.append(diff_matrix_t)
    return np.array(difference_matrix)

def calculate_threshold(difference_matrix):
    '''
    计算差异矩阵序列中的最大值
    参数为差异矩阵序列
    '''

    threshold = np.max(difference_matrix)
    return threshold

def calculate_anomaly_scores(difference_matrix, threshold):
    '''
    计算差异矩阵中大于阈值的元素的数量，并将其作为异常得分返回
    '''

    # Calculate anomaly scores based on the threshold
    anomaly_scores = np.sum(difference_matrix > threshold, axis=1)
    return anomaly_scores

def calculate_temporary_threshold(anomaly_scores):
    '''
    计算临时阈值的函数。它接收异常得分作为输入，并基于异常得分的四分位数计算临时阈值。
    '''
    # Calculate the temporary threshold based on quartiles of anomaly scores
    q1 = np.percentile(anomaly_scores, 25)
    q3 = np.percentile(anomaly_scores, 75)
    temporary_threshold = 1.5 * (q3 - q1) + q3
    return temporary_threshold

def calculate_decision_threshold(anomaly_scores, temporary_threshold):
    '''
    计算决策阈值的函数。它接收异常得分和临时阈值作为输入，并基于临时阈值计算决策阈值。
    在这个函数中，我们首先将决策阈值设置为异常得分中的最大值，然后与临时阈值进行比较。
    如果决策阈值大于临时阈值，则将其替换为异常得分的75%分位数。
    '''
    # Calculate the decision threshold based on the temporary threshold
    decision_threshold = np.max(anomaly_scores)
    if decision_threshold > temporary_threshold:
        decision_threshold = np.percentile(anomaly_scores, 75)
    return decision_threshold

# # 计算差异矩阵序列
# difference_matrix = calculate_difference_matrix(real_matrix, predicted_matrix)
#
# # 计算差异矩阵的阈值
# threshold = calculate_threshold(difference_matrix)
#
# # 计算异常得分
# anomaly_scores = calculate_anomaly_scores(difference_matrix, threshold)
#
# # 计算临时阈值
# temporary_threshold = calculate_temporary_threshold(anomaly_scores)
#
# # 计算最终的异常得分边界
# decision_threshold = calculate_decision_threshold(anomaly_scores, temporary_threshold)
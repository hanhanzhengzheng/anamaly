import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.set_index('date_time')
    return data

def addLabel (data,train_size) :
    # Load the dataset from a CSV file
    df = data

    # Add a label column with all 0s
    df['label'] = 0

    # Split the dataset into training and testing sets
    train_df, test_df = data_split(df, train_size)

    # 在测试集中随机选择位置以引入异常
    num_anomalies = 20  # Number of anomalies to introduce
    random_positions = np.random.choice(test_df.index, num_anomalies, replace=False)
    #np.random.choice在时间戳索引中随机抽取num_anomalies个数据，返回一维数组，里面是随机抽取的位置

    # 修改具有异常值的选定位置
    for position in random_positions:
        # Get the column names of the features
        feature_columns = df.columns[0:-1]  # Exclude the timestamp and label columns

        # Select a random column to modify
        # 选择四个随机列进行修改
        columns_to_modify = np.random.choice(feature_columns, 4, replace=False)

        for column_to_modify in columns_to_modify:
            # 获取列的分布统计信息
            mean = df[column_to_modify].mean()
            std = df[column_to_modify].std()

            # 在平均值周围的特定范围内生成异常值
            anomaly_value = np.random.normal(loc=mean, scale=std * 0.1)

            # 将异常值赋给相应位置
            test_df.at[position, column_to_modify] = anomaly_value

            # 更新标签列为1，表示异常
            test_df.at[position, 'label'] = 1

    # 将训练集和测试集生成csv文件并存储
    train_df.to_csv('train_dataset.csv', index=False)
    test_df.to_csv('test_dataset.csv', index=False)

    # Convert the datasets to TensorFlow tensors
    train_data = tf.data.Dataset.from_tensor_slices((train_df.iloc[:, 0:-1].astype('float32').values, train_df['label'].astype('float32').values))
    test_data = tf.data.Dataset.from_tensor_slices((test_df.iloc[:, 0:-1].astype('float32').values, test_df['label'].astype('float32').values))

    return train_df,test_df

#处理缺失值
def isNull (data) :
    missing_values = data.isnull().sum()

    # 输出缺失值的统计信息
    print(missing_values)
    data_filled = data

    # 判断数据集中是否有缺失值
    if missing_values.sum() > 0:
        print("数据集中存在缺失值")
        data_filled = data.fillna(data.mean())
    else:
        print("数据集中没有缺失值")
    return data_filled

#处理异常值，将数据集变为正常数据
def isAbnormal (data) :

    # 计算每列的均值和标准差
    median = data.median()

    # 定义异常值的阈值
    threshold = 3

    # 检测异常值
    outliers = (data < (median - threshold * median)) | (data > (median + threshold * median))

    # 将异常值替换为中位数
    data[outliers] = np.nan
    data_filled = data.fillna(median)
    return data

#归一化
def maxminScaler(data) :
    scaler = MinMaxScaler()

    # 对数据进行归一化处理
    temp_data = scaler.fit_transform(data)

    # # 将归一化后的特征列转换为DataFrame
    # normalized_data = pd.DataFrame(temp_data, columns=data.columns)
    #
    # # 将归一化后的数据保存到当前路径
    # normalized_data.to_csv('normalized_data.csv', index=False)
    return temp_data

# 数据切分
def data_split(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

#Build a feature matrix构建特征矩阵
def buildFeature (data,size,step) :
    column_names = np.arange(data.shape[1]).tolist()

    # 设置窗口大小和步长
    window_size = size
    step_size = step

    # 构建特征矩阵
    feature_matrix = []
    label_matrix = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window_data1 = data[i:i + window_size]
        window_matrix1 = np.dot(window_data1.T, window_data1)
        window_data2 = data[i+window_size-1]
        window_matrix2 = np.dot(window_data2.T, window_data2)
        feature_matrix.append(window_matrix1)
        label_matrix.append(window_matrix2)

    # 将特征矩阵转换为数组
    feature_matrix = np.array(feature_matrix)
    label_matrix = np.array(label_matrix)
    return feature_matrix,label_matrix

# 定义特征矩阵拼接方法
def concatenate_feature (data,channl) :
    feature_sequence = []
    for i in range(channl):
        feature_sequence.append(data[i:i + 10])

data = 'PVODdatasets_v1.0/station00.csv'
dataset = pd.read_csv(data)
dataset = dataset.set_index('date_time')
plt.rc("font", family='Microsoft YaHei')
dataset.plot()
# plt.show()
a_train,a_test = addLabel(dataset,0.8)

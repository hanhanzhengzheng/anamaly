import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from model import *

# 读取数据
data = load_data('PVODdatasets_v1.0/station00.csv')
# 处理缺失值
# data = isNull (data)


# 添加标签
train_df,test_df = addLabel(data,0.8)
train_df = train_df.to_numpy()
test_df = test_df.to_numpy()
train_df_feature = train_df[:,:-1]
train_df_label = train_df[:,-1]
test_df_feature = test_df[:,:-1]
test_df_label = test_df[:,-1]

#归一化
train_df_feature = maxminScaler(train_df_feature)
test_df_feature = maxminScaler(test_df_feature)

#构建特征工程
train_feature,train_label = buildFeature (train_df_feature,10,5)

test_feature,test_label = buildFeature (test_df_feature,10,5)

# 创建编码器、注意力GRU和解码器对象
input_shape = train_feature[0].shape
input_reshaped = np.expand_dims(input_shape, axis=-1)
attention_units = 64
model = build_autoencoder(input_reshaped, attention_units)

# 4. 模型训练
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_feature, train_label, batch_size=64, epochs=15, validation_split=0.2)

# 5. 保存模型参数
model.save_weights('model_weights.h5')


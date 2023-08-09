import xgboost as xgb
import matplotlib.pyplot as plt

def selectFeature (data) :
    # 准备数据
    X = data.drop(['timestamp', 'target_variable'], axis=1)
    y = data['target_variable']

    # 训练XGBoost模型
    model = xgb.XGBRegressor()
    model.fit(X, y)

    # 获取特征重要性得分
    feature_importances = model.feature_importances_

    # 绘制特征重要性得分图
    plt.figure(figsize=(10, 6))
    plt.bar(X.columns, feature_importances)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance')
    plt.show()
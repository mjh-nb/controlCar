# 文件名: 2_train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train():
    print("正在读取数据...")
    try:
        df = pd.read_csv('brain_data.csv')
    except FileNotFoundError:
        print("错误：找不到 brain_data.csv，请先运行采集脚本！")
        return

    # 分离特征和标签
    X = df.drop('Label', axis=1) # 前10列
    y = df['Label']              # 最后一列

    # 划分考卷（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === 核心：建立随机森林 ===
    print("正在训练随机森林模型...")
    # n_estimators=100: 用100棵树来投票
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 验证一下准不准
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> 模型准确率: {acc * 100:.2f}%")
    print("\n详细报告:")
    print(classification_report(y_test, y_pred))

    # 保存模型
    joblib.dump(model, 'brain_model.pkl')
    print("模型已保存为 'brain_model.pkl'，可以发给小车用了！")

if __name__ == "__main__":
    train()
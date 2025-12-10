
import joblib
import time
import numpy as np
import requests
import time

from neuropy import NeuroSkyPy


CONFIDENCE_THRESHOLD = 0.4  # 置信度阈值 (60%)
# 只有当超过60%的树都认为是某个动作时，才执行。否则停止。

# 加载模型
print("正在加载大脑模型...")
try:
    model = joblib.load('brain_model.pkl')
except:
    print("错误：没找到模型文件，请先运行训练脚本！")
    exit()


neuropy = NeuroSkyPy("COM7", 57600)
neuropy.start()

def get_real_time_data():
    return [neuropy.attention, neuropy.meditation, neuropy.delta, neuropy.theta, neuropy.lowAlpha,
            neuropy.highAlpha, neuropy.lowBeta,
            neuropy.highBeta, neuropy.lowGamma, neuropy.midGamma]

# 2. 你的模型输出函数（用你的实际模型替换这里）
def get_brain_signal():
    # 1. 获取实时脑波
    # reshape(1, -1) 是因为模型一次习惯预测一批，我们只有一条
    raw_data = get_real_time_data()
    input_data = [raw_data]

    # 2. 让随机森林投票 (使用 predict_proba)
    # 结果类似于: [[0.1, 0.7, 0.05, 0.05, 0.1]]
    probabilities = model.predict_proba(input_data)[0]

    # 3. 找票数最高的动作
    max_index = np.argmax(probabilities)  # 哪个下标概率最大
    max_prob = probabilities[max_index]  # 最大的概率是多少
    predicted_action = model.classes_[max_index]  # 对应的动作名

    # 4. 【核心逻辑】阈值判定
    final_cmd = "stop"

    # 如果本来就是 stop，或者概率太低(不确定)
    if predicted_action == "stop":
        final_cmd = "stop"
        print(f"判断: 停止 (概率: {max_prob:.2f})")

    elif max_prob < CONFIDENCE_THRESHOLD:
        # 比如：虽然觉得像左转，但只有40%把握，为了安全，强制停止
        final_cmd = "stop"
        print(f"判断: 不确定({predicted_action} 只有 {max_prob:.2f}) -> 强制停止")

    else:
        # 信心满满，执行！
        final_cmd = predicted_action
        print(f"判断: !!! {final_cmd} !!! (概率: {max_prob:.2f})")


    if final_cmd=="stop":
        final_cmd="停止"
    elif final_cmd=="back":
        final_cmd="后退"
    elif final_cmd=="forward":
        final_cmd="前进"
    elif final_cmd=="right":
        final_cmd="右转"
    elif final_cmd=="left":
        final_cmd="左转"

    return final_cmd


# 3. 主要发送函数
def send_to_raspberry(signal, speed=50):
    """
    发送信号到树莓派
    signal: 你的模型输出的信号，比如 "前进"
    speed: 速度，默认50
    """

    # 树莓派的IP地址（改成你的树莓派IP）
    PI_IP = "192.168.189.28"
    PI_PORT = 5000

    # 发送的地址
    url = f"http://{PI_IP}:{PI_PORT}/signal"

    # 准备要发送的数据
    data = {
        "signal": signal,
        "speed": speed
    }

    try:
        # 发送POST请求
        response = requests.post(url, json=data)

        # 打印结果
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 发送成功: {signal} {speed}")
            print(f"  树莓派返回: {result}")
        else:
            print(f"✗ 发送失败，错误码: {response.status_code}")

    except Exception as e:
        print(f"✗ 发送出错: {e}")
        print("  请检查：")
        print("  1. 树莓派IP是否正确")
        print("  2. 树莓派服务器是否在运行")
        print("  3. 电脑和树莓派是否在同一网络")


# 4. 主程序
def main():
    print("🚀 开始发送脑电信号...")
    print("💡 按 Ctrl+C 停止")
    print("-" * 40)

    try:
        while True:
            # 1. 从你的模型获取信号
            signal = get_brain_signal()

            # 2. 发送信号到树莓派
            send_to_raspberry(signal)

            # 3. 等待一下再发送下一个信号
            time.sleep(1)  # 每秒发送一次

    except KeyboardInterrupt:
        print("\n👋 停止发送")
    except Exception as e:
        print(f"❌ 程序出错: {e}")


# 5. 运行程序
if __name__ == "__main__":
    # 先安装requests库（如果没有的话）
    try:
        import requests
    except ImportError:
        print("正在安装requests库...")
        import subprocess

        subprocess.check_call(["pip", "install", "requests"])
        print("安装完成，请重新运行程序")
        exit()

    main()
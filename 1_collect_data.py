# 文件名: 1_collect_data.py
import csv
import time
from neuropy import NeuroSkyPy

# 定义动作列表
ACTIONS = ['forward', 'back', 'left', 'right', 'stop']
SAMPLES_PER_ACTION = 200  # 每个动作采200条数据，越多越好

filename = "brain_data.csv"
neuropy = NeuroSkyPy("COM7", 57600)
neuropy.start()

def get_brain_waves():
    return [neuropy.attention, neuropy.meditation, neuropy.delta ,
            neuropy.theta,neuropy.lowAlpha,neuropy.highAlpha,neuropy.lowBeta,
            neuropy.highBeta,neuropy.lowGamma,neuropy.midGamma]


def collect():
    # 1. 创建文件并写入表头
    header = ['Attention', 'Meditation', 'Delta', 'Theta', 'LowAlpha',
              'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MidGamma', 'Label']

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print("=== 开始采集训练数据 ===")

    # 2. 循环采集每个动作
    for action in ACTIONS:
        input(f"\n>>> 准备好采集 [{action}] 了吗？按回车开始...")
        print(f"正在记录 [{action}]... 请保持动作！")

        data_list = []
        for i in range(SAMPLES_PER_ACTION):
            # 获取数据
            features = get_brain_waves()
            # 加上标签
            features.append(action)
            data_list.append(features)

            # 打印进度
            print(f"进度: {i + 1}/{SAMPLES_PER_ACTION}", end='\r')
            time.sleep(0.05)  # 采样间隔 0.05秒

        # 写入文件
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_list)
        print(f"\n[{action}] 采集完成！")

    print(f"\n恭喜！所有数据已保存到 {filename}")


if __name__ == "__main__":
    collect()
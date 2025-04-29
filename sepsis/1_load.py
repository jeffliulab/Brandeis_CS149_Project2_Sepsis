# 文件：sepsis/1_load.py
import os
import pandas as pd

def main():
    # 根目录
    ROOT_DIR = os.getcwd()
    # sepsis 模块目录
    SEPSIS_DIR = os.path.join(ROOT_DIR, 'sepsis')
    # 中间数据存放
    DATA_DIR = os.path.join(SEPSIS_DIR, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)

    ####################################
    ####################################
    # 原始 CSV 路径
    train_csv = os.path.join(SEPSIS_DIR, 'training_data.csv')
    test_csv  = os.path.join(SEPSIS_DIR, 'test_data.csv')

    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)

    # 保存为二进制中间文件
    train.to_pickle(os.path.join(DATA_DIR, 'train_raw.pkl'))
    test .to_pickle(os.path.join(DATA_DIR, 'test_raw.pkl'))
    print(f'✅ 原始数据已加载并保存到 {DATA_DIR}')

if __name__ == '__main__':
    main()

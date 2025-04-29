# 文件：sepsis/2_impute.py
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def main():
    ROOT_DIR   = os.getcwd()
    SEPSIS_DIR = os.path.join(ROOT_DIR, 'sepsis')
    DATA_DIR   = os.path.join(SEPSIS_DIR, 'data')

    train = pd.read_pickle(os.path.join(DATA_DIR, 'train_raw.pkl'))
    test  = pd.read_pickle(os.path.join(DATA_DIR, 'test_raw.pkl'))

    # 将 0 视作缺失
    feat = [c for c in train.columns if c not in ('icustayid','charttime','mortality_90d','gender')]
    train[feat] = train[feat].replace(0, np.nan)
    test [feat] = test [feat].replace(0, np.nan)

    imputer = SimpleImputer(strategy='median')
    train[feat] = imputer.fit_transform(train[feat])
    test [feat] = imputer.transform(test [feat])

    train.to_pickle(os.path.join(DATA_DIR, 'train_imputed.pkl'))
    test .to_pickle(os.path.join(DATA_DIR, 'test_imputed.pkl'))
    print(f'✅ 缺失值已填补并保存到 {DATA_DIR}')

if __name__ == '__main__':
    main()

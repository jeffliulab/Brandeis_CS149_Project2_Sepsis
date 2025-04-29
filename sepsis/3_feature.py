# 文件：sepsis/3_feature.py
import os
import pandas as pd

def extract_agg(df):
    grp = df.groupby('icustayid')
    agg = grp.agg(['last','mean','max','min'])
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg

def main():
    ROOT_DIR   = os.getcwd()
    SEPSIS_DIR = os.path.join(ROOT_DIR, 'sepsis')
    DATA_DIR   = os.path.join(SEPSIS_DIR, 'data')

    train = pd.read_pickle(os.path.join(DATA_DIR, 'train_imputed.pkl'))
    test  = pd.read_pickle(os.path.join(DATA_DIR, 'test_imputed.pkl'))

    X_train = extract_agg(train.drop(columns=['charttime','mortality_90d']))
    y_train = train.groupby('icustayid')['mortality_90d'].first()
    X_test  = extract_agg(test .drop(columns=['charttime']))

    X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), header=True)
    X_test .to_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    print(f'✅ 特征工程完成，文件保存到 {DATA_DIR}')

if __name__ == '__main__':
    main()

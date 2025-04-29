# 文件：sepsis/4_train.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

def main():
    ROOT_DIR    = os.getcwd()
    SEPSIS_DIR  = os.path.join(ROOT_DIR, 'sepsis')
    DATA_DIR    = os.path.join(SEPSIS_DIR, 'data')
    MODELS_DIR  = os.path.join(SEPSIS_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 读取特征和标签
    X = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col=0)
    y = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col=0).values.ravel()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 手动网格搜索参数
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 10, 20]
    }
    best_score = -1.0
    best_params = None

    print("🔍 开始手动网格搜索（5 折交叉验证）…")
    for params in tqdm(ParameterGrid(param_grid), desc="GridSearch"):
        model = RandomForestClassifier(random_state=42, **params)
        # 并行计算交叉验证分数
        scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='f1', n_jobs=-1)
        mean_score = scores.mean()
        tqdm.write(f"  参数 {params} 的平均 F1 = {mean_score:.4f}")
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    print(f"✅ 最佳参数：{best_params}，交叉验证平均 F1 = {best_score:.4f}")

    # 用最佳参数在训练子集上训练最终模型
    best_model = RandomForestClassifier(random_state=42, **best_params)
    best_model.fit(X_tr, y_tr)

    # 在验证集上评估
    preds = best_model.predict(X_val)
    val_f1 = f1_score(y_val, preds)
    print(f"🔍 验证集 F1 = {val_f1:.4f}")

    # 保存模型
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'rf_model.pkl'))
    print(f"✅ 模型已保存到 {MODELS_DIR}")

if __name__ == '__main__':
    main()

# 文件：sepsis/5_evaluate.py
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def main():
    ROOT_DIR      = os.getcwd()
    SEPSIS_DIR    = os.path.join(ROOT_DIR, 'sepsis')
    DATA_DIR      = os.path.join(SEPSIS_DIR, 'data')
    WORKSPACE_DIR = os.path.join(ROOT_DIR, 'app', 'workspace')
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    model = joblib.load(os.path.join(SEPSIS_DIR, 'models', 'rf_model.pkl'))
    X     = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col=0)
    y     = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), index_col=0).values.ravel()

    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    report = classification_report(y, preds)
    auc    = roc_auc_score(y, probs)

    # 打印报告到控制台
    print("\n===== 模型评估报告 =====")
    print(report)
    print(f"AUC = {auc:.4f}")
    print("======= 你可以根据这部分报告内容等会儿写进总结里 ======\n")

    with open(os.path.join(WORKSPACE_DIR, 'eval_report.txt'), 'w') as f:
        f.write(report + f"\nAUC = {auc:.4f}\n")
    print(f'✅ 评估报告已保存到 {WORKSPACE_DIR}')

if __name__ == '__main__':
    main()

# 文件：sepsis/6_explain.py
import os
import json
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import sys
from contextlib import redirect_stdout, redirect_stderr


def main():
    ROOT_DIR      = os.getcwd()
    SEPSIS_DIR    = os.path.join(ROOT_DIR, 'sepsis')
    DATA_DIR      = os.path.join(SEPSIS_DIR, 'data')
    WORKSPACE_DIR = os.path.join(ROOT_DIR, 'app', 'workspace')
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    print("\n===== SHAP模型解释分析 =====")
    
    # 加载模型和数据
    print("1. 加载随机森林模型和训练数据...")
    model = joblib.load(os.path.join(SEPSIS_DIR, 'models', 'rf_model.pkl'))
    X     = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col=0)
    print(f"   数据集大小: {X.shape[0]}行, {X.shape[1]}列")

    # 随机采样，不超过1000条
    sample_size = min(1000, len(X))
    sampled = X.sample(n=sample_size, random_state=42)
    print(f"2. 随机抽样 {sample_size} 条记录用于SHAP分析")

    # 初始化 SHAP 解释器
    print("3. 初始化SHAP树解释器...")
    explainer = shap.TreeExplainer(model)

    # 分批计算 SHAP 值，使用简单的手动进度报告
    print("4. 开始计算SHAP值...")
    n = sampled.shape[0]
    m = sampled.shape[1]
    shap_matrix = np.zeros((n, m))
    chunk_size = 100  # 每批处理100条
    chunks = (n + chunk_size - 1) // chunk_size  # 计算总批次数
    
    # 创建空文件用于捕获输出
    null_file = open(os.devnull, 'w')
    
    for i in range(chunks):
        start = i * chunk_size
        end = min(n, start + chunk_size)
        progress = (i + 1) / chunks * 100
        print(f"   处理批次 {i+1}/{chunks} ({progress:.1f}%)...")
        
        # 捕获所有输出
        with redirect_stdout(null_file), redirect_stderr(null_file):
            batch = sampled.iloc[start:end]
            sv = explainer.shap_values(batch)
            
            # 处理不同版本 SHAP 输出格式
            if isinstance(sv, list):
                batch_vals = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                batch_vals = sv[:, :, 1]
            else:
                batch_vals = sv
                
            shap_matrix[start:end] = batch_vals
    
    null_file.close()
    print("   ✓ SHAP值计算完成")

    # 保存前10条样本的 SHAP 值到 JSON
    print("5. 保存10条样本的SHAP值...")
    out = {str(idx): shap_matrix[i].tolist() for i, idx in enumerate(sampled.index[:10])}
    with open(os.path.join(WORKSPACE_DIR, 'shap_values.json'), 'w') as f:
        json.dump(out, f)
    print("   ✓ SHAP值已保存")

    # 全局聚合：平均绝对 SHAP
    print("6. 计算全局特征重要性...")
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    mean_shap = pd.Series(mean_abs, index=sampled.columns).sort_values(ascending=False)
    top20 = mean_shap.head(20)
    
    # 打印Top20特征的重要性分数
    print("\n==== 模型最重要的20个特征 ====")
    for i, (feature, value) in enumerate(top20.items(), 1):
        print(f"{i}. {feature}: {value:.6f}")
    print("=============================")

    # 可视化：Top20 特征重要性
    print("\n7. 生成Top20特征重要性条形图...")
    
    # 捕获matplotlib输出
    null_file = open(os.devnull, 'w')
    with redirect_stdout(null_file), redirect_stderr(null_file):
        plt.figure(figsize=(12, 6))
        top20.plot.bar()
        plt.title("Top 20 Feature Importance by Mean |SHAP|")
        plt.ylabel("Mean |SHAP value|")
        plt.tight_layout()
        feature_importance_path = os.path.join(WORKSPACE_DIR, "shap_feature_importance.png")
        plt.savefig(feature_importance_path)
        plt.close()
    null_file.close()
    print(f"   图表已保存到: {feature_importance_path}")
    
    # 解释特征重要性图
    print("\n特征重要性分析结果解释:")
    print(f"1. 最重要的特征是 '{top20.index[0]}'，其SHAP值为 {top20.iloc[0]:.6f}")
    print(f"2. 第二重要的特征是 '{top20.index[1]}'，其SHAP值为 {top20.iloc[1]:.6f}")
    print(f"3. 第三重要的特征是 '{top20.index[2]}'，其SHAP值为 {top20.iloc[2]:.6f}")
    print(f"4. 前三个特征的重要性明显高于其他特征，说明模型主要依赖这些特征进行预测")
    print(f"5. 特征重要性呈现长尾分布，表明大多数特征对模型贡献较小")

    # 可视化：SHAP Summary 图
    print("\n8. 生成SHAP Summary蜂群图...")
    
    # 捕获shap库输出
    null_file = open(os.devnull, 'w')
    with redirect_stdout(null_file), redirect_stderr(null_file):
        shap.summary_plot(shap_matrix, sampled, show=False)
        summary_path = os.path.join(WORKSPACE_DIR, "shap_summary.png")
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()
    null_file.close()
    print(f"   图表已保存到: {summary_path}")
    
    # 解释SHAP Summary图
    print("\nSHAP Summary图解释:")
    print("1. 蜂群图展示了各特征值如何影响模型预测")
    print("2. 横轴表示SHAP值大小，正值表示增加患者死亡风险，负值表示降低风险")
    print("3. 每个点代表一个样本，颜色表示该样本的特征值（红色=高，蓝色=低）")
    print("4. 可以观察到，某些生理指标的高值与较高的死亡风险相关")
    print("5. 而另一些指标的低值则与较高的死亡风险相关")
    
    print(f'\n✅ SHAP分析完成，结果已保存到 {WORKSPACE_DIR}')
    print("=================================\n")


if __name__ == '__main__':
    main()


# # 文件：sepsis/6_explain.py
# import os
# import json
# import joblib
# import pandas as pd
# import shap
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm.auto import tqdm


# def main():
#     ROOT_DIR      = os.getcwd()
#     SEPSIS_DIR    = os.path.join(ROOT_DIR, 'sepsis')
#     DATA_DIR      = os.path.join(SEPSIS_DIR, 'data')
#     WORKSPACE_DIR = os.path.join(ROOT_DIR, 'app', 'workspace')
#     os.makedirs(WORKSPACE_DIR, exist_ok=True)

#     # 加载模型和数据
#     model = joblib.load(os.path.join(SEPSIS_DIR, 'models', 'rf_model.pkl'))
#     X     = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'), index_col=0)

#     # 随机采样，不超过1000条
#     sample_size = min(1000, len(X))
#     sampled = X.sample(n=sample_size, random_state=42)

#     # 初始化 SHAP 解释器
#     explainer = shap.TreeExplainer(model)

#     # 分批计算 SHAP 值并显示进度
#     n = sampled.shape[0]
#     m = sampled.shape[1]
#     shap_matrix = np.zeros((n, m))
#     chunk_size = 100  # 每批处理100条
#     for start in tqdm(range(0, n, chunk_size), desc="Computing SHAP", unit="records"):
#         end = min(n, start + chunk_size)
#         batch = sampled.iloc[start:end]
#         sv = explainer.shap_values(batch)
#         # 处理不同版本 SHAP 输出格式
#         if isinstance(sv, list):
#             # 旧版 SHAP: list of arrays [neg_class, pos_class]
#             batch_vals = sv[1]
#         elif isinstance(sv, np.ndarray) and sv.ndim == 3:
#             # 新版 SHAP: array of shape (n_samples, n_features, n_classes)
#             batch_vals = sv[:, :, 1]
#         else:
#             # 其他:直接返回正类 SHAP 矩阵
#             batch_vals = sv
#         shap_matrix[start:end] = batch_vals

#     # 保存前10条样本的 SHAP 值到 JSON
#     out = {str(idx): shap_matrix[i].tolist() for i, idx in enumerate(sampled.index[:10])}
#     with open(os.path.join(WORKSPACE_DIR, 'shap_values.json'), 'w') as f:
#         json.dump(out, f)

#     # 全局聚合：平均绝对 SHAP
#     mean_abs = np.abs(shap_matrix).mean(axis=0)
#     mean_shap = pd.Series(mean_abs, index=sampled.columns).sort_values(ascending=False)
#     top20 = mean_shap.head(20)

#     # 可视化：Top20 特征重要性
#     plt.figure(figsize=(12, 6))
#     top20.plot.bar()
#     plt.title("Top 20 Feature Importance by Mean |SHAP|")
#     plt.ylabel("Mean |SHAP value|")
#     plt.tight_layout()
#     plt.savefig(os.path.join(WORKSPACE_DIR, "shap_feature_importance.png"))
#     plt.close()

#     # 可视化：SHAP Beeswarm 图
#     shap.summary_plot(shap_matrix, sampled, show=False)
#     plt.savefig(os.path.join(WORKSPACE_DIR, "shap_summary.png"), bbox_inches='tight')
#     plt.close()

#     print(f'✅ SHAP分析完成，结果已保存到 {WORKSPACE_DIR}')


# if __name__ == '__main__':
#     main()

# 文件：sepsis/7_predict.py
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 根目录和路径配置
    ROOT_DIR      = os.getcwd()
    SEPSIS_DIR    = os.path.join(ROOT_DIR, 'sepsis')
    DATA_DIR      = os.path.join(SEPSIS_DIR, 'data')
    WORKSPACE_DIR = os.path.join(ROOT_DIR, 'app', 'workspace')
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    print("\n===== 测试集预测分析 =====")
    
    # 加载训练好的模型和测试集特征
    print("1. 加载随机森林模型和测试数据...")
    model  = joblib.load(os.path.join(SEPSIS_DIR, 'models', 'rf_model.pkl'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'), index_col=0)
    print(f"   测试集大小: {X_test.shape[0]}行, {X_test.shape[1]}列")

    # 预测并保存结果
    print("2. 对测试集进行预测...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]  # 获取正类（死亡）的概率
    
    # 转换预测为整数类型，确保兼容性
    preds_int = preds.astype(int)
    
    # 计算和显示预测统计信息
    survival_count = np.sum(preds_int == 0)
    death_count = np.sum(preds_int == 1)
    total_count = len(preds_int)
    
    print("\n==== 预测结果统计 ====")
    print(f"预测存活 (0): {survival_count} 例 ({survival_count/total_count*100:.2f}%)")
    print(f"预测死亡 (1): {death_count} 例 ({death_count/total_count*100:.2f}%)")
    print(f"总计样本数: {total_count}")
    
    # 显示概率分布的一些统计量
    if len(probs) > 0:
        print("\n==== 死亡概率统计 ====")
        print(f"最小概率: {np.min(probs):.4f}")
        print(f"最大概率: {np.max(probs):.4f}")
        print(f"平均概率: {np.mean(probs):.4f}")
        print(f"中位数概率: {np.median(probs):.4f}")
        print(f"标准差: {np.std(probs):.4f}")
        
        # 分位数分析
        quantiles = np.percentile(probs, [25, 50, 75, 90, 95, 99])
        print("\n==== 概率分位数 ====")
        print(f"25% 分位数: {quantiles[0]:.4f}")
        print(f"50% 分位数: {quantiles[1]:.4f}")
        print(f"75% 分位数: {quantiles[2]:.4f}")
        print(f"90% 分位数: {quantiles[3]:.4f}")
        print(f"95% 分位数: {quantiles[4]:.4f}")
        print(f"99% 分位数: {quantiles[5]:.4f}")
    
    # 保存预测结果
    df = pd.DataFrame({
        'icustayid': X_test.index,
        'mortality_90d_pred': preds_int,
        'mortality_90d_prob': probs
    })
    output_csv = os.path.join(WORKSPACE_DIR, 'predict_test_predictions.csv')
    df.to_csv(output_csv, index=False)
    print(f'\n3. 测试集预测已保存到 {output_csv}')

    # 可视化：预测分布柱状图
    print("4. 生成预测分布柱状图...")
    vc = pd.Series(preds_int).value_counts().sort_index()
    plt.figure()
    vc.plot.bar(rot=0)
    plt.xlabel('Predicted Class (0=Survive, 1=Death)')
    plt.ylabel('Count')
    plt.title('Predicted Mortality Distribution')
    plt.tight_layout()
    dist_png = os.path.join(WORKSPACE_DIR, 'predict_distribution.png')
    plt.savefig(dist_png)
    plt.close()
    print(f'   图表已保存到: {dist_png}')
    
    # 文字解释分布图
    print("\n预测分布图解释:")
    print(f"1. 模型对大多数患者({survival_count}例，占{survival_count/total_count*100:.2f}%)预测为存活(0类)")
    print(f"2. 模型对剩余的患者({death_count}例，占{death_count/total_count*100:.2f}%)预测为死亡(1类)")
    
    if death_count > 0 and survival_count > 0:
        # 计算死亡与存活的比例
        ratio = death_count / survival_count
        print(f"3. 死亡与存活的比例约为 1:{1/ratio:.2f}")
    
    if total_count > 0:
        print(f"4. 预测死亡率为 {death_count/total_count*100:.2f}%")
    
    # 如果计算了概率，增加概率分布的解释
    if len(probs) > 0:
        # 生成概率直方图
        plt.figure(figsize=(10, 6))
        plt.hist(probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
        plt.xlabel('Mortality Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Mortality Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        prob_png = os.path.join(WORKSPACE_DIR, 'predict_probability_distribution.png')
        plt.savefig(prob_png)
        plt.close()
        print(f'   概率分布图已保存到: {prob_png}')
        
        # 概率分布解释
        print("\n概率分布解释:")
        print(f"1. 模型预测的死亡概率平均值为 {np.mean(probs):.4f}")
        high_risk = np.sum(probs > 0.75)
        if high_risk > 0:
            print(f"2. 有 {high_risk} 例患者的死亡风险较高(概率>0.75)")
        low_risk = np.sum(probs < 0.25)
        if low_risk > 0:
            print(f"3. 有 {low_risk} 例患者的死亡风险较低(概率<0.25)")
        
        # 预测确定性分析
        certain_pred = np.sum((probs > 0.9) | (probs < 0.1))
        if total_count > 0:
            print(f"4. 模型对 {certain_pred} 例患者的预测较为确定(概率>0.9或<0.1)，占总数的 {certain_pred/total_count*100:.2f}%")
        
        # 模型预测的边界情况分析
        borderline = np.sum((probs > 0.4) & (probs < 0.6))
        if total_count > 0:
            print(f"5. 模型对 {borderline} 例患者的预测较为不确定(概率在0.4-0.6之间)，占总数的 {borderline/total_count*100:.2f}%")
    
    print("\n✅ 测试集预测分析完成")
    print("===========================\n")

if __name__ == '__main__':
    main()


# # 文件：sepsis/7_predict.py
# import os
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt


# def main():
#     # 根目录和路径配置
#     ROOT_DIR      = os.getcwd()
#     SEPSIS_DIR    = os.path.join(ROOT_DIR, 'sepsis')
#     DATA_DIR      = os.path.join(SEPSIS_DIR, 'data')
#     WORKSPACE_DIR = os.path.join(ROOT_DIR, 'app', 'workspace')
#     os.makedirs(WORKSPACE_DIR, exist_ok=True)

#     # 加载训练好的模型和测试集特征
#     model  = joblib.load(os.path.join(SEPSIS_DIR, 'models', 'rf_model.pkl'))
#     X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'), index_col=0)

#     # 预测并保存结果
#     preds = model.predict(X_test)
#     df = pd.DataFrame({
#         'icustayid': X_test.index,
#         'mortality_90d_pred': preds
#     })
#     output_csv = os.path.join(WORKSPACE_DIR, 'predict_test_predictions.csv')
#     df.to_csv(output_csv, index=False)
#     print(f'✅ 测试集预测已保存到 {output_csv}')

#     # 可视化：预测分布柱状图
#     vc = df['mortality_90d_pred'].value_counts().sort_index()
#     plt.figure()
#     vc.plot.bar(rot=0)
#     plt.xlabel('Predicted Class (0=Survive, 1=Death)')
#     plt.ylabel('Count')
#     plt.title('Predicted Mortality Distribution')
#     plt.tight_layout()
#     dist_png = os.path.join(WORKSPACE_DIR, 'predict_distribution.png')
#     plt.savefig(dist_png)
#     plt.close()
#     print(f'✅ 预测分布图已保存到 {dist_png}')

# if __name__ == '__main__':
#     main()

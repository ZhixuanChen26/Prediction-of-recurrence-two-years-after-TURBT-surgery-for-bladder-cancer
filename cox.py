import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings

np.random.seed(42)
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Microsoft YaHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

def prepare_data(data):
    """准备Cox回归数据"""
    exclude_cols = ['病案号']
    
    # 处理随访时间和事件变量
    if '复发间隔日' in data.columns:
        data['survival_time'] = data['复发间隔日'].copy()
        data['event'] = data['两年内是否复发'].copy()
        
        # 复发时间>742天的设为删失
        mask_late = (data['复发间隔日'] > 742) & (data['两年内是否复发'] == 1)
        data.loc[mask_late, 'survival_time'] = 742
        data.loc[mask_late, 'event'] = 0
        
        # 无复发的随访时间设为742天
        data.loc[data['两年内是否复发'] == 0, 'survival_time'] = 742
        
        exclude_cols.extend(['复发间隔日', '两年内是否复发'])
    else:
        data['survival_time'] = 742
        data['event'] = data['两年内是否复发']
        exclude_cols.append('两年内是否复发')
    
    # 定义参考类别字典
    reference_categories = {
        'bmi值': '1.0',     # 正常体重
        '部位值': '3.0',    # 后侧壁
        't值': '0.0',      # T1分期  
        '形状值': '1.0'     # 菜花状
    }
    
    # 对需要one-hot编码的变量进行编码
    onehot_vars = ['bmi值', '形状值', 't值', '部位值']
    
    for var in onehot_vars:
        if var in data.columns:
            # 确保数据类型正确
            data[var] = data[var].astype(str)
            
            # 进行one-hot编码
            var_dummies = pd.get_dummies(data[var], prefix=var, dtype=int)
            
            # 删除指定的参考类别
            ref_value = reference_categories[var]
            ref_col = f"{var}_{ref_value}"
            
            if ref_col in var_dummies.columns:
                var_dummies = var_dummies.drop(ref_col, axis=1)
            
            # 删除原列，添加编码后的列
            data = data.drop(columns=[var])
            data = pd.concat([data, var_dummies], axis=1)
            exclude_cols.append(var)  # 确保原变量被排除
    
    # 获取特征列
    feature_cols = [col for col in data.columns if col not in exclude_cols + ['survival_time', 'event']]
    
    print(f"数据: {data.shape}, 事件: {data['event'].sum()}例, 特征: {len(feature_cols)}个")
    return data, feature_cols

def univariate_cox(data, feature_cols):
    """单因素Cox回归"""
    results = []
    
    for feature in feature_cols:
        try:
            temp_data = data[[feature, 'survival_time', 'event']].copy()
            if temp_data[feature].nunique() < 2:
                continue
                
            cph = CoxPHFitter()
            cph.fit(temp_data, duration_col='survival_time', event_col='event')
            
            results.append({
                'Variable': feature,
                'HR': cph.summary.loc[feature, 'exp(coef)'],
                'P_value': cph.summary.loc[feature, 'p'],
                'CI_lower': cph.summary.loc[feature, 'exp(coef) lower 95%'],
                'CI_upper': cph.summary.loc[feature, 'exp(coef) upper 95%']
            })
        except:
            continue
    
    results_df = pd.DataFrame(results).sort_values('P_value')
    print(f"\n单因素分析: {len(results_df)}个变量")
    print(results_df.head(15).round(4))
    
    return results_df

def select_optimal_penalizer(data, selected_vars):
    """通过5折CV选择最优正则化参数"""
    model_features = ['survival_time', 'event'] + selected_vars
    cv_data = data[model_features].copy()
    
    penalizers = np.logspace(-3, 0, 10)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_score = 0
    best_pen = 0.01
    
    for pen in penalizers:
        fold_scores = []
        try:
            for train_idx, test_idx in kf.split(cv_data):
                train_data = cv_data.iloc[train_idx]
                test_data = cv_data.iloc[test_idx]
                
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(train_data, duration_col='survival_time', event_col='event')
                c_index = cph.score(test_data, scoring_method='concordance_index')
                fold_scores.append(c_index)
            
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_pen = pen
        except:
            continue
    
    print(f"最优penalizer: {best_pen:.3f} (CV C-index: {best_score:.4f})")
    return best_pen

def multivariate_cox_with_assumptions(data, selected_vars):
    """多因素Cox回归 + PH假设检验"""
    model_features = ['survival_time', 'event'] + selected_vars
    model_data = data[model_features].copy()
    
    # 选择最优正则化参数
    optimal_pen = select_optimal_penalizer(data, selected_vars)
    
    # 拟合最终模型
    cph = CoxPHFitter(penalizer=optimal_pen)
    cph.fit(model_data, duration_col='survival_time', event_col='event')
    
    print(f"\n多因素分析结果:")
    summary_display = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(4)
    print(summary_display)
    print(f"C-index: {cph.concordance_index_:.4f}")
    
    # PH假设检验
    print("\n比例风险假设检验:")
    cph.check_assumptions(model_data, p_value_threshold=0.05, show_plots=False)
    
    # 显示最终显著变量
    final_significant = cph.summary[cph.summary['p'] < 0.05]
    if len(final_significant) > 0:
        print(f"\n最终显著变量 (p < 0.05):")
        for var in final_significant.index:
            hr = final_significant.loc[var, 'exp(coef)']
            p_val = final_significant.loc[var, 'p']
            ci_lower = final_significant.loc[var, 'exp(coef) lower 95%']
            ci_upper = final_significant.loc[var, 'exp(coef) upper 95%']
            direction = "增加风险" if cph.summary.loc[var, 'coef'] > 0 else "降低风险"
            print(f"  {var}: HR={hr:.3f} (95%CI: {ci_lower:.3f}-{ci_upper:.3f}), p={p_val:.4f} ({direction})")
    
    return cph, model_data

def plot_results(model, univar_df):
    """绘制结果并保存文件"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 回归系数
    coefs = model.summary['coef'].sort_values()
    colors = ['red' if x < 0 else 'blue' for x in coefs.values]
    ax1.barh(range(len(coefs)), coefs.values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(coefs)))
    ax1.set_yticklabels(coefs.index, fontsize=10)
    ax1.set_title('回归系数')
    ax1.axvline(x=0, color='black', linestyle='--')
    ax1.grid(True, alpha=0.3)
    
    # 风险比
    hrs = model.summary['exp(coef)'].sort_values()
    colors = ['green' if x < 1 else 'red' for x in hrs.values]
    ax2.barh(range(len(hrs)), hrs.values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(hrs)))
    ax2.set_yticklabels(hrs.index, fontsize=10)
    ax2.set_title('风险比 (HR)')
    ax2.axvline(x=1, color='black', linestyle='--')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    univar_df.to_csv('Cox单因素结果.csv', index=False, encoding='utf-8-sig')
    model.summary.to_csv('Cox多因素结果.csv', encoding='utf-8-sig')
    
    print("\n结果已保存:")
    print("- Cox单因素结果.csv")
    print("- Cox多因素结果.csv")
    print(f"- C-index: {model.concordance_index_:.4f}")

def main():
    """主函数"""
    # 加载数据
    data = pd.read_csv('454人插补后总表_两年_复发.csv', encoding='utf-8')
    
    # 数据准备（包含one-hot编码）
    data, feature_cols = prepare_data(data)
    
    # 单因素分析
    univar_results = univariate_cox(data, feature_cols)
    
    # 选择p<0.1的特征
    selected_vars = univar_results[univar_results['P_value'] < 0.1]['Variable'].tolist()
    
    if len(selected_vars) == 0:
        print("无p<0.1的特征")
        return
    
    print(f"\n选择{len(selected_vars)}个特征进入多因素分析:")
    for var in selected_vars:
        row = univar_results[univar_results['Variable'] == var].iloc[0]
        print(f"  {var}: HR={row['HR']:.3f}, p={row['P_value']:.4f}")
    
    # 多因素分析
    multi_model, _ = multivariate_cox_with_assumptions(data, selected_vars)
    
    # 绘图并保存结果
    plot_results(multi_model, univar_results)

if __name__ == "__main__":
    main()
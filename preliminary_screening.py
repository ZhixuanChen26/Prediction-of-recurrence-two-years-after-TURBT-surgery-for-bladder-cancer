import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def enhanced_correlation_filter(data, outcome_col, exclude_cols=None, 
                              corr_threshold=0.7, 
                              p_threshold=0.2,
                              auc_threshold=0.4,
                              corr_method='spearman', 
                              variance_threshold=0.01,
                              bootstrap_n=500,
                              bootstrap_threshold=0.8):
    """
    增强版相关性筛选：单变量预筛选 + 迭代相关性筛选 + Bootstrap验证
    """
    
    if exclude_cols is None:
        exclude_cols = []
    
    print(f"=== 增强版特征筛选开始 ===")
    print(f"参数: corr_threshold={corr_threshold}, p_threshold={p_threshold}")
    print(f"     auc_threshold={auc_threshold}, bootstrap_n={bootstrap_n}")
    
    # 记录所有删除的变量
    all_removed = []
    
    # 获取特征列
    all_feature_cols = [col for col in data.columns 
                       if col != outcome_col and col not in exclude_cols]
    
    print(f"初始特征数: {len(all_feature_cols)}")
    
    # 第一步：处理one-hot同源变量组
    print("\n第一步：处理one-hot同源变量...")
    onehot_groups = identify_onehot_groups(all_feature_cols)
    processed_features, onehot_removed = process_onehot_groups(
        data, onehot_groups, outcome_col, all_removed
    )
    processed_features = [col for col in processed_features if col not in exclude_cols]
    print(f"One-hot处理后: {len(processed_features)} 个特征")
    
    # 第二步：方差筛选
    print(f"\n第二步：方差筛选...")
    numeric_data = data[processed_features].select_dtypes(include=[np.number])
    low_variance_features = []
    
    if len(numeric_data.columns) > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(numeric_data.fillna(0))
        
        low_variance_features = [col for col, keep in zip(numeric_data.columns, selector.get_support()) if not keep]
        high_variance_features = [col for col, keep in zip(numeric_data.columns, selector.get_support()) if keep]
        
        # 记录删除的变量
        for feature in low_variance_features:
            all_removed.append({
                'step': '方差筛选',
                'feature': feature,
                'reason': '低方差',
                'detail': f'方差<{variance_threshold}'
            })
        
        non_numeric_features = [col for col in processed_features 
                               if col not in numeric_data.columns and col not in exclude_cols]
        processed_features = high_variance_features + non_numeric_features
        
        print(f"移除低方差特征 {len(low_variance_features)} 个，剩余: {len(processed_features)}")
    
    # 第三步：单变量预筛选
    print(f"\n第三步：单变量预筛选...")
    univariate_selected = univariate_prefilter(
        data, processed_features, outcome_col, p_threshold, auc_threshold, all_removed
    )
    print(f"单变量筛选后: {len(univariate_selected)} 个特征")
    
    # 第四步：迭代相关性筛选
    print(f"\n第四步：迭代相关性筛选...")
    correlation_selected = iterative_correlation_filter(
        data, univariate_selected, outcome_col, corr_threshold, corr_method, all_removed
    )
    print(f"相关性筛选后: {len(correlation_selected)} 个特征")
    
    # 第五步：Bootstrap稳定性验证
    print(f"\n第五步：Bootstrap稳定性验证 ({bootstrap_n}次)...")
    bootstrap_selected, feature_frequencies = bootstrap_stability_filter(
        data, correlation_selected, outcome_col, bootstrap_n, bootstrap_threshold, all_removed
    )
    print(f"Bootstrap验证后: {len(bootstrap_selected)} 个特征")
    
    # 保存所有结果到一个文件
    save_all_results(all_feature_cols, all_removed, bootstrap_selected, feature_frequencies,
                     {
                         'corr_threshold': corr_threshold,
                         'p_threshold': p_threshold,
                         'auc_threshold': auc_threshold,
                         'bootstrap_n': bootstrap_n,
                         'bootstrap_threshold': bootstrap_threshold
                     })
    
    return bootstrap_selected, feature_frequencies


def identify_onehot_groups(feature_cols):
    """识别one-hot编码的同源变量组"""
    onehot_groups = defaultdict(list)
    onehot_patterns = ['bmi值_', '部位值_']
    
    for col in feature_cols:
        for pattern in onehot_patterns:
            if col.startswith(pattern):
                base_name = pattern.rstrip('_')
                onehot_groups[base_name].append(col)
                break
    
    return {k: v for k, v in onehot_groups.items() if len(v) > 1}


def process_onehot_groups(data, onehot_groups, outcome_col, all_removed):
    """处理one-hot同源变量组"""
    processed_features = []
    
    all_onehot_cols = set()
    for group_cols in onehot_groups.values():
        all_onehot_cols.update(group_cols)
    
    # 添加非one-hot特征
    for col in data.columns:
        if col != outcome_col and col not in all_onehot_cols:
            processed_features.append(col)
    
    # 处理每个one-hot组
    for group_name, group_cols in onehot_groups.items():
        # 计算每列的AUC
        scores = {}
        for col in group_cols:
            try:
                mask = ~(data[col].isnull() | data[outcome_col].isnull())
                if mask.sum() > 0:
                    auc = roc_auc_score(data.loc[mask, outcome_col], data.loc[mask, col])
                    scores[col] = max(auc, 1-auc) - 0.5
                else:
                    scores[col] = 0
            except:
                scores[col] = 0
        
        # 选择最佳列
        best_col = max(scores.keys(), key=lambda x: scores[x])
        processed_features.append(best_col)
        
        # 记录删除的变量
        for col in group_cols:
            if col != best_col:
                all_removed.append({
                    'step': 'One-hot处理',
                    'feature': col,
                    'reason': f'同组保留{best_col}',
                    'detail': f'AUC得分: {scores[col]:.3f} < {scores[best_col]:.3f}'
                })
        
        print(f"  {group_name}: 保留{best_col}, 移除{len(group_cols)-1}个")
    
    return processed_features, []


def univariate_prefilter(data, features, outcome_col, p_threshold, auc_threshold, all_removed):
    """单变量预筛选"""
    selected_features = []
    
    for feature in features:
        try:
            mask = ~(data[feature].isnull() | data[outcome_col].isnull())
            if mask.sum() < 10:
                all_removed.append({
                    'step': '单变量筛选',
                    'feature': feature,
                    'reason': '数据不足',
                    'detail': f'有效样本<10个'
                })
                continue
            
            X = data.loc[mask, feature]
            y = data.loc[mask, outcome_col]
            
            # 计算p值
            X_with_const = sm.add_constant(X)
            model = sm.Logit(y, X_with_const).fit(disp=0)
            p_value = model.pvalues[1]
            
            # 计算AUC
            if X.nunique() > 20:
                X_binned = pd.qcut(X, q=min(5, X.nunique()), duplicates='drop').cat.codes
                auc = roc_auc_score(y, X_binned)
            else:
                auc = roc_auc_score(y, X)
            
            # 判断是否保留
            if p_value < p_threshold and auc > auc_threshold:
                selected_features.append(feature)
            else:
                reason = []
                if p_value >= p_threshold:
                    reason.append(f'p={p_value:.3f}>={p_threshold}')
                if auc <= auc_threshold:
                    reason.append(f'auc={auc:.3f}<={auc_threshold}')
                
                all_removed.append({
                    'step': '单变量筛选',
                    'feature': feature,
                    'reason': '预测能力不足',
                    'detail': ', '.join(reason)
                })
                
        except Exception as e:
            all_removed.append({
                'step': '单变量筛选',
                'feature': feature,
                'reason': '计算错误',
                'detail': str(e)
            })
    
    print(f"  保留: {len(selected_features)}, 移除: {len(features) - len(selected_features)}")
    return selected_features


def iterative_correlation_filter(data, features, outcome_col, threshold, method, all_removed):
    """迭代相关性筛选"""
    current_features = features.copy()
    iteration = 0
    
    while True:
        iteration += 1
        
        # 计算相关矩阵
        numeric_data = data[current_features].select_dtypes(include=[np.number])
        if len(numeric_data.columns) < 2:
            break
            
        if method == 'spearman':
            corr_matrix = numeric_data.corr(method='spearman').abs()
        else:
            corr_matrix = numeric_data.corr(method='pearson').abs()
        
        # 找到最高相关对
        high_corr_pair = None
        max_corr = 0
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > threshold and corr_val > max_corr:
                    max_corr = corr_val
                    high_corr_pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
        
        if high_corr_pair is None:
            break
        
        # 选择删除AUC较低的特征
        feature1, feature2, corr_val = high_corr_pair
        auc1 = calculate_single_auc(data, feature1, outcome_col)
        auc2 = calculate_single_auc(data, feature2, outcome_col)
        
        if auc1 >= auc2:
            removed_feature = feature2
            kept_feature = feature1
        else:
            removed_feature = feature1
            kept_feature = feature2
        
        current_features.remove(removed_feature)
        all_removed.append({
            'step': '相关性筛选',
            'feature': removed_feature,
            'reason': f'与{kept_feature}高相关',
            'detail': f'相关系数={corr_val:.3f}, 第{iteration}轮'
        })
    
    print(f"  完成{iteration}轮迭代，移除{len(features) - len(current_features)}个特征")
    return current_features


def calculate_single_auc(data, feature, outcome_col):
    """计算单个特征的AUC"""
    try:
        mask = ~(data[feature].isnull() | data[outcome_col].isnull())
        if mask.sum() < 10:
            return 0
        
        X = data.loc[mask, feature]
        y = data.loc[mask, outcome_col]
        
        if X.nunique() > 20:
            X_binned = pd.qcut(X, q=min(5, X.nunique()), duplicates='drop').cat.codes
            auc = roc_auc_score(y, X_binned)
        else:
            auc = roc_auc_score(y, X)
        
        return max(auc, 1-auc) - 0.5
    except:
        return 0


def bootstrap_stability_filter(data, features, outcome_col, n_bootstrap, threshold, all_removed):
    """Bootstrap稳定性验证"""
    feature_counts = defaultdict(int)
    
    print(f"  执行{n_bootstrap}次Bootstrap...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 50 == 0:
            print(f"    完成{i + 1}/{n_bootstrap}次")
        
        # Bootstrap采样
        n_samples = len(data)
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_data = data.iloc[bootstrap_indices].reset_index(drop=True)
        
        try:
            X = bootstrap_data[features].fillna(0)
            y = bootstrap_data[outcome_col]
            
            # 标准化特征
            X_scaled = (X - X.mean()) / (X.std() + 1e-8)
            
            # L1正则化逻辑回归
            model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
            model.fit(X_scaled, y)
            
            # 记录被选中的特征
            selected_indices = np.where(model.coef_[0] != 0)[0]
            for idx in selected_indices:
                feature_counts[features[idx]] += 1
                
        except:
            continue
    
    # 计算特征频率
    feature_frequencies = {}
    for feature in features:
        feature_frequencies[feature] = feature_counts[feature] / n_bootstrap
    
    # 选择频率>=阈值的特征
    selected_features = [feature for feature, freq in feature_frequencies.items() 
                        if freq >= threshold]
    
    # 记录被删除的特征
    for feature in features:
        if feature not in selected_features:
            freq = feature_frequencies[feature]
            all_removed.append({
                'step': 'Bootstrap验证',
                'feature': feature,
                'reason': 'Bootstrap频率不足',
                'detail': f'频率={freq:.1%}<{threshold:.1%}'
            })
    
    return selected_features, feature_frequencies


def save_all_results(original_features, all_removed, final_features, feature_frequencies, parameters):
    """保存所有结果到一个Excel文件"""
    
    with pd.ExcelWriter('特征筛选完整结果.xlsx', engine='openpyxl') as writer:
        
        # 1. 参数设置和汇总
        summary_data = [
            ['参数设置', ''],
            ['相关性阈值', parameters['corr_threshold']],
            ['p值阈值', parameters['p_threshold']],
            ['AUC阈值', parameters['auc_threshold']],
            ['Bootstrap次数', parameters['bootstrap_n']],
            ['Bootstrap频率阈值', parameters['bootstrap_threshold']],
            ['', ''],
            ['筛选结果汇总', ''],
            ['原始特征数', len(original_features)],
            ['最终保留特征数', len(final_features)],
            ['删除特征数', len(all_removed)],
            ['删除比例', f"{len(all_removed)/len(original_features):.1%}"]
        ]
        
        summary_df = pd.DataFrame(summary_data, columns=['项目', '值'])
        summary_df.to_excel(writer, sheet_name='汇总', index=False)
        
        # 2. 删除变量详情
        removed_df = pd.DataFrame(all_removed)
        if not removed_df.empty:
            removed_df = removed_df[['step', 'feature', 'reason', 'detail']]
            removed_df.columns = ['筛选步骤', '特征名', '删除原因', '详细信息']
        removed_df.to_excel(writer, sheet_name='删除变量详情', index=False)
        
        # 3. 最终保留特征
        final_data = []
        for i, feature in enumerate(final_features, 1):
            freq = feature_frequencies.get(feature, 0)
            final_data.append([i, feature, f"{freq:.1%}"])
        
        final_df = pd.DataFrame(final_data, columns=['序号', '特征名', 'Bootstrap频率'])
        final_df.to_excel(writer, sheet_name='最终特征', index=False)
        
        # 4. 各步骤统计
        step_stats = []
        step_counts = {}
        for item in all_removed:
            step = item['step']
            step_counts[step] = step_counts.get(step, 0) + 1
        
        current_count = len(original_features)
        step_stats.append(['原始特征', current_count, 0, ''])
        
        for step in ['One-hot处理', '方差筛选', '单变量筛选', '相关性筛选', 'Bootstrap验证']:
            removed_count = step_counts.get(step, 0)
            current_count -= removed_count
            step_stats.append([step, current_count, removed_count, f"-{removed_count}"])
        
        steps_df = pd.DataFrame(step_stats, columns=['步骤', '剩余特征数', '删除数量', '变化'])
        steps_df.to_excel(writer, sheet_name='各步骤统计', index=False)
    
    print(f"\n=== 所有结果已保存到: 特征筛选完整结果.xlsx ===")
    print(f"包含4个工作表: 汇总、删除变量详情、最终特征、各步骤统计")


# 使用示例
if __name__ == "__main__":
    data = pd.read_csv('454人插补后总表_两年.csv')
    
    outcome = "两年内是否复发"
    exclude = ["病案号", "一年内是否复发"]
    
    selected_features, feature_frequencies = enhanced_correlation_filter(
        data=data,
        outcome_col=outcome,
        exclude_cols=exclude,
        corr_threshold=0.7,
        p_threshold=0.2,
        auc_threshold=0.4,
        corr_method='spearman',
        variance_threshold=0.01,
        bootstrap_n=500,       
        bootstrap_threshold=0.8
    )
    
    print(f"\n=== 最终筛选出 {len(selected_features)} 个特征 ===")
    sorted_features = sorted([(f, feature_frequencies.get(f, 0)) for f in selected_features],
                           key=lambda x: x[1], reverse=True)
    
    for i, (feature, freq) in enumerate(sorted_features, 1):
        print(f"{i:2d}. {feature:40s} ({freq:.1%})")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


def improved_elasticnet_selection(data, outcome_col, force_include=None,
                                 target_features=15, cv_folds=10, 
                                 random_state=42, exclude_cols=None):
    """
    改进的ElasticNet特征筛选，支持强制保留特定特征
    
    Parameters:
    -----------
    data : DataFrame
        输入数据
    outcome_col : str
        结局变量列名
    force_include : list
        强制保留的特征列表
    target_features : int
        目标保留的特征数量
    """
    
    if exclude_cols is None:
        exclude_cols = []
    if force_include is None:
        force_include = []

    print(f"=== 改进的ElasticNet特征筛选 ===")
    print(f"目标特征数: {target_features}")
    print(f"强制保留特征: {force_include}")

    # ───────────────── 数据预处理 ─────────────────
    data_processed = data.copy()
    
    # 定义参考类别字典和需要编码的变量
    reference_categories = {
        'bmi值': '1.0',
        '部位值': '3.0', 
    }
    
    categorical_vars_to_encode = ['bmi值', '部位值']
    
    # One-hot编码
    for var in categorical_vars_to_encode:
        if var in data_processed.columns:
            data_processed[var] = data_processed[var].astype(str)
            var_dummies = pd.get_dummies(data_processed[var], prefix=var, dtype=int)
            
            ref_value = reference_categories[var]
            ref_col = f"{var}_{ref_value}"
            
            if ref_col in var_dummies.columns:
                var_dummies = var_dummies.drop(ref_col, axis=1)
            
            data_processed = data_processed.drop(columns=[var])
            data_processed = pd.concat([data_processed, var_dummies], axis=1)

    # ───────────────── 准备数据 ─────────────────
    feature_cols = [col for col in data_processed.columns
                   if col != outcome_col and col not in exclude_cols]
    
    X = data_processed[feature_cols].copy()
    y = data_processed[outcome_col].copy()

    # 清理数据
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    print(f"\n数据概况:")
    print(f"  样本量: {len(X_clean)}")
    print(f"  候选特征数: {len(feature_cols)}")
    print(f"  事件数: {sum(y_clean == 1)} ({sum(y_clean == 1) / len(y_clean) * 100:.1f}%)")

    # ───────────────── 标准化 ─────────────────
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )

    # ───────────────── 多种ElasticNet参数组合 ─────────────────
    # 更宽松的正则化参数
    alpha_values = np.logspace(-4, 0, 20)  # 从0.0001到1.0
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # 不同的L1/L2混合比例
    
    print(f"\n尝试不同的ElasticNet参数组合...")
    
    best_auc = 0
    best_params = {}
    best_selected_features = []
    results_summary = []
    
    for l1_ratio in l1_ratios:
        print(f"  测试 l1_ratio = {l1_ratio}")
        
        # ElasticNet交叉验证
        elastic_cv = ElasticNetCV(
            alphas=alpha_values,
            l1_ratio=l1_ratio,
            cv=cv_folds,
            random_state=random_state,
            max_iter=2000,
            selection='random'
        )
        
        elastic_cv.fit(X_scaled, y_clean)
        
        # 获取选中的特征
        coefficients = elastic_cv.coef_
        selected_mask = np.abs(coefficients) > 1e-6  # 非零系数
        selected_features = X_clean.columns[selected_mask].tolist()
        
        # 强制包含指定特征
        final_features = list(set(selected_features + force_include))
        final_features = [f for f in final_features if f in X_clean.columns]
        
        if len(final_features) == 0:
            continue
            
        # 用选中的特征训练逻辑回归评估性能
        lr = LogisticRegression(random_state=random_state, max_iter=1000)
        cv_scores = cross_val_score(
            lr, X_scaled[final_features], y_clean,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='roc_auc'
        )
        
        mean_auc = cv_scores.mean()
        
        results_summary.append({
            'l1_ratio': l1_ratio,
            'alpha': elastic_cv.alpha_,
            'n_features': len(final_features),
            'cv_auc': mean_auc,
            'cv_std': cv_scores.std(),
            'features': final_features
        })
        
        print(f"    选中特征数: {len(final_features)}, CV AUC: {mean_auc:.4f} ± {cv_scores.std():.4f}")
        
        # 修复的选择逻辑：优先选择AUC最高的参数组合
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = {
                'l1_ratio': l1_ratio,
                'alpha': elastic_cv.alpha_,
                'model': elastic_cv
            }
            best_selected_features = final_features

    # ───────────────── 结果汇总 ─────────────────
    results_df = pd.DataFrame(results_summary).sort_values('cv_auc', ascending=False)
    
    print(f"\n=== ElasticNet参数搜索结果 ===")
    print(f"最佳参数: l1_ratio={best_params['l1_ratio']}, alpha={best_params['alpha']:.6f}")
    print(f"最佳CV AUC: {best_auc:.4f}")
    print(f"选中特征数: {len(best_selected_features)}")
    
    print(f"\n前5个参数组合:")
    for i, (_, row) in enumerate(results_df.head().iterrows(), 1):
        print(f"  {i}. l1_ratio={row['l1_ratio']:.1f}, features={row['n_features']:2d}, "
              f"AUC={row['cv_auc']:.4f}±{row['cv_std']:.4f}")

    # ───────────────── 最终模型和特征重要性 ─────────────────
    print(f"\n=== 最终选中的特征 ===")
    
    # 用最佳参数重新训练获取系数
    best_model = best_params['model']
    final_coefficients = best_model.coef_
    
    # 创建特征重要性表
    feature_importance = pd.DataFrame({
        '特征名': X_clean.columns,
        '系数': final_coefficients,
        '绝对系数': np.abs(final_coefficients),
        '是否选中': np.abs(final_coefficients) > 1e-6
    })
    
    # 添加强制保留标记
    feature_importance['强制保留'] = feature_importance['特征名'].isin(force_include)
    
    # 确保强制保留的特征被标记为选中
    force_mask = feature_importance['强制保留']
    feature_importance.loc[force_mask, '是否选中'] = True
    
    # 按重要性排序
    feature_importance = feature_importance.sort_values('绝对系数', ascending=False)
    
    # 显示选中的特征
    selected_importance = feature_importance[feature_importance['是否选中']].copy()
    
    print(f"共选中 {len(selected_importance)} 个特征:")
    for i, (_, row) in enumerate(selected_importance.iterrows(), 1):
        direction = "↑风险" if row['系数'] > 0 else "↓保护"
        force_mark = " [强制保留]" if row['强制保留'] else ""
        print(f"  {i:2d}. {row['特征名']:35s} 系数:{row['系数']:8.4f} ({direction}){force_mark}")

    # ───────────────── 最终验证 ─────────────────
    print(f"\n=== 最终模型验证 ===")
    
    final_lr = LogisticRegression(random_state=random_state, max_iter=1000)
    final_features = selected_importance['特征名'].tolist()
    
    # 5折交叉验证
    final_cv = cross_validate(
        final_lr, X_scaled[final_features], y_clean,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring=['roc_auc', 'accuracy'],
        return_train_score=True
    )
    
    print(f"5折交叉验证结果:")
    print(f"  测试集AUC: {final_cv['test_roc_auc'].mean():.4f} ± {final_cv['test_roc_auc'].std():.4f}")
    print(f"  测试集准确率: {final_cv['test_accuracy'].mean():.4f} ± {final_cv['test_accuracy'].std():.4f}")
    print(f"  训练集AUC: {final_cv['train_roc_auc'].mean():.4f} ± {final_cv['train_roc_auc'].std():.4f}")
    
    overfitting = final_cv['train_roc_auc'].mean() - final_cv['test_roc_auc'].mean()
    print(f"  过拟合程度: {overfitting:.4f}")
    
    if overfitting > 0.1:
        print(f"  存在过拟合风险")
    elif overfitting < 0.05:
        print(f"  模型泛化良好")

    # ───────────────── 保存结果 ─────────────────
    # 保存参数搜索结果
    results_df.to_csv('ElasticNet参数搜索结果.csv', index=False, encoding='utf-8-sig')
    
    # 保存特征重要性
    feature_importance.to_csv('ElasticNet特征重要性.csv', index=False, encoding='utf-8-sig')
    
    # 保存最终选中的特征列表
    pd.DataFrame({'特征名': final_features}).to_csv('ElasticNet最终特征.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存:")
    print(f"  - ElasticNet参数搜索结果.csv")
    print(f"  - ElasticNet特征重要性.csv") 
    print(f"  - ElasticNet最终特征.csv")

    return {
        'best_params': best_params,
        'selected_features': final_features,
        'feature_importance': feature_importance,
        'cv_results': final_cv,
        'search_results': results_df,
        'X_clean': X_clean,
        'y_clean': y_clean
    }


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('lasso筛选用数据.csv', encoding='utf-8')
    
    # 运行改进的ElasticNet筛选
    results = improved_elasticnet_selection(
        data=data,
        outcome_col="两年内是否复发",
        force_include=['恶性值'],  # 强制保留恶性值
        target_features=15,       # 目标保留15个特征
        cv_folds=10,
        random_state=42
    )
    
    print(f"\n=== 筛选完成 ===")
    print(f"最终选中 {len(results['selected_features'])} 个特征")
    print(f"CV AUC: {results['cv_results']['test_roc_auc'].mean():.4f}")
    
    # 显示最终特征列表
    print(f"\n最终特征列表:")
    for i, feature in enumerate(results['selected_features'], 1):
        print(f"{i:2d}. {feature}")
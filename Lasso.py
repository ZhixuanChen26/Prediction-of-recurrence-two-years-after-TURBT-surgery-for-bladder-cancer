import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


def relaxed_lasso_logistic_regression(data, outcome_col, force_include=None,
                                     target_features=15, cv_folds=10, 
                                     random_state=42, exclude_cols=None):
    
    if exclude_cols is None:
        exclude_cols = []
    if force_include is None:
        force_include = []

    print(f"=== 极宽松LASSO回归 ===")
    print(f"目标特征数: {target_features}")
    print(f"强制保留特征: {force_include}")

    # 数据预处理
    data_processed = data.copy()
    
    reference_categories = {
        'bmi值': '1.0', '部位值': '3.0'
    }
    categorical_vars_to_encode = ['bmi值', '部位值']
    
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

    # 准备数据
    selected_vars = [col for col in data_processed.columns
                     if col != outcome_col and col not in exclude_cols]
    
    X = data_processed[selected_vars].copy()
    y = data_processed[outcome_col].copy()

    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    print(f"\n数据概况:")
    print(f"  样本量: {len(X_clean)}, 特征数: {len(selected_vars)}")
    print(f"  事件数: {sum(y_clean == 1)} ({sum(y_clean == 1) / len(y_clean) * 100:.1f}%)")

    # 列类型划分
    ordinal_cols = ['恶性值', '浸润值', '年龄值']
    numeric_features = []
    categorical_features = []

    for col in X_clean.columns:
        if col in ordinal_cols or not col.startswith(('bmi值_', '形状值_', '部位值_', 't值_')):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    # 预处理管道
    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(('cat', StandardScaler(), categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # 极宽松的LASSO参数
    c_values = np.logspace(-2, 4, 30) 
    
    lasso_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('lasso', LogisticRegressionCV(
            Cs=c_values,
            penalty='l1',
            solver='liblinear',
            cv=cv_folds,
            scoring='roc_auc',
            random_state=random_state,
            max_iter=5000,
            class_weight='balanced'
        ))
    ])

    # 拟合模型
    print(f"\n正在进行{cv_folds}折交叉验证...")
    lasso_pipeline.fit(X_clean, y_clean)

    best_C = lasso_pipeline.named_steps['lasso'].C_[0]
    print(f"最优C = {best_C:.4f}, lambda = {1 / best_C:.4f}")

    # 特征名和系数
    feature_names = numeric_features + categorical_features
    coefficients = lasso_pipeline.named_steps['lasso'].coef_[0]
    
    # 宽松的阈值判断特征是否被选中
    coef_threshold = max(1e-8, np.percentile(np.abs(coefficients[coefficients != 0]), 5))
    
    feature_importance = pd.DataFrame({
        '特征名': feature_names,
        '系数': coefficients,
        '绝对系数': np.abs(coefficients),
        '是否选中': np.abs(coefficients) > coef_threshold
    }).sort_values('绝对系数', ascending=False)

    # 强制包含指定特征
    if force_include:
        force_mask = feature_importance['特征名'].isin(force_include)
        feature_importance.loc[force_mask, '是否选中'] = True

    selected_features = feature_importance[feature_importance['是否选中']]['特征名'].tolist()

    # 嵌套交叉验证评估
    print(f"\n进行嵌套交叉验证评估...")
    nested_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    nested_scores = cross_validate(
        lasso_pipeline, X_clean, y_clean,
        cv=nested_cv,
        scoring=['roc_auc', 'accuracy'],
        return_train_score=True
    )

    mean_auc = nested_scores['test_roc_auc'].mean()
    auc_std = nested_scores['test_roc_auc'].std()
    overfitting = nested_scores['train_roc_auc'].mean() - mean_auc

    print(f"\n=== LASSO结果 ===")
    print(f"选中特征数: {len(selected_features)}")
    print(f"嵌套CV AUC: {mean_auc:.4f} ± {auc_std:.4f}")
    print(f"过拟合程度: {overfitting:.4f}")

    selected_importance = feature_importance[feature_importance['是否选中']].copy()
    
    print(f"\n选中的特征（按重要性排序）:")
    feature_list_with_coef = []
    for i, (_, row) in enumerate(selected_importance.iterrows(), 1):
        direction = "↑风险" if row['系数'] > 0 else "↓保护"
        force_mark = " [强制保留]" if row['特征名'] in force_include else ""
        feature_desc = f"{row['特征名']} (系数:{row['系数']:.4f}, {direction})"
        feature_list_with_coef.append(feature_desc)
        print(f"  {i:2d}. {row['特征名']:35s} 系数:{row['系数']:8.4f} ({direction}){force_mark}")

    # 保存结果
    feature_importance.to_csv('LASSO特征重要性.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'特征描述': feature_list_with_coef}).to_csv('极宽松LASSO最终特征.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到: LASSO特征重要性.csv, LASSO最终特征.csv")

    return {
        'selected_features': selected_features,
        'feature_list_with_coef': feature_list_with_coef,
        'feature_importance': feature_importance,
        'cv_auc': mean_auc,
        'cv_std': auc_std,
        'best_C': best_C,
        'overfitting': overfitting
    }


if __name__ == "__main__":
    data = pd.read_csv('lasso筛选用数据.csv', encoding='utf-8')

    results = relaxed_lasso_logistic_regression(
        data=data,
        outcome_col="两年内是否复发",
        force_include=['恶性值'],
        target_features=15,
        cv_folds=10,
        random_state=42
    )

    print(f"\n=== 最终结果 ===")
    print(f"选中特征数: {len(results['selected_features'])}")
    print(f"CV AUC: {results['cv_auc']:.4f}")
    
    print(f"\n最终特征列表（带系数和方向）:")
    for i, feature_desc in enumerate(results['feature_list_with_coef'], 1):
        print(f"{i:2d}. {feature_desc}")
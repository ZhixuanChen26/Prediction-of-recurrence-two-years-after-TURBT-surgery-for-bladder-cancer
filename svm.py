import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, brier_score_loss, roc_curve
import warnings
warnings.filterwarnings('ignore')


class FrequencyTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=20):
        self.max_categories = max_categories
        self.top_categories_ = {}
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if X[col].dtype == 'object':
                    value_counts = X[col].value_counts()
                    if len(value_counts) > self.max_categories:
                        self.top_categories_[col] = value_counts.head(self.max_categories).index.tolist()
                    else:
                        self.top_categories_[col] = value_counts.index.tolist()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if isinstance(X_transformed, pd.DataFrame):
            for col in X_transformed.columns:
                if col in self.top_categories_:
                    mask = ~X_transformed[col].isin(self.top_categories_[col])
                    X_transformed.loc[mask, col] = 'Other'
        return X_transformed


def create_svm_preprocessor(exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    
    def build_preprocessor(feature_cols, X_sample):
        numeric_features = []
        bmi_features = []
        site_features = []
        string_features = []
        
        for col in feature_cols:
            if col in exclude_cols:
                continue
            elif col == 'bmi值':
                bmi_features.append(col)
            elif col == '部位值':
                site_features.append(col)
            elif X_sample[col].dtype == 'object':
                string_features.append(col)
            else:
                numeric_features.append(col)
        
        transformers = []
        
        if numeric_features:
            numeric_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, numeric_features))
        
        if bmi_features:
            bmi_unique = X_sample['bmi值'].unique()
            if X_sample['bmi值'].dtype in ['int64', 'float64']:
                bmi_categories = [sorted(bmi_unique)]
                ref_category = 1 if 1 in bmi_unique else 1.0
            else:
                bmi_categories = [sorted(X_sample['bmi值'].astype(str).unique().tolist())]
                ref_category = '1.0' if '1.0' in bmi_categories[0] else '1'
            
            bmi_encoder = OneHotEncoder(
                categories=bmi_categories, drop=[ref_category], 
                handle_unknown='ignore', sparse_output=False
            )
            transformers.append(('bmi', bmi_encoder, bmi_features))
        
        if site_features:
            site_unique = X_sample['部位值'].unique()
            
            if X_sample['部位值'].dtype in ['int64', 'float64']:
                site_categories = [sorted(site_unique)]
                ref_category = 3 if 3 in site_unique else 3.0
            else:
                site_categories = [sorted(X_sample['部位值'].astype(str).unique().tolist())]
                ref_category = '3.0' if '3.0' in site_categories[0] else '3'
            
            site_encoder = OneHotEncoder(
                categories=site_categories, drop=[ref_category], 
                handle_unknown='ignore', sparse_output=False
            )
            transformers.append(('site', site_encoder, site_features))
        
        if string_features:
            string_pipeline = Pipeline([
                ('freq_trim', FrequencyTrimmer(max_categories=15)),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat_str', string_pipeline, string_features))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        return preprocessor
    
    return build_preprocessor


def calculate_youden_threshold(y_true, y_prob):
    """计算Youden最优阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    """基于指定阈值计算所有指标"""
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算指标
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    # 从混淆矩阵中提取TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    
    # 计算敏感性、特异性、准确性
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
    
    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'brier': brier,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'ppv': ppv,
        'npv': npv,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'confusion_matrix': cm
    }


def improved_nested_cv_svm_youden(data, outcome_col, exclude_cols=None, outer_cv_folds=5, 
                                 inner_cv_folds=5, param_search_iterations=30, 
                                 scoring_metric='roc_auc', random_state=42):
    
    if exclude_cols is None:
        exclude_cols = []
    
    # 数据预处理：红细胞计数特征工程
    data_processed = data.copy()
    rbc_col = '红细胞计数(RBC#)-尿液'
    
    if rbc_col in data_processed.columns:
        rbc_values = data_processed[rbc_col]
        data_processed[rbc_col] = np.log1p(rbc_values)
    
    feature_cols = [col for col in data_processed.columns 
                   if col != outcome_col and col not in exclude_cols]
    
    X = data_processed[feature_cols].copy()
    y = data_processed[outcome_col].copy()
    
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]
    
    build_preprocessor = create_svm_preprocessor(exclude_cols)
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)
    
    # 存储每折的结果
    fold_metrics = []
    fold_thresholds = []
    best_params_list = []
    best_fold_info = {'best_auc': 0, 'best_fold': 1, 'best_params': None}
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_clean, y_clean), 1):
        X_outer_train, X_outer_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_outer_train, y_outer_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        preprocessor = build_preprocessor(feature_cols, X_outer_train)
        
        base_svm = SVC(
            probability=True,
            random_state=random_state,
            max_iter=3000
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_svm)
        ])
        
        C_strong_reg = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
        
        param_dist = {
            'classifier__kernel': ['linear'],
            'classifier__C': C_strong_reg,
            'classifier__class_weight': [None, 'balanced']
        }
        
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state+fold_idx)
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=6,
            cv=inner_cv, scoring=scoring_metric,
            random_state=random_state+fold_idx, n_jobs=1, verbose=0
        )
        
        search.fit(X_outer_train, y_outer_train)
        
        best_params = search.best_params_
        best_params_list.append(best_params)
        
        # 获取测试集预测概率
        y_test_pred_proba = search.predict_proba(X_outer_test)[:, 1]
        
        # 计算Youden最优阈值
        optimal_threshold = calculate_youden_threshold(y_outer_test, y_test_pred_proba)
        fold_thresholds.append(optimal_threshold)
        
        # 基于最优阈值计算指标
        metrics = calculate_metrics_at_threshold(y_outer_test, y_test_pred_proba, optimal_threshold)
        fold_metrics.append(metrics)
        
        print(f"Fold {fold_idx}: 最优阈值={optimal_threshold:.4f}, AUC={metrics['auc']:.4f}, "
              f"敏感性={metrics['sensitivity']:.4f}, 特异性={metrics['specificity']:.4f}, F1={metrics['f1']:.4f}")
        
        if metrics['auc'] > best_fold_info['best_auc']:
            best_fold_info.update({
                'best_auc': metrics['auc'],
                'best_fold': fold_idx,
                'best_params': best_params,
                'best_threshold': optimal_threshold
            })
    
    # 计算所有指标的均值和置信区间
    metric_names = ['auc', 'pr_auc', 'f1', 'brier', 'sensitivity', 'specificity', 
                   'accuracy', 'ppv', 'npv', 'tp', 'tn', 'fp', 'fn']
    
    results = {}
    for metric in metric_names:
        values = [fold_metrics[i][metric] for i in range(len(fold_metrics))]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        se_val = std_val / np.sqrt(len(values))
        ci_lower = mean_val - 1.96 * se_val
        ci_upper = mean_val + 1.96 * se_val
        
        results[f'{metric}_mean'] = mean_val
        results[f'{metric}_std'] = std_val
        results[f'{metric}_se'] = se_val
        results[f'{metric}_ci'] = (ci_lower, ci_upper)
        results[f'{metric}_values'] = values
    
    # 计算阈值的均值和置信区间
    threshold_mean = np.mean(fold_thresholds)
    threshold_std = np.std(fold_thresholds, ddof=1)
    threshold_se = threshold_std / np.sqrt(len(fold_thresholds))
    threshold_ci = (threshold_mean - 1.96 * threshold_se, threshold_mean + 1.96 * threshold_se)
    
    results.update({
        'threshold_mean': threshold_mean,
        'threshold_ci': threshold_ci,
        'threshold_values': fold_thresholds,
        'best_fold_info': best_fold_info,
        'best_params_list': best_params_list,
        'X_clean': X_clean,
        'y_clean': y_clean,
        'feature_cols': feature_cols
    })
    
    return results


def save_youden_metrics_results(results):
    """保存基于Youden最优阈值的所有指标到文件"""
    metrics_data = {
        'Metric': ['AUC', 'PR-AUC', 'F1 Score', 'Brier Score', 'Sensitivity', 'Specificity', 
                  'Accuracy', 'PPV', 'NPV', 'TP', 'TN', 'FP', 'FN', 'Optimal Threshold'],
        'Mean': [
            results['auc_mean'], results['pr_auc_mean'], results['f1_mean'], results['brier_mean'],
            results['sensitivity_mean'], results['specificity_mean'], results['accuracy_mean'],
            results['ppv_mean'], results['npv_mean'], results['tp_mean'], results['tn_mean'],
            results['fp_mean'], results['fn_mean'], results['threshold_mean']
        ],
        '95% CI Lower': [
            results['auc_ci'][0], results['pr_auc_ci'][0], results['f1_ci'][0], results['brier_ci'][0],
            results['sensitivity_ci'][0], results['specificity_ci'][0], results['accuracy_ci'][0],
            results['ppv_ci'][0], results['npv_ci'][0], results['tp_ci'][0], results['tn_ci'][0],
            results['fp_ci'][0], results['fn_ci'][0], results['threshold_ci'][0]
        ],
        '95% CI Upper': [
            results['auc_ci'][1], results['pr_auc_ci'][1], results['f1_ci'][1], results['brier_ci'][1],
            results['sensitivity_ci'][1], results['specificity_ci'][1], results['accuracy_ci'][1],
            results['ppv_ci'][1], results['npv_ci'][1], results['tp_ci'][1], results['tn_ci'][1],
            results['fp_ci'][1], results['fn_ci'][1], results['threshold_ci'][1]
        ]
    }
    
    # 添加格式化的结果列
    metrics_data['Mean (95% CI)'] = []
    for i, metric in enumerate(metrics_data['Metric']):
        mean_val = metrics_data['Mean'][i]
        ci_lower = metrics_data['95% CI Lower'][i]
        ci_upper = metrics_data['95% CI Upper'][i]
        
        if metric in ['TP', 'TN', 'FP', 'FN']:
            # 对于计数指标，保留2位小数
            formatted = f"{mean_val:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"
        else:
            # 对于比例指标，保留4位小数
            formatted = f"{mean_val:.4f} ({ci_lower:.4f}, {ci_upper:.4f})"
        
        metrics_data['Mean (95% CI)'].append(formatted)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv('SVM_Youden最优阈值指标.csv', index=False, encoding='utf-8-sig')
    
    print("\n=== SVM模型基于Youden最优阈值的性能指标 ===")
    print(metrics_df[['Metric', 'Mean (95% CI)']].to_string(index=False))
    
    return metrics_df


def train_final_svm_model(data, outcome_col, results, exclude_cols=None, random_state=42):
    if exclude_cols is None:
        exclude_cols = []
    
    X_clean = results['X_clean']
    y_clean = results['y_clean']
    feature_cols = results['feature_cols']
    best_params = results['best_fold_info']['best_params']
    
    build_preprocessor = create_svm_preprocessor(exclude_cols)
    preprocessor = build_preprocessor(feature_cols, X_clean)
    
    classifier_final = SVC(
        probability=True,
        random_state=random_state,
        max_iter=5000
    )
    
    classifier_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
    classifier_final.set_params(**classifier_params)
    
    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier_final)
    ])
    
    final_model.fit(X_clean, y_clean)
    
    return final_model


def main():
    data = pd.read_csv('rf_input.csv', encoding='utf-8')
    
    # 基于Youden最优阈值的嵌套交叉验证
    results = improved_nested_cv_svm_youden(
        data=data, outcome_col="两年内是否复发", exclude_cols=None,
        outer_cv_folds=5, inner_cv_folds=5, param_search_iterations=30,
        scoring_metric='roc_auc', random_state=42
    )
    
    # 训练最终模型
    final_model = train_final_svm_model(
        data, "两年内是否复发", results, random_state=42
    )
    
    # 保存结果
    metrics_df = save_youden_metrics_results(results)
    
    # 保存模型摘要
    best_params = results['best_fold_info']['best_params']
    summary = {
        'model_type': 'SVM_Youden_Optimal',
        'feature_engineering': '红细胞计数Log1p变换',
        'best_fold': results['best_fold_info']['best_fold'],
        'optimal_threshold_mean': results['threshold_mean'],
        'auc_mean_ci': f"{results['auc_mean']:.4f} ({results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f})",
        'sensitivity_mean_ci': f"{results['sensitivity_mean']:.4f} ({results['sensitivity_ci'][0]:.4f}, {results['sensitivity_ci'][1]:.4f})",
        'specificity_mean_ci': f"{results['specificity_mean']:.4f} ({results['specificity_ci'][0]:.4f}, {results['specificity_ci'][1]:.4f})",
        'f1_mean_ci': f"{results['f1_mean']:.4f} ({results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f})",
        'best_kernel': best_params['classifier__kernel'],
        'best_C': best_params['classifier__C'],
        'best_class_weight': str(best_params['classifier__class_weight'])
    }
    
    pd.DataFrame([summary]).to_csv('SVM_Youden建模摘要.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== SVM基于Youden最优阈值建模完成 ===")
    print(f"最优阈值: {results['threshold_mean']:.4f}")
    print(f"AUC: {results['auc_mean']:.4f} ({results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f})")
    print(f"敏感性: {results['sensitivity_mean']:.4f} ({results['sensitivity_ci'][0]:.4f}, {results['sensitivity_ci'][1]:.4f})")
    print(f"特异性: {results['specificity_mean']:.4f} ({results['specificity_ci'][0]:.4f}, {results['specificity_ci'][1]:.4f})")
    print(f"F1 Score: {results['f1_mean']:.4f} ({results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f})")
    print(f"核函数: {summary['best_kernel']}")
    print(f"正则化参数C: {summary['best_C']}")
    print(f"类别权重: {summary['best_class_weight']}")
    print(f"最佳Fold: {results['best_fold_info']['best_fold']}")
    
    return results, final_model


if __name__ == "__main__":
    results, final_model = main()
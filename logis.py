import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


def prepare_data_for_logistic(data, feature_cols, outcome_col, onehot_cols=None, lg1p_cols=None):
    
    if onehot_cols is None:
        onehot_cols = {'部位值': '3.0', 'bmi值': '1.0'}
    
    if lg1p_cols is None:
        lg1p_cols = ['红细胞计数(RBC#)-尿液']
    
    # 提取特征和结果变量
    X = data[feature_cols].copy()
    y = data[outcome_col].copy()
    
    # 删除结果变量缺失的行
    mask = ~y.isnull()
    X = X.loc[mask]
    y = y.loc[mask]
    
    # 处理特征
    X_parts = []
    feature_names = []
    
    for col in feature_cols:
        if col in onehot_cols:
            # One-hot编码，去掉参考类别
            col_str = X[col].astype(str)
            dummies = pd.get_dummies(col_str, prefix=col, dtype=int)
            
            # 移除参考类别
            ref_col = f"{col}_{onehot_cols[col]}"
            if ref_col in dummies.columns:
                dummies = dummies.drop(ref_col, axis=1)
            
            X_parts.append(dummies)
            feature_names.extend(dummies.columns.tolist())
            
        elif col in lg1p_cols:
            # 对指定列进行lg1p变换
            transformed_col = np.log1p(X[col])
            new_col_name = f"{col}_lg1p"
            transformed_df = pd.DataFrame({new_col_name: transformed_col}, index=X.index)
            
            X_parts.append(transformed_df)
            feature_names.append(new_col_name)
            
        else:
            # 数值型特征，直接使用
            X_parts.append(X[[col]])
            feature_names.append(col)
    
    # 合并所有特征
    X_processed = pd.concat(X_parts, axis=1)
    
    return X_processed, y, feature_names


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


def enhanced_nested_cv_logistic_youden(X, y, feature_names, outer_cv_folds=5, inner_cv_folds=5, 
                                      scoring_metric='roc_auc', random_state=42):
    
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)
    
    # 存储每折的结果
    fold_metrics = []
    fold_thresholds = []
    best_params_list = []
    best_fold_info = {'best_auc': 0, 'best_fold': 1, 'best_params': None}
    
    # 扩展的参数搜索空间
    def get_param_combinations():
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        class_weights = [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
        
        combinations = []
        
        # L2正则化 + lbfgs
        for C in C_values:
            for cw in class_weights:
                combinations.append({
                    'C': C, 'penalty': 'l2', 'solver': 'lbfgs', 
                    'class_weight': cw, 'max_iter': 2000
                })
        
        # L1正则化 + liblinear
        for C in C_values[:4]:  # L1计算较慢，减少参数
            for cw in class_weights:
                combinations.append({
                    'C': C, 'penalty': 'l1', 'solver': 'liblinear', 
                    'class_weight': cw, 'max_iter': 2000
                })
        
        # ElasticNet + saga
        for C in [0.1, 1.0, 10.0]:
            for l1_ratio in [0.3, 0.5, 0.7]:
                for cw in [None, 'balanced']:
                    combinations.append({
                        'C': C, 'penalty': 'elasticnet', 'solver': 'saga',
                        'l1_ratio': l1_ratio, 'class_weight': cw, 'max_iter': 2000
                    })
        
        return combinations
    
    param_combinations = get_param_combinations()
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
        y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 内层CV参数搜索
        best_score = 0
        best_params = None
        
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state+fold_idx)
        
        # 手动网格搜索以处理参数兼容性
        for params in param_combinations:
            try:
                cv_scores = []
                for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
                    X_inner_train = X_outer_train.iloc[inner_train_idx]
                    X_inner_val = X_outer_train.iloc[inner_val_idx]
                    y_inner_train = y_outer_train.iloc[inner_train_idx]
                    y_inner_val = y_outer_train.iloc[inner_val_idx]
                    
                    model = LogisticRegression(random_state=random_state, **params)
                    model.fit(X_inner_train, y_inner_train)
                    y_pred = model.predict_proba(X_inner_val)[:, 1]
                    score = roc_auc_score(y_inner_val, y_pred)
                    cv_scores.append(score)
                
                mean_score = np.mean(cv_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    
            except Exception as e:
                continue  # 跳过不兼容的参数组合
        
        if best_params is None:
            # 如果没找到合适参数，使用默认参数
            best_params = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'class_weight': None, 'max_iter': 2000}
        
        best_params_list.append(best_params)
        
        # 用最佳参数训练最终模型
        final_model = LogisticRegression(random_state=random_state, **best_params)
        final_model.fit(X_outer_train, y_outer_train)
        
        # 获取测试集预测概率
        y_test_pred_proba = final_model.predict_proba(X_outer_test)[:, 1]
        
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
        'X_clean': X,
        'y_clean': y,
        'feature_names': feature_names
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
    metrics_df.to_csv('逻辑回归_Youden最优阈值指标.csv', index=False, encoding='utf-8-sig')
    
    print("\n=== 逻辑回归模型基于Youden最优阈值的性能指标 ===")
    print(metrics_df[['Metric', 'Mean (95% CI)']].to_string(index=False))
    
    return metrics_df


def fit_logistic_regression(X, y, feature_names, best_params=None):
    if best_params is None:
        best_params = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'class_weight': None, 'max_iter': 2000}
    
    sklearn_model = LogisticRegression(random_state=42, **best_params)
    sklearn_model.fit(X, y)
    
    return sklearn_model


def main(data_file, outcome_col='两年内是否复发', exclude_cols=None,
         onehot_cols=None, lg1p_cols=None, cv_folds=5, random_state=42):
    
    if onehot_cols is None:
        onehot_cols = {'部位值': '3.0', 'bmi值': '1.0'}
    
    if lg1p_cols is None:
        lg1p_cols = ['红细胞计数(RBC#)-尿液']
    
    if exclude_cols is None:
        exclude_cols = []
    
    # 1. 读取数据
    data = pd.read_csv(data_file, encoding='utf-8')
    
    # 2. 自动识别特征列
    feature_cols = [col for col in data.columns 
                   if col != outcome_col and col not in exclude_cols]
    
    # 3. 数据准备
    X, y, feature_names = prepare_data_for_logistic(data, feature_cols, outcome_col, onehot_cols, lg1p_cols)
    
    # 4. 基于Youden最优阈值的嵌套交叉验证
    results = enhanced_nested_cv_logistic_youden(
        X, y, feature_names, 
        outer_cv_folds=5, inner_cv_folds=5, 
        scoring_metric='roc_auc', 
        random_state=random_state
    )
    
    # 5. 训练最终模型
    best_params = results['best_fold_info']['best_params']
    sklearn_model = fit_logistic_regression(X, y, feature_names, best_params)
    
    # 6. 保存结果
    metrics_df = save_youden_metrics_results(results)
    
    # 7. 保存模型摘要
    summary = {
        'model_type': 'LogisticRegression_Youden_Optimal',
        'best_fold': results['best_fold_info']['best_fold'],
        'optimal_threshold_mean': results['threshold_mean'],
        'auc_mean_ci': f"{results['auc_mean']:.4f} ({results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f})",
        'sensitivity_mean_ci': f"{results['sensitivity_mean']:.4f} ({results['sensitivity_ci'][0]:.4f}, {results['sensitivity_ci'][1]:.4f})",
        'specificity_mean_ci': f"{results['specificity_mean']:.4f} ({results['specificity_ci'][0]:.4f}, {results['specificity_ci'][1]:.4f})",
        'f1_mean_ci': f"{results['f1_mean']:.4f} ({results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f})",
        'best_penalty': best_params.get('penalty', 'l2'),
        'best_C': best_params.get('C', 1.0),
        'best_solver': best_params.get('solver', 'lbfgs'),
        'best_class_weight': str(best_params.get('class_weight', None))
    }
    
    pd.DataFrame([summary]).to_csv('逻辑回归_Youden建模摘要.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== 逻辑回归基于Youden最优阈值建模完成 ===")
    print(f"最优阈值: {results['threshold_mean']:.4f}")
    print(f"AUC: {results['auc_mean']:.4f} ({results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f})")
    print(f"敏感性: {results['sensitivity_mean']:.4f} ({results['sensitivity_ci'][0]:.4f}, {results['sensitivity_ci'][1]:.4f})")
    print(f"特异性: {results['specificity_mean']:.4f} ({results['specificity_ci'][0]:.4f}, {results['specificity_ci'][1]:.4f})")
    print(f"F1 Score: {results['f1_mean']:.4f} ({results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f})")
    print(f"正则化: {summary['best_penalty']}")
    print(f"正则化参数C: {summary['best_C']}")
    print(f"求解器: {summary['best_solver']}")
    print(f"类别权重: {summary['best_class_weight']}")
    print(f"最佳Fold: {results['best_fold_info']['best_fold']}")
    
    return sklearn_model, results, X, y


if __name__ == "__main__":
    
    results = main(
        data_file='rf_input.csv',
        outcome_col='两年内是否复发',
        exclude_cols=None,
        onehot_cols={'部位值': '3.0', 'bmi值': '1.0'},
        lg1p_cols=['红细胞计数(RBC#)-尿液'],
        cv_folds=5,
        random_state=42
    )
    
    sklearn_model, nested_cv_results, X, y = results
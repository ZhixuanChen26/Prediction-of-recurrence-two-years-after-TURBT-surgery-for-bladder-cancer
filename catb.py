import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, brier_score_loss, roc_curve
import warnings
import random
warnings.filterwarnings('ignore')


class FrequencyTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=20):
        self.max_categories = max_categories
        self.top_categories_ = {}
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if isinstance(X[col], pd.DataFrame):
                    col_data = X[col].iloc[:, 0]
                else:
                    col_data = X[col]
                
                if col_data.dtype == 'object' or col_data.dtype.name == 'category':
                    value_counts = col_data.value_counts()
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
                    X_transformed[col] = X_transformed[col].astype(str)
        return X_transformed


class CatBoostNativePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=50):
        self.max_categories = max_categories
        self.categorical_features_ = []
        self.numerical_features_ = []
        self.feature_names_ = []
        self.freq_trimmer_ = None
        
    def fit(self, X, y=None):
        self.categorical_features_ = []
        self.numerical_features_ = []
        self.feature_names_ = []
        self.freq_trimmer_ = None
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            self.freq_trimmer_ = FrequencyTrimmer(max_categories=self.max_categories)
            
            for col in X.columns:
                unique_vals = set(X[col].dropna().unique())
                
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    self.categorical_features_.append(col)
                elif len(unique_vals) == 2:
                    self.numerical_features_.append(col)
                elif len(unique_vals) >= 3 and len(unique_vals) <= 10:
                    is_integer_like = all(
                        isinstance(v, (int, float, np.integer, np.floating)) and 
                        (pd.isna(v) or float(v).is_integer())
                        for v in unique_vals
                    )
                    if is_integer_like:
                        self.categorical_features_.append(col)
                    else:
                        self.numerical_features_.append(col)
                else:
                    self.numerical_features_.append(col)
            
            if self.categorical_features_:
                unique_cat_features = list(dict.fromkeys(self.categorical_features_))
                cat_data = X[unique_cat_features]
                self.freq_trimmer_.fit(cat_data)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if isinstance(X_transformed, pd.DataFrame):
            if self.categorical_features_ and self.freq_trimmer_:
                unique_cat_features = list(dict.fromkeys(self.categorical_features_))
                cat_data = X_transformed[unique_cat_features]
                cat_data_trimmed = self.freq_trimmer_.transform(cat_data)
                X_transformed[unique_cat_features] = cat_data_trimmed
            
            for col in self.categorical_features_:
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].fillna(-1).astype(int).astype(str)
                    X_transformed[col] = X_transformed[col].replace('-1', 'missing')
            
            for col in self.numerical_features_:
                if col in X_transformed.columns:
                    X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')
        return X_transformed
    
    def get_categorical_feature_indices(self, X_transformed):
        if not self.categorical_features_:
            return []
        indices = []
        if hasattr(X_transformed, 'columns'):
            for col in self.categorical_features_:
                if col in X_transformed.columns:
                    indices.append(X_transformed.columns.get_loc(col))
        return indices


def create_catboost_preprocessor(exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    
    def build_preprocessor(feature_cols, X_sample):
        filtered_feature_cols = [col for col in feature_cols if col not in exclude_cols]
        preprocessor = CatBoostNativePreprocessor(max_categories=50)
        X_filtered = X_sample[filtered_feature_cols]
        preprocessor.fit(X_filtered)
        
        class FilteredPreprocessor(BaseEstimator, TransformerMixin):
            def __init__(self, preprocessor, feature_cols):
                self.preprocessor = preprocessor
                self.feature_cols = feature_cols
            
            def fit(self, X, y=None):
                self.preprocessor.fit(X[self.feature_cols], y)
                return self
            
            def transform(self, X):
                return self.preprocessor.transform(X[self.feature_cols])
            
            def get_categorical_feature_indices(self, X_transformed=None):
                if X_transformed is None:
                    dummy_data = pd.DataFrame(columns=self.feature_cols)
                    return self.preprocessor.get_categorical_feature_indices(dummy_data)
                else:
                    return self.preprocessor.get_categorical_feature_indices(X_transformed)
            
            @property
            def categorical_features_(self):
                return self.preprocessor.categorical_features_
            
            @property
            def feature_names_(self):
                return self.preprocessor.feature_names_
        
        return FilteredPreprocessor(preprocessor, filtered_feature_cols)
    return build_preprocessor


def cv_eval_catboost_with_early_stopping(params, X, y, cat_features, inner_cv, random_seed):
    fold_aucs = []
    fold_iterations = []
    
    for tr_idx, va_idx in inner_cv.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = CatBoostClassifier(
            **params,
            random_seed=random_seed,
            auto_class_weights='Balanced',
            early_stopping_rounds=100,
            eval_metric='AUC',
            verbose=False,
            thread_count=-1
        )
        
        model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va))
        
        y_va_pred = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, y_va_pred)
        fold_aucs.append(fold_auc)
        
        best_iter = model.get_best_iteration()
        if best_iter is None:
            best_iter = params.get('iterations', 1000)
        fold_iterations.append(best_iter)
    
    return np.mean(fold_aucs), np.mean(fold_iterations)


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


def improved_nested_cv_catboost_youden(data, outcome_col, exclude_cols=None, outer_cv_folds=5, 
                                      inner_cv_folds=5, param_search_iterations=30, 
                                      scoring_metric='roc_auc', random_state=42):
    
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in data.columns 
                   if col != outcome_col and col not in exclude_cols]
    
    X = data[feature_cols].copy()
    y = data[outcome_col].copy()
    
    mask = ~y.isnull()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]
    
    build_preprocessor = create_catboost_preprocessor(exclude_cols)
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)
    
    # 存储每折的结果
    fold_metrics = []
    fold_thresholds = []
    best_params_list = []
    best_iterations_list = []
    best_fold_info = {'best_auc': 0, 'best_fold': 1, 'best_params': None}
    
    param_dist = {
        'iterations': [2000, 3000],
        'learning_rate': [0.005, 0.008, 0.01, 0.015],
        'depth': [3, 4],  
        'l2_leaf_reg': [10, 15, 20, 25, 30],
        'min_data_in_leaf': [40, 60, 80, 100],
        'subsample': [0.8], 
        'colsample_bylevel': [0.8],  
        'random_strength': [8, 10, 15], 
        'border_count': [32, 64],
        'one_hot_max_size': [4],
        'loss_function': ['Logloss']
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_clean, y_clean), 1):
        X_outer_train, X_outer_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_outer_train, y_outer_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        
        preprocessor = build_preprocessor(feature_cols, X_outer_train)
        preprocessor.fit(X_outer_train, y_outer_train)
        X_train_processed = preprocessor.transform(X_outer_train)
        cat_features = preprocessor.get_categorical_feature_indices(X_train_processed)
        
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state+fold_idx)
        
        random.seed(random_state + fold_idx)
        best_auc = 0
        best_params = None
        best_iter_mean = 0
        
        for search_iter in range(param_search_iterations):
            current_params = {key: random.choice(param_dist[key]) for key in param_dist.keys()}
            
            try:
                cv_auc, cv_iter = cv_eval_catboost_with_early_stopping(
                    current_params, X_train_processed, y_outer_train, 
                    cat_features, inner_cv, random_state + fold_idx
                )
                
                if cv_auc > best_auc:
                    best_auc = cv_auc
                    best_params = current_params.copy()
                    best_iter_mean = cv_iter
                    
            except Exception:
                continue
        
        if best_params is None:
            best_params = {
                'iterations': 1000, 'learning_rate': 0.01, 'depth': 3, 
                'l2_leaf_reg': 15, 'subsample': 0.8, 'colsample_bylevel': 0.8,
                'random_strength': 0, 'border_count': 128, 'one_hot_max_size': 4
            }
            best_iter_mean = 500
        
        best_params_list.append(best_params)
        
        final_params = best_params.copy()
        final_iterations = max(50, int(best_iter_mean))
        final_params['iterations'] = final_iterations
        
        final_catboost = CatBoostClassifier(
            **final_params,
            random_seed=random_state,
            verbose=False,
            thread_count=-1,
            auto_class_weights='Balanced',
            use_best_model=False
        )
        
        final_catboost.fit(X_train_processed, y_outer_train, cat_features=cat_features)
        best_iterations_list.append(final_iterations)
        
        X_test_processed = preprocessor.transform(X_outer_test)
        y_test_pred_proba = final_catboost.predict_proba(X_test_processed)[:, 1]
        
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
                'best_iteration': final_iterations,
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
    
    # 计算迭代次数均值
    mean_iterations = np.mean(best_iterations_list)
    
    results.update({
        'threshold_mean': threshold_mean,
        'threshold_ci': threshold_ci,
        'threshold_values': fold_thresholds,
        'best_fold_info': best_fold_info,
        'best_params_list': best_params_list,
        'best_iterations_list': best_iterations_list,
        'mean_iterations': mean_iterations,
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
    metrics_df.to_csv('CatBoost_Youden最优阈值指标.csv', index=False, encoding='utf-8-sig')
    
    print("\n=== CatBoost模型基于Youden最优阈值的性能指标 ===")
    print(metrics_df[['Metric', 'Mean (95% CI)']].to_string(index=False))
    
    return metrics_df


def train_final_model_improved(data, outcome_col, results, exclude_cols=None, random_state=42):
    if exclude_cols is None:
        exclude_cols = []
    
    X_clean = results['X_clean']
    y_clean = results['y_clean']
    feature_cols = results['feature_cols']
    best_params = results['best_fold_info']['best_params']
    mean_iterations = results['mean_iterations']
    
    build_preprocessor = create_catboost_preprocessor(exclude_cols)
    preprocessor = build_preprocessor(feature_cols, X_clean)
    
    preprocessor.fit(X_clean, y_clean)
    X_processed = preprocessor.transform(X_clean)
    cat_features = preprocessor.get_categorical_feature_indices(X_processed)
    
    final_iterations = max(50, int(mean_iterations))
    final_params = best_params.copy()
    final_params['iterations'] = final_iterations
    
    classifier_final = CatBoostClassifier(
        **final_params,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
        auto_class_weights='Balanced',
        use_best_model=False
    )
    
    classifier_final.fit(X_processed, y_clean, cat_features=cat_features)
    
    return classifier_final, preprocessor, final_iterations


def main():
    data = pd.read_csv('rf_input.csv', encoding='utf-8')
    
    # 基于Youden最优阈值的嵌套交叉验证
    results = improved_nested_cv_catboost_youden(
        data=data, outcome_col="两年内是否复发", exclude_cols=None,
        outer_cv_folds=5, inner_cv_folds=5, param_search_iterations=25,
        scoring_metric='roc_auc', random_state=42
    )
    
    # 训练最终模型
    final_model, preprocessor, best_iteration = train_final_model_improved(
        data, "两年内是否复发", results, random_state=42
    )
    
    # 保存结果
    metrics_df = save_youden_metrics_results(results)
    
    # 保存模型摘要
    summary = {
        'model_type': 'CatBoost_Youden_Optimal',
        'mean_iterations': results['mean_iterations'],
        'final_model_iterations': best_iteration,
        'optimal_threshold_mean': results['threshold_mean'],
        'auc_mean_ci': f"{results['auc_mean']:.4f} ({results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f})",
        'sensitivity_mean_ci': f"{results['sensitivity_mean']:.4f} ({results['sensitivity_ci'][0]:.4f}, {results['sensitivity_ci'][1]:.4f})",
        'specificity_mean_ci': f"{results['specificity_mean']:.4f} ({results['specificity_ci'][0]:.4f}, {results['specificity_ci'][1]:.4f})",
        'f1_mean_ci': f"{results['f1_mean']:.4f} ({results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f})",
        'best_fold': results['best_fold_info']['best_fold']
    }
    
    pd.DataFrame([summary]).to_csv('CatBoost_Youden建模摘要.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== CatBoost基于Youden最优阈值建模完成 ===")
    print(f"最优阈值: {results['threshold_mean']:.4f}")
    print(f"AUC: {results['auc_mean']:.4f} ({results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f})")
    print(f"敏感性: {results['sensitivity_mean']:.4f} ({results['sensitivity_ci'][0]:.4f}, {results['sensitivity_ci'][1]:.4f})")
    print(f"特异性: {results['specificity_mean']:.4f} ({results['specificity_ci'][0]:.4f}, {results['specificity_ci'][1]:.4f})")
    print(f"F1 Score: {results['f1_mean']:.4f} ({results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f})")
    print(f"平均迭代次数: {results['mean_iterations']:.0f}")
    print(f"最佳Fold: {results['best_fold_info']['best_fold']}")
    
    return results, final_model, preprocessor


if __name__ == "__main__":
    results, final_model, preprocessor = main()
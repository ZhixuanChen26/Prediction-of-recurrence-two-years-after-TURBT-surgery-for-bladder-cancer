import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Microsoft YaHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


class FrequencyTrimmer(BaseEstimator, TransformerMixin):
    """频率截断器：处理高基数分类变量"""
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


def create_preprocessor(exclude_cols=None):
    """创建数据预处理器"""
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
        
        print(f"特征分类: 数值{len(numeric_features)}个, BMI{len(bmi_features)}个, 部位{len(site_features)}个, 字符串{len(string_features)}个")
        
        transformers = []
        
        # 数值特征：直接使用
        if numeric_features:
            transformers.append(('num', 'passthrough', numeric_features))
        
        # BMI特征：One-hot编码
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
        
        # 部位特征：One-hot编码
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
        
        # 字符串特征：频率截断 + One-hot编码
        if string_features:
            string_pipeline = Pipeline([
                ('freq_trim', FrequencyTrimmer(max_categories=15)),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat_str', string_pipeline, string_features))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        return preprocessor
    
    return build_preprocessor


class RFEXGBoostSelector:
    """RFE+XGBoost特征选择器"""
    
    def __init__(self, data_file='rf_input.csv', outcome_col='两年内是否复发', 
                 exclude_cols=None, random_state=42):
        """
        初始化特征选择器
        
        Parameters:
        -----------
        data_file : str
            数据文件路径
        outcome_col : str
            结果变量列名
        exclude_cols : list
            要排除的列名
        random_state : int
            随机种子
        """
        self.data_file = data_file
        self.outcome_col = outcome_col
        self.exclude_cols = exclude_cols or []
        self.random_state = random_state
        self.X = None
        self.y = None
        self.original_features = None
        self.processed_features = None
        self.preprocessor = None
        self.results = {}
        
        print("=== RFE+XGBoost特征选择器初始化 ===")
        print(f"数据文件: {data_file}")
        print(f"结果变量: {outcome_col}")
        if exclude_cols:
            print(f"排除特征: {exclude_cols}")
    
    def load_data(self):
        """加载和预处理数据"""
        print(f"\n正在加载数据: {self.data_file}")
        
        try:
            data = pd.read_csv(self.data_file, encoding='utf-8')
            
            # 自动识别特征列
            feature_cols = [col for col in data.columns 
                           if col != self.outcome_col and col not in self.exclude_cols]
            
            self.original_features = feature_cols
            self.X = data[feature_cols].copy()
            self.y = data[self.outcome_col].copy()
            
            # 数据清洗
            mask = ~(self.X.isnull().any(axis=1) | self.y.isnull())
            self.X = self.X.loc[mask]
            self.y = self.y.loc[mask]
            
            print(f"数据加载成功:")
            print(f"  样本量: {len(self.X)}")
            print(f"  原始特征数: {len(feature_cols)}")
            print(f"  事件率: {sum(self.y == 1) / len(self.y) * 100:.1f}%")
            
            # 创建预处理器
            build_preprocessor = create_preprocessor(self.exclude_cols)
            self.preprocessor = build_preprocessor(feature_cols, self.X)
            
            # 预处理数据
            self.preprocessor.fit(self.X, self.y)
            X_processed = self.preprocessor.transform(self.X)
            
            # 获取处理后的特征名
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                self.processed_features = self.preprocessor.get_feature_names_out().tolist()
            else:
                self.processed_features = [f'feature_{i}' for i in range(X_processed.shape[1])]
            
            print(f"  预处理后特征数: {len(self.processed_features)}")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False
    
    def baseline_performance(self):
        """评估所有特征的基线性能"""
        print(f"\n=== 基线性能评估（所有特征）===")
        
        X_processed = self.preprocessor.transform(self.X)
        
        # 基础XGBoost模型
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # 交叉验证评估
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(xgb_model, X_processed, self.y, cv=cv, scoring='roc_auc')
        
        baseline_auc = np.mean(cv_scores)
        baseline_std = np.std(cv_scores)
        
        print(f"基线AUC (所有{len(self.processed_features)}个特征): {baseline_auc:.4f} ± {baseline_std:.4f}")
        
        self.results['baseline'] = {
            'n_features': len(self.processed_features),
            'auc_mean': baseline_auc,
            'auc_std': baseline_std,
            'cv_scores': cv_scores,
            'features': self.processed_features.copy()
        }
        
        return baseline_auc
    
    def rfe_feature_selection(self, min_features=5, max_features=None, step=1):
        """
        使用RFE进行特征选择
        
        Parameters:
        -----------
        min_features : int
            最小特征数
        max_features : int
            最大特征数（None表示使用所有特征）
        step : int
            每次消除的特征数
        """
        print(f"\n=== RFE特征选择 ===")
        
        X_processed = self.preprocessor.transform(self.X)
        
        if max_features is None:
            max_features = len(self.processed_features)
        
        print(f"特征范围: {min_features} - {max_features}")
        print(f"每次消除: {step}个特征")
        
        # 基础XGBoost估计器
        estimator = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # RFECV：交叉验证自动选择最优特征数
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rfecv = RFECV(
            estimator=estimator,
            min_features_to_select=min_features,
            step=step,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        print(f"正在进行RFECV...")
        rfecv.fit(X_processed, self.y)
        
        # 获取选中的特征
        selected_features = [self.processed_features[i] for i in range(len(self.processed_features)) 
                           if rfecv.support_[i]]
        
        optimal_n_features = rfecv.n_features_
        optimal_auc = np.max(rfecv.cv_results_['mean_test_score'])
        
        print(f"最优特征数: {optimal_n_features}")
        print(f"最优AUC: {optimal_auc:.4f}")
        print(f"选中的特征:")
        for i, feature in enumerate(selected_features, 1):
            print(f"  {i:2d}. {feature}")
        
        self.results['rfecv'] = {
            'rfecv_object': rfecv,
            'n_features': optimal_n_features,
            'auc_mean': optimal_auc,
            'selected_features': selected_features,
            'feature_ranking': rfecv.ranking_,
            'cv_results': rfecv.cv_results_
        }
        
        return selected_features, optimal_n_features, optimal_auc
    
    def manual_rfe_analysis(self, feature_range=None, step=1):
        """
        手动RFE分析：测试不同特征数量的性能
        
        Parameters:
        -----------
        feature_range : list
            要测试的特征数量列表，如[5, 10, 15, 20]
        step : int
            步长
        """
        print(f"\n=== 手动RFE分析 ===")
        
        X_processed = self.preprocessor.transform(self.X)
        
        if feature_range is None:
            max_features = len(self.processed_features)
            feature_range = list(range(5, min(max_features + 1, 21), step))
        
        print(f"测试特征数量: {feature_range}")
        
        # 基础估计器
        estimator = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        results = []
        
        for n_features in feature_range:
            print(f"  测试 {n_features} 个特征...", end=' ')
            
            # RFE特征选择
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            rfe.fit(X_processed, self.y)
            
            # 选中的特征
            X_selected = X_processed[:, rfe.support_]
            selected_feature_names = [self.processed_features[i] for i in range(len(self.processed_features)) 
                                    if rfe.support_[i]]
            
            # 交叉验证评估
            cv_scores = cross_val_score(estimator, X_selected, self.y, cv=cv, scoring='roc_auc')
            
            mean_auc = np.mean(cv_scores)
            std_auc = np.std(cv_scores)
            
            print(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            
            results.append({
                'n_features': n_features,
                'auc_mean': mean_auc,
                'auc_std': std_auc,
                'cv_scores': cv_scores,
                'selected_features': selected_feature_names,
                'feature_ranking': rfe.ranking_
            })
        
        self.results['manual_rfe'] = results
        
        # 找到AUC最高的前3组
        sorted_results = sorted(results, key=lambda x: x['auc_mean'], reverse=True)
        top5_results = sorted_results[:5]
        
        print(f"\n=== AUC最高的前3组特征 ===")
        for rank, result in enumerate(top5_results, 1):
            print(f"\n第{rank}名: {result['n_features']}个特征, AUC={result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
            
            # 映射回原始特征名
            original_mapping = self.map_to_original_features(result['selected_features'])
            unique_features = []
            for processed_name, original_name in original_mapping.items():
                feature_name = original_name if original_name else processed_name
                if feature_name not in unique_features:
                    unique_features.append(feature_name)
            
            print(f"  特征列表:")
            for i, feature in enumerate(unique_features, 1):
                print(f"    {i:2d}. {feature}")
        
        return results
    
    def feature_importance_analysis(self, selected_features=None):
        """
        特征重要性分析（简化版）
        
        Parameters:
        -----------
        selected_features : list
            选中的特征名列表（如果为None，使用手动RFE最佳结果）
        """
        print(f"\n=== 特征重要性分析（最佳组合）===")
        
        # 使用手动RFE的最佳结果
        if 'manual_rfe' in self.results:
            manual_results = self.results['manual_rfe']
            best_result = max(manual_results, key=lambda x: x['auc_mean'])
            selected_features = best_result['selected_features']
        elif selected_features is None:
            print("没有可用的特征选择结果")
            return None
        
        X_processed = self.preprocessor.transform(self.X)
        
        # 获取选中特征的索引
        selected_indices = [i for i, name in enumerate(self.processed_features) 
                          if name in selected_features]
        X_selected = X_processed[:, selected_indices]
        
        # 训练XGBoost模型
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        xgb_model.fit(X_selected, self.y)
        
        # 获取特征重要性
        importance_gain = xgb_model.feature_importances_
        
        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            '特征名': selected_features,
            'Gain重要性': importance_gain
        }).sort_values('Gain重要性', ascending=False)
        
        print(f"最佳组合的特征重要性排序:")
        print("-" * 50)
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            # 映射回原始特征名
            original_name = self.map_to_original_features([row['特征名']])
            display_name = list(original_name.values())[0] if list(original_name.values())[0] else row['特征名']
            print(f"{i:2d}. {display_name:30} {row['Gain重要性']:8.4f}")
        
        self.results['feature_importance'] = importance_df
        
        return importance_df
    
    def plot_results(self, save_path=None):
        """绘制分析结果"""
        
        if 'manual_rfe' not in self.results:
            print("没有手动RFE结果可以绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 特征数量 vs AUC曲线
        manual_results = self.results['manual_rfe']
        n_features_list = [r['n_features'] for r in manual_results]
        auc_means = [r['auc_mean'] for r in manual_results]
        auc_stds = [r['auc_std'] for r in manual_results]
        
        ax1.errorbar(n_features_list, auc_means, yerr=auc_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2)
        ax1.set_xlabel('特征数量')
        ax1.set_ylabel('交叉验证AUC')
        ax1.set_title('RFE特征选择: 特征数量 vs 性能')
        ax1.grid(True, alpha=0.3)
        
        # 标记最佳点
        best_idx = np.argmax(auc_means)
        ax1.scatter(n_features_list[best_idx], auc_means[best_idx], 
                   color='red', s=100, zorder=5)
        ax1.annotate(f'最佳: {n_features_list[best_idx]}特征\nAUC={auc_means[best_idx]:.4f}',
                    xy=(n_features_list[best_idx], auc_means[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 添加基线
        if 'baseline' in self.results:
            baseline_auc = self.results['baseline']['auc_mean']
            ax1.axhline(y=baseline_auc, color='red', linestyle='--', alpha=0.7, 
                       label=f'基线(所有特征): {baseline_auc:.4f}')
            ax1.legend()
        
        # 2. 特征重要性柱状图
        if 'feature_importance' in self.results:
            importance_df = self.results['feature_importance'].head(15)
            
            y_pos = range(len(importance_df))
            ax2.barh(y_pos, importance_df['Gain重要性'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([name[:25] for name in importance_df['特征名']])
            ax2.set_xlabel('特征重要性 (Gain)')
            ax2.set_title('选中特征重要性排序')
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def save_results(self, filename='RFE_XGBoost特征选择结果.xlsx'):
        """保存分析结果到Excel"""
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # 1. 基线结果
            if 'baseline' in self.results:
                baseline_df = pd.DataFrame([{
                    '特征数': self.results['baseline']['n_features'],
                    'AUC均值': self.results['baseline']['auc_mean'],
                    'AUC标准差': self.results['baseline']['auc_std']
                }])
                baseline_df.to_excel(writer, sheet_name='基线性能', index=False)
            
            # 2. 手动RFE结果
            if 'manual_rfe' in self.results:
                manual_df = pd.DataFrame([{
                    '特征数': r['n_features'],
                    'AUC均值': r['auc_mean'],
                    'AUC标准差': r['auc_std']
                } for r in self.results['manual_rfe']])
                manual_df.to_excel(writer, sheet_name='RFE性能曲线', index=False)
            
            # 3. RFECV最优结果
            if 'rfecv' in self.results:
                rfecv_result = self.results['rfecv']
                rfecv_df = pd.DataFrame([{
                    '最优特征数': rfecv_result['n_features'],
                    'AUC': rfecv_result['auc_mean']
                }])
                rfecv_df.to_excel(writer, sheet_name='RFECV最优结果', index=False)
                
                # 选中的特征
                selected_df = pd.DataFrame({
                    '排名': range(1, len(rfecv_result['selected_features']) + 1),
                    '特征名': rfecv_result['selected_features']
                })
                selected_df.to_excel(writer, sheet_name='RFECV选中特征', index=False)
            
            # 4. 特征重要性
            if 'feature_importance' in self.results:
                importance_df = self.results['feature_importance']
                importance_df.to_excel(writer, sheet_name='特征重要性', index=False)
            
            # 5. 原始特征列表
            original_df = pd.DataFrame({
                '排名': range(1, len(self.original_features) + 1),
                '原始特征名': self.original_features
            })
            original_df.to_excel(writer, sheet_name='原始特征列表', index=False)
        
        print(f"结果已保存到: {filename}")
    
    def get_final_features(self):
        """获取最终推荐的特征列表（简化版）"""
        
        # 使用手动RFE的最佳结果
        if 'manual_rfe' in self.results:
            manual_results = self.results['manual_rfe']
            best_result = max(manual_results, key=lambda x: x['auc_mean'])
            
            selected_features = best_result['selected_features']
            n_features = best_result['n_features']
            auc = best_result['auc_mean']
            
            print(f"\n=== 最终推荐特征 ===")
            print(f"特征数量: {n_features}, 预期AUC: {auc:.4f}")
            
            # 映射回原始特征名并去重
            original_mapping = self.map_to_original_features(selected_features)
            unique_features = []
            for processed_name, original_name in original_mapping.items():
                feature_name = original_name if original_name else processed_name
                if feature_name not in unique_features:
                    unique_features.append(feature_name)
            
            for i, feature in enumerate(unique_features, 1):
                print(f"  {i:2d}. {feature}")
            
            return unique_features
        
        else:
            print("没有可用的特征选择结果")
            return None
    
    def map_to_original_features(self, processed_features):
        """将处理后的特征名映射回原始特征名"""
        mapping = {}
        
        for processed_name in processed_features:
            # 查找对应的原始特征
            original_name = None
            
            # 数值特征：直接匹配
            if processed_name in self.original_features:
                original_name = processed_name
            
            # One-hot编码特征：提取前缀
            else:
                for orig_feature in self.original_features:
                    if processed_name.startswith(f"{orig_feature}_"):
                        original_name = orig_feature
                        break
                
                # 如果还没找到，可能是pipeline产生的
                if original_name is None:
                    parts = processed_name.split('__')
                    if len(parts) > 1:
                        feature_part = parts[-1]
                        for orig_feature in self.original_features:
                            if feature_part.startswith(f"{orig_feature}_"):
                                original_name = orig_feature
                                break
            
            mapping[processed_name] = original_name
        
        return mapping
    
    def run_complete_analysis(self, min_features=5, max_features=20, step=1):
        """运行完整的特征选择分析（简化版）"""
        
        print("🚀 开始RFE+XGBoost特征选择分析")
        
        # 1. 加载数据
        if not self.load_data():
            print("❌ 数据加载失败，分析终止")
            return None
        
        # 2. 基线性能（简化输出）
        baseline_auc = self.baseline_performance()
        
        # 3. 跳过RFECV，直接进行手动RFE分析
        manual_results = self.manual_rfe_analysis(
            feature_range=list(range(min_features, min(max_features + 1, len(self.processed_features) + 1), step))
        )
        
        # 4. 特征重要性分析（仅分析最佳组合）
        importance_df = self.feature_importance_analysis()
        
        # 5. 保存结果（减少文件输出）
        self.save_results()
        
        # 6. 最终推荐（简化版）
        final_features = self.get_final_features()
        
        # 7. 总结（简化版）
        if 'manual_rfe' in self.results:
            best_manual = max(self.results['manual_rfe'], key=lambda x: x['auc_mean'])
            print(f"\n🎉 特征选择完成！")
            print(f"✅ 最佳特征数: {best_manual['n_features']}")
            print(f"✅ 基线AUC: {baseline_auc:.4f} → 优化后AUC: {best_manual['auc_mean']:.4f}")
            print(f"✅ 性能提升: {best_manual['auc_mean'] - baseline_auc:+.4f}")
        
        return final_features


def main():
    """主函数"""
    
    # 创建特征选择器
    selector = RFEXGBoostSelector(
        data_file='rf_input.csv',
        outcome_col='两年内是否复发',
        exclude_cols=None,  # 可以排除某些特征
        random_state=42
    )
    
    # 运行完整分析
    final_features = selector.run_complete_analysis(
        min_features=5,      # 最少保留5个特征
        max_features=20,     # 最多测试20个特征
        step=1               # 每次减少1个特征
    )
    
    return selector, final_features


if __name__ == "__main__":
    # 运行特征选择
    selector, final_features = main()
    
    print(f"\n💡 建议:")
    print(f"  - 使用前3组特征分别训练模型并对比性能")
    print(f"  - 推荐优先使用第1名的特征组合")


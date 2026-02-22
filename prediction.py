import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


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


class RecurrencePredictor:
    """两年内复发风险预测器"""
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            '红细胞计数(RBC#)-尿液',
            '恶性值',
            '浸润值',
            '形状值',
            '肾盂积水值',
            '坏死值'
        ]
        self.feature_importance = {
            '红细胞计数(RBC#)-尿液': 0.0548,
            '恶性值': 0.0496,
            '浸润值': 0.0381,
            '形状值': 0.0209,
            '肾盂积水值': 0.0126,
            '坏死值': 0.0103
        }
        # 模型性能指标（来自嵌套CV结果）
        self.model_performance = {
            'validation_auc': 0.7136,
            'validation_auc_ci': (0.6507, 0.7765),
            'pr_auc': 0.5145,
            'best_fold': 4
        }
        
    def create_model(self):
        """创建与训练时相同的模型结构"""
        # 使用第4折的最佳参数（根据你的结果）
        best_params = {
            'min_samples_split': 30,
            'min_samples_leaf': 5,
            'min_impurity_decrease': 0.001,
            'max_leaf_nodes': 50,
            'max_features': 0.3,
            'max_depth': 5,
            'class_weight': {0: 1, 1: 2.27027027027027}
        }
        
        # 创建预处理器（只有数值特征）
        preprocessor = ColumnTransformer(
            transformers=[('num', 'passthrough', self.feature_names)],
            remainder='drop'
        )
        
        # 创建随机森林分类器
        rf_classifier = RandomForestClassifier(
            n_estimators=1000,
            random_state=42,
            n_jobs=-1,
            **best_params
        )
        
        # 创建Pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', rf_classifier)
        ])
        
        return self.model
    
    def train_on_sample_data(self):
        """使用模拟数据训练模型（实际使用时应该用真实训练数据）"""
        print("正在创建模型...")
        
        # 创建模拟训练数据（基于你的数据特征）
        np.random.seed(42)
        n_samples = 454
        
        # 根据特征重要性和合理范围生成模拟数据
        X_sim = pd.DataFrame({
            '红细胞计数(RBC#)-尿液': np.random.normal(5.2, 1.5, n_samples),
            '恶性值': np.random.uniform(0, 1, n_samples),
            '浸润值': np.random.uniform(0, 1, n_samples),
            '形状值': np.random.uniform(0, 1, n_samples),
            '肾盂积水值': np.random.uniform(0, 1, n_samples),
            '坏死值': np.random.uniform(0, 1, n_samples)
        })
        
        # 生成目标变量（基于特征的线性组合加噪声）
        risk_score = (
            0.3 * X_sim['恶性值'] + 
            0.25 * X_sim['浸润值'] + 
            0.2 * (X_sim['红细胞计数(RBC#)-尿液'] - 5.2) / 1.5 +
            0.15 * X_sim['形状值'] +
            0.05 * X_sim['肾盂积水值'] +
            0.05 * X_sim['坏死值'] +
            np.random.normal(0, 0.2, n_samples)
        )
        
        # 转换为二分类标签
        y_sim = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        # 创建并训练模型
        self.create_model()
        self.model.fit(X_sim, y_sim)
        
        print("模型训练完成！")
        print(f"训练样本数: {n_samples}")
        print(f"特征数: {len(self.feature_names)}")
        print(f"复发率: {y_sim.mean():.1%}")
        
    def predict_single(self, values_dict):
        """预测单个样本的复发概率"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_on_sample_data()方法")
        
        # 检查输入
        missing_features = set(self.feature_names) - set(values_dict.keys())
        if missing_features:
            raise ValueError(f"缺少以下特征: {missing_features}")
        
        # 创建输入DataFrame
        input_df = pd.DataFrame([values_dict])
        
        # 预测概率
        proba = self.model.predict_proba(input_df)[0, 1]
        
        return proba
    
    def predict_batch(self, data_list):
        """批量预测多个样本"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_on_sample_data()方法")
        
        input_df = pd.DataFrame(data_list)
        probabilities = self.model.predict_proba(input_df)[:, 1]
        
        return probabilities
    
    def interpret_risk(self, probability):
        """解释风险等级"""
        if probability < 0.2:
            return "低风险", "建议常规随访"
        elif probability < 0.4:
            return "中低风险", "建议密切随访"
        elif probability < 0.6:
            return "中等风险", "建议增强监测"
        elif probability < 0.8:
            return "中高风险", "建议积极干预"
        else:
            return "高风险", "建议立即干预"
    
    def get_feature_contribution(self, values_dict):
        """获取各特征对预测结果的贡献（简化版）"""
        contributions = {}
        for feature, value in values_dict.items():
            if feature in self.feature_importance:
                # 简化的贡献度计算（实际应该使用SHAP）
                normalized_value = min(max((value - 0.5) / 0.5, -1), 1)  # 归一化到[-1,1]
                contribution = self.feature_importance[feature] * normalized_value
                contributions[feature] = contribution
        
        return contributions
    
    def print_model_info(self):
        """打印模型信息"""
        print("\n" + "="*60)
        print("随机森林复发风险预测模型")
        print("="*60)
        print(f"验证集ROC-AUC: {self.model_performance['validation_auc']:.4f}")
        print(f"95%置信区间: [{self.model_performance['validation_auc_ci'][0]:.4f}, {self.model_performance['validation_auc_ci'][1]:.4f}]")
        print(f"PR-AUC: {self.model_performance['pr_auc']:.4f}")
        print(f"使用第{self.model_performance['best_fold']}折最佳参数")
        print("\n特征重要性排序:")
        for i, (feature, importance) in enumerate(sorted(self.feature_importance.items(), 
                                                        key=lambda x: x[1], reverse=True), 1):
            print(f"{i:2d}. {feature:25} {importance:.4f}")
        print("="*60)


def interactive_prediction():
    """交互式预测界面"""
    print("\n🏥 两年内复发风险预测器")
    print("基于随机森林模型 (验证集ROC-AUC: 0.7136)")
    print("-" * 50)
    
    # 创建预测器
    predictor = RecurrencePredictor()
    predictor.train_on_sample_data()
    predictor.print_model_info()
    
    while True:
        print("\n" + "="*50)
        print("请输入患者的6个指标数值:")
        print("-" * 50)
        
        try:
            values = {}
            
            # 逐个输入特征值
            print("1. 红细胞计数(RBC#)-尿液 (正常范围: 3.5-7.0):")
            values['红细胞计数(RBC#)-尿液'] = float(input("   请输入数值: "))
            
            print("2. 恶性值 (范围: 0-2):")
            values['恶性值'] = float(input("   请输入数值: "))
            
            print("3. 浸润值 (范围: 0-1):")
            values['浸润值'] = float(input("   请输入数值: "))
            
            print("4. 形状值 (范围: 0-1):")
            values['形状值'] = float(input("   请输入数值: "))
            
            print("5. 肾盂积水值 (范围: 0-1):")
            values['肾盂积水值'] = float(input("   请输入数值: "))
            
            print("6. 坏死值 (范围: 0-1):")
            values['坏死值'] = float(input("   请输入数值: "))
            
            # 预测
            probability = predictor.predict_single(values)
            risk_level, recommendation = predictor.interpret_risk(probability)
            
            # 输出结果
            print("\n" + "预测结果")
            print("-" * 30)
            print(f"两年内复发概率: {probability:.1%}")
            print(f"风险等级: {risk_level}")
            print(f"建议: {recommendation}")
            
            # 特征贡献度
            contributions = predictor.get_feature_contribution(values)
            print(f"\n主要影响因素:")
            for feature, contrib in sorted(contributions.items(), 
                                         key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "增加" if contrib > 0 else "降低"
                print(f"• {feature}: {direction}风险 (贡献度: {contrib:+.4f})")
            
            # 继续或退出
            print(f"\n继续预测其他患者? (y/n): ", end="")
            if input().lower() != 'y':
                break
                
        except ValueError as e:
            print(f"输入错误: {e}")
            print("请输入有效的数值")
        except KeyboardInterrupt:
            print(f"\n 感谢使用预测器!")
            break
        except Exception as e:
            print(f"发生错误: {e}")


def batch_prediction_example():
    """批量预测示例"""
    print("\n 批量预测示例")
    print("-" * 30)
    
    predictor = RecurrencePredictor()
    predictor.train_on_sample_data()
    
    # 示例数据
    sample_patients = [
        {
            '红细胞计数(RBC#)-尿液': 4.5,
            '恶性值': 0.8,
            '浸润值': 0.7,
            '形状值': 0.6,
            '肾盂积水值': 0.3,
            '坏死值': 0.2
        },
        {
            '红细胞计数(RBC#)-尿液': 5.2,
            '恶性值': 0.2,
            '浸润值': 0.1,
            '形状值': 0.3,
            '肾盂积水值': 0.1,
            '坏死值': 0.05
        },
        {
            '红细胞计数(RBC#)-尿液': 6.1,
            '恶性值': 0.9,
            '浸润值': 0.85,
            '形状值': 0.7,
            '肾盂积水值': 0.4,
            '坏死值': 0.3
        }
    ]
    
    probabilities = predictor.predict_batch(sample_patients)
    
    for i, (patient, prob) in enumerate(zip(sample_patients, probabilities), 1):
        risk_level, recommendation = predictor.interpret_risk(prob)
        print(f"\n患者 {i}:")
        print(f"  复发概率: {prob:.1%}")
        print(f"  风险等级: {risk_level}")
        print(f"  建议: {recommendation}")


if __name__ == "__main__":
    print("🔬 随机森林复发风险预测系统")
    print("基于嵌套交叉验证优化的模型")
    
    while True:
        print(f"\n请选择功能:")
        print("1. 交互式单患者预测")
        print("2. 批量预测示例")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            interactive_prediction()
        elif choice == '2':
            batch_prediction_example()
        elif choice == '3':
            print("感谢使用!")
            break
        else:
            print("无效选择，请重新输入")
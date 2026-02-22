import pandas as pd
import miceforest as mf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def miceforest_imputation_rubins(data_path, output_path='454人插补后总表_两年.csv', 
                                num_datasets=5, save_all_datasets=False):
    print("开始MICE插补 (Rubin's Rules)...")
    data = pd.read_csv(data_path)
    print(f"原始数据: {data.shape}")
    
    # 删除ID列和高缺失列
    outcome_col = '两年内是否复发'
    id_cols = ['就诊标识（医渡云计算）', '病案号', '就诊次', '统一手术名称',
               '恶性程度', '形态', '细胞学来源', '是否浸润', '手术开始时间',
               '复发时间','复发间隔日', '来源值','一年内二次手术','两年内二次手术',
               '二次手术开始时间', '两次手术间隔天数', '一年内是否复发','是否多次手术', 
               '手术次数','肿瘤部位（如三角区、侧壁、前壁等）','数目','形态（菜花状、乳头状或丘状病变）',
               '钙化（有或无）','囊性坏死（有或无）','边界（清晰或不清晰）','蒂（有或无）','t分期',
               '肾盂是否积水（无、单、双侧）','备注','体温']
    
    missing_rates = data.isna().mean()
    high_missing_cols = missing_rates[missing_rates > 0.7].index.tolist()
    cols_to_drop = [c for c in id_cols + high_missing_cols 
                   if c in data.columns and c != outcome_col]
    clean_data = data.drop(columns=cols_to_drop)
    print(f"保留{clean_data.shape[1]}个变量")
    
    # 处理分类变量
    obj_cols = clean_data.select_dtypes(include=['object']).columns
    for c in obj_cols:
        clean_data[c] = clean_data[c].astype('category')
    
    # 核心改进：使用最佳实践参数
    print(f"生成{num_datasets}个插补数据集...")
    mice_imputer = mf.ImputationKernel(
        data=clean_data,
        num_datasets=num_datasets,
        mean_match_candidates=5,  # 关键改进1：避免极端值
        save_all_iterations_data=True,
        random_state=42
    )
    mice_imputer.mice(iterations=5, verbose=False)
    
    # 处理所有插补数据集
    all_datasets = []
    for i in range(num_datasets):
        imputed_temp = mice_imputer.complete_data(dataset=i)
        
        # 编码分类变量
        mapping = {'性别': {'男': 1, '女': 0}, '是否吸烟': {'是': 1, '否': 0}, '是否饮酒': {'是': 1, '否': 0}}
        for col, mp in mapping.items():
            if col in imputed_temp.columns:
                imputed_temp[col] = imputed_temp[col].map(mp).astype('Int8')
        
        # 添加病案号
        imputed_temp['病案号'] = data['病案号']
        all_datasets.append(imputed_temp)
        
        # 可选：保存单个数据集
        if save_all_datasets:
            imputed_temp.to_csv(f"{output_path.replace('.csv', f'_dataset_{i}.csv')}", 
                              index=False, encoding='utf-8-sig')
    
    # 使用第一个数据集作为默认输出
    default_dataset = all_datasets[0]
    default_dataset.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"插补完成! 保存到: {output_path}")
    print(f"最终数据: {default_dataset.shape}")
    
    return all_datasets, default_dataset

def evaluate_with_rubins_rules(all_datasets, target_col, drop_cols=None, cv_folds=5):
    """
    使用Rubin's Rules评估，避免数据泄漏
    """
    drop_cols = drop_cols or []
    all_aucs = []
    
    print(f"\n使用Rubin's Rules评估{len(all_datasets)}个插补数据集...")
    
    for i, dataset in enumerate(all_datasets):
        X = dataset.drop(columns=drop_cols + [target_col])
        y = dataset[target_col]
        
        pipe = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        )
        
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
        aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        
        all_aucs.append(aucs)
        print(f"数据集{i}: AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")
    
    # Rubin's Rules计算
    all_aucs = np.array(all_aucs)
    dataset_means = all_aucs.mean(axis=1)
    pooled_mean = dataset_means.mean()
    
    # 计算方差组分
    within_var = all_aucs.var(axis=1, ddof=1).mean()
    between_var = dataset_means.var(ddof=1)
    
    # 总方差 (Rubin's Rules公式)
    m = len(all_datasets)
    total_var = within_var + (1 + 1/m) * between_var
    pooled_std = np.sqrt(total_var)
    
    print(f"\n=== Rubin's Rules 结果 ===")
    print(f"合并估计AUC: {pooled_mean:.4f} ± {pooled_std:.4f}")
    print(f"数据集内方差: {within_var:.6f}")
    print(f"数据集间方差: {between_var:.6f}")
    
    return pooled_mean, pooled_std, dataset_means

if __name__ == "__main__":
    # 核心改进3：增加数据集数量到10个
    all_datasets, default_dataset = miceforest_imputation_rubins(
        '454人总表试.csv', 
        '454人插补后总表_两年_试.csv',
        num_datasets=5, 
        save_all_datasets=True
    )
    
    # 使用改进的评估方法
    pooled_auc, pooled_std, individual_aucs = evaluate_with_rubins_rules(
        all_datasets,
        target_col="两年内是否复发",
        drop_cols=["病案号"]
    )
    
    print(f"\n最终结果:")
    print(f"AUC: {pooled_auc:.4f} ± {pooled_std:.4f}")
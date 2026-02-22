import pandas as pd

def extract_selected_features():
    """
    从原始数据中提取筛选出的特征和医生补充的特征
    """
    
    # 读取原始数据
    data = pd.read_csv('454人插补后总表_两年.csv')
    
    # 筛选出的21个特征
    selected_features = [
        '恶性值',
        #'红细胞体积分布宽度CV(RDW-CV)-静脉血',
        '乙型肝炎病毒表面抗原(HBV-sAg)-未知',
        '单核细胞百分比(Mono%)-静脉血',
        #'低密度脂蛋白胆固醇(LDL-C)-静脉血',
        '是否吸烟',
        '丙氨酸氨基转移酶(ALT)-静脉血',
        '肾盂积水值',
        '部位值',
        't值',
        '甘油三酯(TG)-静脉血',
        '边界清晰值',
        '平均红细胞血红蛋白含量(MCH)-静脉血',
        '细菌计数(BACT#)-尿液',
        '尿酸(UA)-未知',
        '坏死值',
        #'乙型肝炎病毒e抗体(HBV-eAb)-未知',
        '总钙(Ca)-静脉血',
        '浸润值',
        '钙化值',
        '胱抑素(Cys-C)-静脉血',
        '形状值',
        '淋巴细胞计数(Lymph#)-静脉血',
        #'碱性磷酸酶(ALP)-静脉血',
        #'白蛋白/球蛋白比值(ALB/GLO)-静脉血',
        #'高血压', 
        #'红细胞比容(Hct)-静脉血',
        #'淋巴细胞百分比(Lymph%)-静脉血',
        '大小（mm）',
        '平扫时病变ct值',
        '天门冬氨酸氨基转移酶(AST)-静脉血',
        '同型半胱氨酸(HCY)-静脉血',
        #'单淋比',
        '中性粒细胞百分比(Neut%)-静脉血',
        '板淋比',
        '红细胞计数(RBC#)-尿液', 
        #'年龄',   
        '单核细胞计数(Mono#)-静脉血',  
        '平均红细胞血红蛋白含量(MCH)-静脉血',    
        '球蛋白(GLO)-未知',
        '嗜酸性粒细胞百分比(Eos%)-静脉血',
    ]

            
    # 医生建议补充的特征
    doctor_suggested = [
    ]
    
    # 结局变量
    outcome = '两年内是否复发'
    
    # 合并所有需要的列
    all_features = selected_features + doctor_suggested + [outcome]
    
    print(f"总共提取 {len(all_features)} 列:")
    print(f"  - 筛选特征: {len(selected_features)} 个")
    print(f"  - 医生建议: {len(doctor_suggested)} 个") 
    print(f"  - 结局变量: 1 个")
    
    # 检查哪些列在原数据中存在
    existing_features = []
    missing_features = []
    
    for feature in all_features:
        if feature in data.columns:
            existing_features.append(feature)
        else:
            missing_features.append(feature)
    
    if missing_features:
        print(f"\n警告: 以下 {len(missing_features)} 个特征在原数据中未找到:")
        for feature in missing_features:
            print(f"  - {feature}")
        
        # 显示可能的匹配
        print(f"\n原数据中的相似列名:")
        for missing in missing_features:
            similar_cols = [col for col in data.columns if missing.split('(')[0].strip() in col]
            if similar_cols:
                print(f"  {missing} -> 可能匹配: {similar_cols}")
    
    # 提取存在的特征
    extracted_data = data[existing_features].copy()
    
    print(f"\n成功提取 {len(existing_features)} 列，数据形状: {extracted_data.shape}")
    
    # 保存提取的数据
    output_file = 'lasso筛选用数据.csv'
    extracted_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"数据已保存到: {output_file}")
    
    # 显示基本信息
    print(f"\n数据基本信息:")
    print(f"  样本数: {len(extracted_data)}")
    print(f"  特征数: {len(extracted_data.columns) - 1}")  # 减去结局变量
    print(f"  结局变量 '{outcome}' 分布:")
    if outcome in extracted_data.columns:
        print(f"    {extracted_data[outcome].value_counts().to_dict()}")
    
    # 检查缺失值
    missing_info = extracted_data.isnull().sum()
    if missing_info.sum() > 0:
        print(f"\n缺失值情况:")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count} ({missing_count/len(extracted_data)*100:.1f}%)")
    else:
        print(f"\n无缺失值")
    
    return extracted_data, existing_features, missing_features


if __name__ == "__main__":
    # 执行特征提取
    extracted_data, existing_features, missing_features = extract_selected_features()
    
    print(f"\n=== 提取完成 ===")
    print(f"可以使用 'lasso筛选用数据.csv' 进行下一步的LASSO筛选")
    
    # 显示提取的特征列表
    print(f"\n提取的特征列表:")
    for i, feature in enumerate(existing_features, 1):
        if feature != '两年内是否复发':  # 不显示结局变量
            print(f"{i:2d}. {feature}")
    
    if missing_features:
        print(f"\n需要手动检查的缺失特征:")
        for feature in missing_features:
            print(f"  - {feature}")
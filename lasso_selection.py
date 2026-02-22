import pandas as pd

def main():
    # 输入输出文件路径
    input_file = "lasso筛选用数据.csv"
    output_file = "rf_input.csv"
    
    # 要提取的特征列
    feature_cols = [
        "恶性值",
        #"细菌计数(BACT#)-尿液",
        "坏死值",
        #"部位值",
        #"钙化值",
        "肾盂积水值",
        #"是否吸烟",
        #"同型半胱氨酸(HCY)-静脉血",
        #"中淋比",
        #"边界清晰值",
        #"板淋比",
        #"肾盂造影ct值",
        #"嗜酸性粒细胞百分比(Eos%)-静脉血",
        #"胱抑素(Cys-C)-静脉血",
        #"t值",
        #"大小（mm）",
        #"年龄",
        #"丙氨酸氨基转移酶(ALT)-静脉血",
        #"天门冬氨酸氨基转移酶(AST)-静脉血",
        #"中性粒细胞百分比(Neut%)-静脉血",
        "浸润值",
        "形状值",
        #"数目值",
        #"单核细胞计数(Mono#)-静脉血",
        #'平均红细胞血红蛋白含量(MCH)-静脉血',
        '红细胞计数(RBC#)-尿液',
    ]
    
    outcome_col = "两年内是否复发"
    
    # 读取数据
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # 检查列是否存在
    missing = set(feature_cols + [outcome_col]) - set(df.columns)
    if missing:
        raise KeyError(f"以下列在输入文件中未找到：{missing}")
    
    # 提取并保存
    df_subset = df[feature_cols + [outcome_col]]
    df_subset.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已将 {len(feature_cols) + 1} 列导出到 {output_file}")

if __name__ == "__main__":
    main()

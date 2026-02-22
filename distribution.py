import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('454人插补后总表_两年.csv', encoding='utf-8')

#target_variable = '是否吸烟'      
#target_variable = '大小（mm）'    
target_variable = '中淋比'  
#target_variable = '恶性值' 
#target_variable = '细菌计数(BACT#)-尿液'   
#target_variable = '坏死值'   
#target_variable = '部位值'
#target_variable = '钙化值'     
#target_variable = '肾盂积水值'   
#target_variable = '同型半胱氨酸(HCY)-静脉血'   
#target_variable = '边界清晰值'  
#target_variable = '板淋比' 
#target_variable = '平均红细胞血红蛋白含量(MCH)-静脉血'  
#target_variable = '嗜酸性粒细胞百分比(Eos%)-静脉血'  
#target_variable = '丙氨酸氨基转移酶(ALT)-静脉血'  
#target_variable = '胱抑素(Cys-C)-静脉血'  
#target_variable = 't值'  
#target_variable = '天门冬氨酸氨基转移酶(AST)-静脉血'  
#target_variable = '浸润值'  
#target_variable = '年龄'  
#target_variable = '形状值' 
#target_variable = '数目值' 
#target_variable = '单核细胞计数(Mono#)-静脉血' 
#target_variable = '高血压' 

outcome_col = '两年内是否复发'

def quick_analysis(df, var_name, outcome_col):
    
    df_clean = df.dropna(subset=[var_name, outcome_col]).copy()
    df_clean['复发标签'] = df_clean[outcome_col].map({0: '未复发', 1: '复发'})

    unique_values = df_clean[var_name].nunique()
    is_categorical = unique_values <= 10 or df_clean[var_name].dtype == 'object'
    
    print(f"分析变量: {var_name}")
    print(f"有效数据: {len(df_clean)} 例")
    print(f"变量类型: {'分类变量' if is_categorical else '连续变量'}")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{var_name} 与复发关系分析', fontsize=14, fontweight='bold')
    
    if is_categorical:
        # 分类变量分析
        
        # 1. 总体分布
        var_counts = df_clean[var_name].value_counts()
        axes[0, 0].bar(range(len(var_counts)), var_counts.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('总体分布')
        axes[0, 0].set_xticks(range(len(var_counts)))
        axes[0, 0].set_xticklabels(var_counts.index, rotation=45)
        
        # 2. 分组分布
        crosstab = pd.crosstab(df_clean[var_name], df_clean['复发标签'])
        crosstab.plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'lightcoral'])
        axes[0, 1].set_title('分组分布')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 复发率
        recur_rate = df_clean.groupby(var_name)[outcome_col].mean() * 100
        axes[1, 0].bar(range(len(recur_rate)), recur_rate.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('复发率 (%)')
        axes[1, 0].set_xticks(range(len(recur_rate)))
        axes[1, 0].set_xticklabels(recur_rate.index, rotation=45)
        
        # 添加数值标签
        for i, v in enumerate(recur_rate.values):
            axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 4. 箱线图（如果是有序分类）
        try:
            sns.boxplot(data=df_clean, x='复发标签', y=var_name, ax=axes[1, 1])
            axes[1, 1].set_title('箱线图比较')
        except:
            axes[1, 1].text(0.5, 0.5, '无法绘制箱线图\n（非数值型分类变量）', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('箱线图比较')
        
        # 统计检验
        chi2, p_val, _, _ = chi2_contingency(crosstab)
        print(f"卡方检验 p值: {p_val:.4f}")
        print(f"结论: {'有显著关联' if p_val < 0.05 else '无显著关联'}")
        
        # 复发率统计
        print("\n复发率统计:")
        for category in recur_rate.index:
            rate = recur_rate[category]
            count = df_clean[df_clean[var_name] == category].shape[0]
            recur_count = df_clean[(df_clean[var_name] == category) & (df_clean[outcome_col] == 1)].shape[0]
            print(f"{category}: {recur_count}/{count} ({rate:.1f}%)")
    
    else:
        # 连续变量分析
        non_recur = df_clean[df_clean[outcome_col] == 0][var_name]
        recur = df_clean[df_clean[outcome_col] == 1][var_name]
        
        # 1. 总体分布
        axes[0, 0].hist(df_clean[var_name], bins=20, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('总体分布')
        
        # 2. 分组分布
        axes[0, 1].hist([non_recur, recur], bins=15, alpha=0.7, 
                       color=['lightblue', 'lightcoral'], label=['未复发', '复发'])
        axes[0, 1].set_title('分组分布')
        axes[0, 1].legend()
        
        # 3. 箱线图
        sns.boxplot(data=df_clean, x='复发标签', y=var_name, ax=axes[1, 0])
        axes[1, 0].set_title('箱线图比较')
        
        # 4. 散点图
        np.random.seed(42)
        for recur_status in [0, 1]:
            data_subset = df_clean[df_clean[outcome_col] == recur_status]
            x_vals = data_subset[var_name]
            y_vals = np.array([recur_status] * len(data_subset)) + np.random.normal(0, 0.05, len(data_subset))
            color = 'lightcoral' if recur_status == 1 else 'lightblue'
            label = '复发' if recur_status == 1 else '未复发'
            axes[1, 1].scatter(x_vals, y_vals, alpha=0.6, color=color, label=label, s=30)
        
        axes[1, 1].set_title('数据点分布')
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(['未复发', '复发'])
        axes[1, 1].legend()
        axes[1, 1].set_ylim(-0.3, 1.3)
        
        # 统计检验
        _, t_p = ttest_ind(non_recur, recur)
        _, u_p = mannwhitneyu(non_recur, recur, alternative='two-sided')
        
        print(f"t检验 p值: {t_p:.4f}")
        print(f"U检验 p值: {u_p:.4f}")
        print(f"结论: {'有显著差异' if min(t_p, u_p) < 0.05 else '无显著差异'}")
        
        print(f"\n描述统计:")
        print(f"未复发组: 均值={non_recur.mean():.2f}, 中位数={non_recur.median():.2f}")
        print(f"复发组: 均值={recur.mean():.2f}, 中位数={recur.median():.2f}")
        print(f"差异: 均值差={recur.mean()-non_recur.mean():.2f}, 中位数差={recur.median()-non_recur.median():.2f}")
    
    plt.tight_layout()
    plt.show()

quick_analysis(df, target_variable, outcome_col)
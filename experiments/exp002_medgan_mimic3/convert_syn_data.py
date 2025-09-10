#!/usr/bin/env python3
"""
将合成数据转换为pandas DataFrame，列名为疾病代码
"""

import pickle
import numpy as np
import pandas as pd

def convert_syn_data_to_dataframe():
    """
    将合成数据转换为pandas DataFrame
    """
    print("正在加载数据...")
    
    # 加载合成数据
    syn_data = np.load('syn/medgan_mimic3_1.npy')
    print(f"合成数据形状: {syn_data.shape}")
    print(f"数据类型: {syn_data.dtype}")
    
    # 加载疾病代码映射
    with open('processed_mimic.types', 'rb') as f:
        disease_codes_dict = pickle.load(f)
    
    print(f"疾病代码数量: {len(disease_codes_dict)}")
    
    # 验证数据维度
    if syn_data.shape[1] != len(disease_codes_dict):
        raise ValueError(f"数据维度不匹配: 合成数据列数 {syn_data.shape[1]}, 疾病代码数 {len(disease_codes_dict)}")
    
    # 创建列名列表（按索引顺序）
    column_names = [None] * len(disease_codes_dict)
    for disease_code, index in disease_codes_dict.items():
        column_names[index] = disease_code
    
    print("前10个列名:", column_names[:10])
    print("后10个列名:", column_names[-10:])
    
    # 创建DataFrame
    print("正在创建DataFrame...")
    df = pd.DataFrame(syn_data, columns=column_names)
    
    print(f"DataFrame形状: {df.shape}")
    print(f"DataFrame列数: {len(df.columns)}")
    print(f"DataFrame行数: {len(df)}")
    
    # 显示一些统计信息
    print("\n数据统计信息:")
    print(f"非零值数量: {(df != 0).sum().sum()}")
    print(f"零值数量: {(df == 0).sum().sum()}")
    print(f"数据范围: {df.min().min()} 到 {df.max().max()}")
    
    # 显示前几行和前几列
    print("\n前5行前10列:")
    print(df.iloc[:5, :10])
    
    # 保存DataFrame
    output_file = 'syn_data_mimic3_dataframe.pkl'
    print(f"\n正在保存DataFrame到 {output_file}...")
    df.to_pickle(output_file)
    print("保存完成！")
    
    # 也可以保存为CSV（可选，但文件会很大）
    # print("正在保存为CSV...")
    # df.to_csv('syn_data_mimic3_dataframe.csv', index=False)
    # print("CSV保存完成！")
    
    return df

if __name__ == "__main__":
    df = convert_syn_data_to_dataframe()
    print("\n转换完成！")
    print(f"DataFrame已保存，形状: {df.shape}")

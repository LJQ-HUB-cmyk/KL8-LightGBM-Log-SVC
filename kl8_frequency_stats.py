# -*- coding: utf-8 -*-
"""
快乐8历史号码频率统计分析脚本
统计所有号码的历史出现频率并按降序排列
"""

import pandas as pd
from collections import Counter

def load_and_analyze_frequency():
    """加载数据并分析每个号码的历史频率"""
    
    # 读取CSV文件
    df = pd.read_csv('kuaile8.csv', encoding='utf-8')
    print(f"数据加载成功，共 {len(df)} 期数据")
    print(f"数据范围：从第 {df['期号'].min()} 期到第 {df['期号'].max()} 期")
    
    # 收集所有号码
    all_numbers = []
    for _, row in df.iterrows():
        # 解析号码字符串
        numbers_str = row['号码'].strip('"')
        numbers = [int(num) for num in numbers_str.split(',')]
        all_numbers.extend(numbers)
    
    # 统计频率
    frequency_counter = Counter(all_numbers)
    
    # 计算总期数和理论频率
    total_periods = len(df)
    numbers_per_draw = 20  # 每期开20个号码
    total_numbers_drawn = total_periods * numbers_per_draw
    theoretical_frequency = total_numbers_drawn / 80  # 理论上每个号码应该出现的次数
    
    print(f"\n=== 频率统计摘要 ===")
    print(f"总期数: {total_periods}")
    print(f"总号码数: {total_numbers_drawn}")
    print(f"理论频率 (每个号码): {theoretical_frequency:.2f} 次")
    
    # 创建结果列表
    results = []
    for number in range(1, 81):  # 快乐8号码范围1-80
        frequency = frequency_counter.get(number, 0)
        percentage = (frequency / total_periods) * 100
        deviation = frequency - theoretical_frequency
        deviation_percent = (deviation / theoretical_frequency) * 100 if theoretical_frequency > 0 else 0
        
        results.append({
            '号码': f"{number:02d}",
            '出现次数': frequency,
            '出现率(%)': f"{percentage:.2f}",
            '理论偏差': f"{deviation:+.1f}",
            '偏差率(%)': f"{deviation_percent:+.2f}"
        })
    
    # 按出现次数降序排列
    results.sort(key=lambda x: x['出现次数'], reverse=True)
    
    # 创建DataFrame便于显示
    results_df = pd.DataFrame(results)
    
    print(f"\n=== 号码历史频率统计 (按出现次数降序排列) ===")
    print(results_df.to_string(index=False))
    
    # 统计分析
    frequencies = [r['出现次数'] for r in results]
    max_freq = max(frequencies)
    min_freq = min(frequencies)
    avg_freq = sum(frequencies) / len(frequencies)
    
    print(f"\n=== 统计分析 ===")
    print(f"最高频率: {max_freq} 次")
    print(f"最低频率: {min_freq} 次")
    print(f"平均频率: {avg_freq:.2f} 次")
    print(f"频率差值: {max_freq - min_freq} 次")
    
    # 找出极值号码
    top_5 = results[:5]
    bottom_5 = results[-5:]
    
    print(f"\n=== TOP 5 热门号码 ===")
    for i, item in enumerate(top_5, 1):
        print(f"{i}. 号码 {item['号码']}: {item['出现次数']} 次 ({item['出现率(%)']}%)")
    
    print(f"\n=== TOP 5 冷门号码 ===")
    for i, item in enumerate(reversed(bottom_5), 1):
        print(f"{i}. 号码 {item['号码']}: {item['出现次数']} 次 ({item['出现率(%)']}%)")
    
    # 保存结果到CSV
    results_df.to_csv('kl8_frequency_analysis.csv', index=False, encoding='utf-8')
    print(f"\n详细统计结果已保存到 'kl8_frequency_analysis.csv'")
    
    return results_df

if __name__ == "__main__":
    load_and_analyze_frequency() 
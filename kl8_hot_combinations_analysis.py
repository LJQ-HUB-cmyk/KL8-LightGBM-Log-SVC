#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8最近20期热组合统计分析
统计二连号、三连号等热门组合的出现频率
"""

import pandas as pd
from itertools import combinations
from collections import Counter, defaultdict
import csv

def load_latest_20_periods():
    """加载最近20期数据"""
    with open('kuaile8.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # 获取最近20期
    latest_20 = data[-20:]
    
    periods_data = []
    for row in latest_20:
        period = row['期号']
        date = row['日期']
        numbers = [int(x) for x in row['号码'].strip('"').split(',')]
        periods_data.append({
            'period': period,
            'date': date,
            'numbers': sorted(numbers)
        })
    
    return periods_data

def analyze_hot_combinations(periods_data):
    """分析热门组合"""
    
    # 统计不同长度的组合
    combo_stats = {
        '二连号': Counter(),
        '三连号': Counter(), 
        '四连号': Counter(),
        '五连号': Counter()
    }
    
    # 统计单个号码出现频率
    single_numbers = Counter()
    
    # 统计每期的组合
    period_details = []
    
    for period_info in periods_data:
        period = period_info['period']
        date = period_info['date']
        numbers = period_info['numbers']
        
        # 统计单个号码
        single_numbers.update(numbers)
        
        # 生成组合
        combos_2 = list(combinations(numbers, 2))
        combos_3 = list(combinations(numbers, 3))
        combos_4 = list(combinations(numbers, 4))
        combos_5 = list(combinations(numbers, 5))
        
        # 更新统计
        combo_stats['二连号'].update(combos_2)
        combo_stats['三连号'].update(combos_3)
        combo_stats['四连号'].update(combos_4)
        combo_stats['五连号'].update(combos_5)
        
        # 记录期次详情
        period_details.append({
            'period': period,
            'date': date,
            'numbers': numbers,
            'combo_2_count': len(combos_2),
            'combo_3_count': len(combos_3),
            'combo_4_count': len(combos_4),
            'combo_5_count': len(combos_5)
        })
    
    return combo_stats, single_numbers, period_details

def format_combination(combo):
    """格式化组合显示"""
    return ','.join([f'{num:02d}' for num in combo])

def print_analysis_report(combo_stats, single_numbers, period_details):
    """打印分析报告"""
    
    print("="*60)
    print("               快乐8最近20期热组合统计报告")
    print("="*60)
    print(f"统计期数: {period_details[0]['period']} - {period_details[-1]['period']}")
    print(f"统计日期: {period_details[0]['date']} - {period_details[-1]['date']}")
    print()
    
    # 1. 单个号码频率统计 (TOP 20)
    print("🔥 单个号码出现频率统计 (TOP 20)")
    print("-" * 40)
    for i, (num, freq) in enumerate(single_numbers.most_common(20), 1):
        percentage = (freq / 20) * 100
        print(f"{i:2d}. 号码 {num:02d}: {freq:2d}次 ({percentage:5.1f}%)")
    print()
    
    # 2. 二连号组合统计 (TOP 15)
    print("🔥 二连号热门组合 (TOP 15)")
    print("-" * 40)
    for i, (combo, freq) in enumerate(combo_stats['二连号'].most_common(15), 1):
        percentage = (freq / 20) * 100
        print(f"{i:2d}. [{format_combination(combo)}]: {freq:2d}次 ({percentage:5.1f}%)")
    print()
    
    # 3. 三连号组合统计 (TOP 10)
    print("🔥 三连号热门组合 (TOP 10)")
    print("-" * 40)
    for i, (combo, freq) in enumerate(combo_stats['三连号'].most_common(10), 1):
        percentage = (freq / 20) * 100
        print(f"{i:2d}. [{format_combination(combo)}]: {freq:2d}次 ({percentage:5.1f}%)")
    print()
    
    # 4. 四连号组合统计 (TOP 10)
    print("🔥 四连号热门组合 (TOP 10)")
    print("-" * 40)
    four_combos = combo_stats['四连号'].most_common(10)
    if four_combos:
        for i, (combo, freq) in enumerate(four_combos, 1):
            percentage = (freq / 20) * 100
            print(f"{i:2d}. [{format_combination(combo)}]: {freq:2d}次 ({percentage:5.1f}%)")
    else:
        print("无四连号组合出现超过1次")
    print()
    
    # 5. 五连号组合统计 (TOP 5)
    print("🔥 五连号热门组合 (TOP 5)")
    print("-" * 40)
    five_combos = combo_stats['五连号'].most_common(5)
    if five_combos:
        for i, (combo, freq) in enumerate(five_combos, 1):
            percentage = (freq / 20) * 100
            print(f"{i:2d}. [{format_combination(combo)}]: {freq:2d}次 ({percentage:5.1f}%)")
    else:
        print("无五连号组合出现超过1次")
    print()
    
    # 6. 统计摘要
    print("📊 统计摘要")
    print("-" * 40)
    total_combos = {
        '二连号': sum(combo_stats['二连号'].values()),
        '三连号': sum(combo_stats['三连号'].values()),
        '四连号': sum(combo_stats['四连号'].values()),
        '五连号': sum(combo_stats['五连号'].values())
    }
    
    unique_combos = {
        '二连号': len(combo_stats['二连号']),
        '三连号': len(combo_stats['三连号']),
        '四连号': len(combo_stats['四连号']),
        '五连号': len(combo_stats['五连号'])
    }
    
    print(f"二连号: 总计 {total_combos['二连号']} 个组合，其中 {unique_combos['二连号']} 个不同组合")
    print(f"三连号: 总计 {total_combos['三连号']} 个组合，其中 {unique_combos['三连号']} 个不同组合")
    print(f"四连号: 总计 {total_combos['四连号']} 个组合，其中 {unique_combos['四连号']} 个不同组合")
    print(f"五连号: 总计 {total_combos['五连号']} 个组合，其中 {unique_combos['五连号']} 个不同组合")
    print()
    
    # 7. 最近5期详情
    print("📋 最近5期开奖详情")
    print("-" * 60)
    for period_info in period_details[-5:]:
        numbers_str = ','.join([f'{num:02d}' for num in period_info['numbers']])
        print(f"期号 {period_info['period']} ({period_info['date']})")
        print(f"    开奖号码: {numbers_str}")
        print(f"    组合数量: 二连({period_info['combo_2_count']}) 三连({period_info['combo_3_count']}) "
              f"四连({period_info['combo_4_count']}) 五连({period_info['combo_5_count']})")
        print()

def save_to_csv(combo_stats, single_numbers, period_details):
    """保存统计结果到CSV文件"""
    
    # 保存单个号码频率
    with open('kl8_hot_numbers_latest20.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['排名', '号码', '出现次数', '出现率%'])
        for i, (num, freq) in enumerate(single_numbers.most_common(), 1):
            percentage = (freq / 20) * 100
            writer.writerow([i, f'{num:02d}', freq, f'{percentage:.1f}'])
    
    # 保存二连号组合
    with open('kl8_hot_combinations_2_latest20.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['排名', '组合', '出现次数', '出现率%'])
        for i, (combo, freq) in enumerate(combo_stats['二连号'].most_common(), 1):
            percentage = (freq / 20) * 100
            combo_str = format_combination(combo)
            writer.writerow([i, combo_str, freq, f'{percentage:.1f}'])
    
    # 保存三连号组合
    with open('kl8_hot_combinations_3_latest20.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['排名', '组合', '出现次数', '出现率%'])
        for i, (combo, freq) in enumerate(combo_stats['三连号'].most_common(), 1):
            percentage = (freq / 20) * 100
            combo_str = format_combination(combo)
            writer.writerow([i, combo_str, freq, f'{percentage:.1f}'])

def main():
    """主函数"""
    
    # 加载数据
    print("正在加载最近20期数据...")
    periods_data = load_latest_20_periods()
    
    # 分析组合
    print("正在分析热门组合...")
    combo_stats, single_numbers, period_details = analyze_hot_combinations(periods_data)
    
    # 打印报告
    print_analysis_report(combo_stats, single_numbers, period_details)
    
    # 保存到CSV
    print("正在保存统计结果到CSV文件...")
    save_to_csv(combo_stats, single_numbers, period_details)
    
    print("="*60)
    print("分析完成！统计结果已保存到以下文件：")
    print("- kl8_hot_numbers_latest20.csv (单个号码频率)")
    print("- kl8_hot_combinations_2_latest20.csv (二连号组合)")
    print("- kl8_hot_combinations_3_latest20.csv (三连号组合)")
    print("="*60)

if __name__ == "__main__":
    main() 
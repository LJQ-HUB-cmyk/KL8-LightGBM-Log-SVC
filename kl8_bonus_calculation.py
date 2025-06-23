# -*- coding: utf-8 -*-
"""
快乐8推荐结果验证与奖金计算器
=================================

本脚本旨在自动评估 `kl8_analyzer.py` 生成的推荐号码的实际表现。

工作流程:
1.  读取 `kuaile8.csv` 文件，获取所有历史开奖数据。
2.  确定最新的一期为"评估期"，倒数第二期为"报告数据截止期"。
3.  根据"报告数据截止期"，在当前目录下查找对应的分析报告文件
    (kl8_analysis_output_*.txt)。
4.  从找到的报告中解析出推荐的号码组合。
5.  使用"评估期"的实际开奖号码，核对所有推荐投注的中奖情况。
6.  计算总奖金，并将详细的中奖结果（包括中奖号码、奖级、金额）
    追加记录到主报告文件 `latest_kl8_calculation.txt` 中。
7.  主报告文件会自动管理记录数量，只保留最新的N条评估记录和错误日志。
"""

import os
import re
import glob
import csv
from datetime import datetime
import traceback
from typing import Optional, Tuple, List, Dict

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 脚本需要查找的分析报告文件名的模式
REPORT_PATTERN = "kl8_analysis_output_*.txt"
# 开奖数据源CSV文件
CSV_FILE = "kuaile8.csv"
# 最终生成的主评估报告文件名
MAIN_REPORT_FILE = "latest_kl8_calculation.txt"

# 主报告文件中保留的最大记录数
MAX_NORMAL_RECORDS = 10  # 保留最近10次评估
MAX_ERROR_LOGS = 20      # 保留最近20条错误日志

# 奖金对照表 (元) - 快乐8奖金表 [已废弃，现使用下方的play_prize_rules详细配置]
# PRIZE_TABLE = {
#     1: 100_000,    # 一等奖 (15+ hit)
#     2: 10_000,     # 二等奖 (13-14 hit)
#     3: 1_000,      # 三等奖 (11-12 hit)
#     4: 100,        # 四等奖 (9-10 hit)
#     5: 10,         # 五等奖 (7-8 hit)
#     6: 5,          # 六等奖 (5-6 hit)
# }

# ==============================================================================
# --- 工具函数 ---
# ==============================================================================

def log_message(message: str, level: str = "INFO"):
    """一个简单的日志打印函数，用于在控制台显示脚本执行状态。"""
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')} - {message}")

def robust_file_read(file_path: str) -> Optional[str]:
    """
    一个健壮的文件读取函数，能自动尝试多种编码格式。

    Args:
        file_path (str): 待读取文件的路径。

    Returns:
        Optional[str]: 文件内容字符串，如果失败则返回 None。
    """
    if not os.path.exists(file_path):
        log_message(f"文件未找到: {file_path}", "ERROR")
        return None
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, IOError):
            continue
    log_message(f"无法使用任何支持的编码打开文件: {file_path}", "ERROR")
    return None

# ==============================================================================
# --- 数据解析与查找模块 ---
# ==============================================================================

def get_period_data_from_csv(csv_content: str) -> Tuple[Optional[Dict], Optional[List]]:
    """
    从CSV文件内容中解析出所有期号的开奖数据。

    Args:
        csv_content (str): 从CSV文件读取的字符串内容。

    Returns:
        Tuple[Optional[Dict], Optional[List]]:
            - 一个以期号为键，开奖数据为值的字典。
            - 一个按升序排序的期号列表。
            如果解析失败则返回 (None, None)。
    """
    if not csv_content:
        log_message("输入的CSV内容为空。", "WARNING")
        return None, None
    period_map, periods_list = {}, []
    try:
        reader = csv.reader(csv_content.splitlines())
        next(reader)  # 跳过表头
        for i, row in enumerate(reader):
            if len(row) >= 3 and re.match(r'^\d{4,7}$', row[0]):
                try:
                    period, date, numbers_str = row[0], row[1], row[2]
                    numbers = sorted(map(int, numbers_str.split(',')))
                    if len(numbers) != 20 or not all(1 <= n <= 80 for n in numbers):
                        continue
                    period_map[period] = {'date': date, 'numbers': numbers}
                    periods_list.append(period)
                except (ValueError, IndexError):
                    log_message(f"CSV文件第 {i+2} 行数据格式无效，已跳过: {row}", "WARNING")
    except Exception as e:
        log_message(f"解析CSV数据时发生严重错误: {e}", "ERROR")
        return None, None
    
    if not period_map:
        log_message("未能从CSV中解析到任何有效的开奖数据。", "WARNING")
        return None, None
        
    return period_map, sorted(periods_list, key=int)

def find_matching_report(target_period: str) -> Optional[str]:
    """
    在当前目录查找其数据截止期与 `target_period` 匹配的最新分析报告。

    Args:
        target_period (str): 目标报告的数据截止期号。

    Returns:
        Optional[str]: 找到的报告文件的路径，如果未找到则返回 None。
    """
    log_message(f"正在查找数据截止期为 {target_period} 的分析报告...")
    candidates = []
    # 使用 SCRIPT_DIR 确保在任何工作目录下都能找到与脚本同级的报告文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file_path in glob.glob(os.path.join(script_dir, REPORT_PATTERN)):
        content = robust_file_read(file_path)
        if not content: continue
        
        match = re.search(r'分析基于数据:\s*截至\s*(\d+)\s*期', content)
        if match and match.group(1) == target_period:
            try:
                # 从文件名中提取时间戳以确定最新报告
                timestamp_str_match = re.search(r'_(\d{8}_\d{6})\.txt$', file_path)
                if timestamp_str_match:
                    timestamp_str = timestamp_str_match.group(1)
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidates.append((timestamp, file_path))
            except (AttributeError, ValueError):
                continue
    
    if not candidates:
        log_message(f"未找到数据截止期为 {target_period} 的分析报告。", "WARNING")
        return None
        
    candidates.sort(reverse=True)
    latest_report = candidates[0][1]
    log_message(f"找到匹配的最新报告: {os.path.basename(latest_report)}", "INFO")
    return latest_report

def parse_recommendations_from_report(content: str) -> Tuple[Dict, List]:
    """
    从分析报告内容中解析出各种玩法的推荐号码组合。

    Args:
        content (str): 分析报告的文本内容。

    Returns:
        Tuple[Dict, List]:
            - 各玩法推荐字典, e.g., {1: [3, 15], 2: [7, 23], ...}
            - 复式推荐号码列表, e.g., [1,2,3,4,5,...,20]
    """
    play_recommendations = {}
    
    # 解析各玩法推荐 (选一: [03 15 23] 格式)
    play_patterns = {
        '选一': 1, '选二': 2, '选三': 3, '选四': 4, '选五': 5,
        '选六': 6, '选七': 7, '选八': 8, '选九': 9, '选十': 10
    }
    
    for play_name, play_type in play_patterns.items():
        pattern = re.compile(rf'{re.escape(play_name)}:\s*\[([\d\s]+)\]')
        match = pattern.search(content)
        if match:
            try:
                numbers = sorted(map(int, match.group(1).split()))
                if len(numbers) == play_type:  # 验证号码数量是否正确
                    play_recommendations[play_type] = numbers
            except ValueError:
                continue
    
    # 解析复式推荐
    complex_numbers = []
    complex_patterns = [
        r'复式推荐:\s*\[([\d\s]+)\]',
        r'推荐号码池.*:\s*([\d\s]+)',
        r'复式参考.*:\s*([\d\s]+)'
    ]
    
    for pattern in complex_patterns:
        match = re.search(pattern, content)
        if match:
            try:
                complex_numbers = sorted(map(int, match.group(1).split()))
                break
            except ValueError:
                continue
        
    log_message(f"从报告中解析出 {len(play_recommendations)} 种玩法推荐，{len(complex_numbers)} 个复式号码。", "INFO")
    return play_recommendations, complex_numbers

def calculate_multi_play_prize(play_recommendations: Dict, prize_numbers: List) -> Tuple[int, Dict, List]:
    """
    计算多种玩法的投注奖金和中奖情况。

    Args:
        play_recommendations (Dict): 各玩法推荐字典 {play_type: numbers}。
        prize_numbers (List): 开奖号码列表。

    Returns:
        Tuple[int, Dict, List]:
            - 总奖金金额
            - 各玩法各奖级的中奖次数字典
            - 详细中奖信息列表
    """
    if not play_recommendations or not prize_numbers:
        return 0, {}, []
        
    # 快乐8各玩法中奖规则 - 根据福彩快乐8官方规则设定
    play_prize_rules = {
        1: {1: ("一等奖", 4.6)},  # 选一：中1个号码，奖金4.6元
        2: {2: ("一等奖", 19)},   # 选二：中2个号码，奖金19元
        3: {3: ("一等奖", 53), 2: ("二等奖", 3)},   # 选三：中3个53元，中2个3元
        4: {4: ("一等奖", 100), 3: ("二等奖", 5), 2: ("三等奖", 3)},    # 选四：中4个100元，中3个5元，中2个3元
        5: {5: ("一等奖", 1000), 4: ("二等奖", 21), 3: ("三等奖", 3)},     # 选五：中5个1000元，中4个21元，中3个3元
        6: {6: ("一等奖", 3000), 5: ("二等奖", 30), 4: ("三等奖", 10), 3: ("四等奖", 3)},     # 选六：中6个3000元，中5个30元，中4个10元，中3个3元
        7: {7: ("一等奖", 10000), 6: ("二等奖", 288), 5: ("三等奖", 28), 4: ("四等奖", 4), 0: ("五等奖", 2)},  # 选七：中7个10000元，中6个288元，中5个28元，中4个4元，全不中2元
        8: {8: ("一等奖", 50000), 7: ("二等奖", 800), 6: ("三等奖", 88), 5: ("四等奖", 10), 4: ("五等奖", 3), 0: ("六等奖", 2)},   # 选八：中8个50000元，中7个800元，中6个88元，中5个10元，中4个3元，全不中2元
        9: {9: ("一等奖", 300000), 8: ("二等奖", 2000), 7: ("三等奖", 200), 6: ("四等奖", 20), 5: ("五等奖", 5), 4: ("六等奖", 3), 0: ("七等奖", 2)}, # 选九：中9个300000元，中8个2000元，中7个200元，中6个20元，中5个5元，中4个3元，全不中2元
        10: {10: ("一等奖", 5000000), 9: ("二等奖", 8000), 8: ("三等奖", 800), 7: ("四等奖", 80), 6: ("五等奖", 5), 5: ("六等奖", 3), 0: ("七等奖", 2)}  # 选十：中10个一等奖500万元，中9个二等奖8000元，中8个三等奖800元，中7个四等奖80元，中6个五等奖5元，中5个六等奖3元，全不中七等奖2元
    }
    
    prize_set = set(prize_numbers)
    total_amount = 0
    prize_count_dict = {}
    detailed_winnings = []
    
    for play_type, recommended_numbers in play_recommendations.items():
        if play_type not in play_prize_rules:
            continue
            
        hit_count = len(set(recommended_numbers) & prize_set)
        prize_rules = play_prize_rules[play_type]
        
        if hit_count in prize_rules:
            prize_level_name, amount = prize_rules[hit_count]
            
            # 记录中奖统计
            play_key = f"选{play_type}"
            if play_key not in prize_count_dict:
                prize_count_dict[play_key] = {}
            prize_count_dict[play_key][prize_level_name] = prize_count_dict[play_key].get(prize_level_name, 0) + 1
            
            total_amount += amount
            detailed_winnings.append({
                'play_type': play_type,
                'play_name': f"选{play_type}",
                'hit_count': hit_count,
                'prize_level': prize_level_name,
                'amount': amount,
                'recommended_numbers': recommended_numbers
            })
        else:
            # 记录未中奖的玩法
            detailed_winnings.append({
                'play_type': play_type,
                'play_name': f"选{play_type}",
                'hit_count': hit_count,
                'prize_level': None,
                'amount': 0,
                'recommended_numbers': recommended_numbers
            })
            
    return total_amount, prize_count_dict, detailed_winnings

def calculate_prize(tickets: List, prize_numbers: List) -> Tuple[int, Dict, List]:
    """
    为向后兼容保留的旧版本中奖计算函数（选十玩法）。

    Args:
        tickets (List): 投注号码组合列表。
        prize_numbers (List): 开奖号码列表。

    Returns:
        Tuple[int, Dict, List]:
            - 总奖金金额
            - 各奖级的中奖次数字典
            - 详细中奖信息列表
    """
    if not tickets or not prize_numbers:
        return 0, {}, []
        
    # 转换为多玩法格式进行计算
    play_recommendations = {}
    for i, ticket in enumerate(tickets):
        if len(ticket) == 20:  # 选十玩法
            play_recommendations[10] = ticket
            break  # 只处理第一个选十组合
    
    return calculate_multi_play_prize(play_recommendations, prize_numbers)

def format_multi_play_results_for_report(winning_list: List[Dict], prize_numbers: List) -> List[str]:
    """格式化多种玩法中奖信息为报告字符串。"""
    if not winning_list: 
        return ["  各玩法均未中奖。"]
    
    result = [f"  开奖号码: {' '.join(f'{n:02d}' for n in sorted(prize_numbers))}\n"]
    
    # 按玩法类型分组显示
    for win in winning_list:
        numbers_str = ' '.join(f'{n:02d}' for n in win['recommended_numbers'])
        if win['prize_level']:
            result.append(f"  {win['play_name']}: [{numbers_str}] | 命中 {win['hit_count']} 个 | {win['prize_level']} | {win['amount']} 元")
        else:
            result.append(f"  {win['play_name']}: [{numbers_str}] | 命中 {win['hit_count']} 个 | 未中奖")
    return result

def format_winning_tickets_for_report(winning_list: List[Dict], prize_numbers: List) -> List[str]:
    """为向后兼容保留的旧版本格式化函数。"""
    # 如果是新格式的数据，调用新函数
    if winning_list and 'play_name' in winning_list[0]:
        return format_multi_play_results_for_report(winning_list, prize_numbers)
    
    # 旧格式处理
    if not winning_list: return ["  无中奖票。"]
    
    result = [f"  开奖号码: {' '.join(f'{n:02d}' for n in sorted(prize_numbers))}\n"]
    for win in winning_list:
        ticket_str = ' '.join(f'{n:02d}' for n in win['ticket_numbers'])
        result.append(f"  注 {win['ticket_no']}: [{ticket_str}] | 命中 {win['hit_count']} 个 | {win['prize_level']} | {win['amount']} 元")
    return result

def manage_report(new_entry: Optional[Dict] = None, new_error: Optional[str] = None):
    """
    管理主报告文件，添加新记录或错误日志，并维护文件大小。

    Args:
        new_entry (Optional[Dict]): 要添加的新评估记录。
        new_error (Optional[str]): 要添加的错误日志。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, MAIN_REPORT_FILE)
    
    # 读取现有内容
    existing_content = ""
    if os.path.exists(report_path):
        existing_content = robust_file_read(report_path) or ""
    
    # 解析现有记录和错误日志
    normal_records, error_logs = [], []
    report_counter = 0
    current_section = None
    
    # 查找现有验证报告的最大编号
    report_numbers = re.findall(r'### 验证报告 #(\d+)', existing_content)
    if report_numbers:
        report_counter = max(int(num) for num in report_numbers)
    
    for line in existing_content.split('\n'):
        if line.strip() == "=== 评估记录 ===":
            current_section = "normal"
            continue
        elif line.strip() == "=== 错误日志 ===":
            current_section = "error"
            continue
        elif line.strip() and current_section == "normal":
            normal_records.append(line)
        elif line.strip() and current_section == "error":
            error_logs.append(line)
    
    # 添加新记录
    if new_entry:
        report_counter += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算各玩法的中奖情况
        play_details = new_entry.get('details', [])
        play_stats = []
        total_bets = 10  # 10种玩法各投注1注
        total_wins = 0
        
        # 从详情中提取各玩法验证结果
        for detail in play_details:
            if '选' in detail and ':' in detail:
                if '未中奖' not in detail and '命中 0 个' not in detail:
                    total_wins += 1
                play_stats.append(detail.strip())
        
        # 生成格式化的验证报告
        new_report = f"""### 验证报告 #{report_counter} (期号: {new_entry['period']})
验证时间: {timestamp}
实际开奖号码: {new_entry.get('winning_numbers', '未知')}
总投注注数: {total_bets}
总中奖注数: {total_wins}
总奖金: {new_entry['total_amount']} 元
中奖率: {(total_wins/total_bets*100):.1f}%

各玩法验证结果:
{chr(10).join(play_stats) if play_stats else '  暂无数据'}
"""
        normal_records.append(new_report)
    
    if new_error:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_logs.append(f"[{timestamp}] {new_error}")
    
    # 保持记录数量限制
    normal_records = normal_records[-MAX_NORMAL_RECORDS:]
    error_logs = error_logs[-MAX_ERROR_LOGS:]
    
    # 写入文件
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("快乐8推荐结果验证报告\n")
            f.write("=" * 40 + "\n\n")
            f.write("=== 评估记录 ===\n")
            f.write('\n'.join(normal_records))
            f.write("\n\n=== 错误日志 ===\n")
            f.write('\n'.join(error_logs))
        log_message(f"报告已更新: {MAIN_REPORT_FILE}", "INFO")
    except Exception as e:
        log_message(f"写入报告文件失败: {e}", "ERROR")

def main_process():
    """主处理流程。"""
    log_message("快乐8奖金计算器启动", "INFO")
    
    try:
        # 1. 加载CSV数据
        csv_content = robust_file_read(CSV_FILE)
        if not csv_content:
            raise Exception(f"无法读取数据文件 {CSV_FILE}")
        
        period_data, periods = get_period_data_from_csv(csv_content)
        if not period_data or not periods:
            raise Exception("CSV数据解析失败或为空")
        
        if len(periods) < 2:
            raise Exception("数据不足，至少需要2期数据")
        
        # 2. 确定评估期和数据截止期
        eval_period = periods[-1]  # 最新期作为评估期
        data_cutoff_period = periods[-2]  # 倒数第二期作为数据截止期
        
        log_message(f"评估期: {eval_period}, 数据截止期: {data_cutoff_period}", "INFO")
        
        # 3. 查找匹配的分析报告
        report_file = find_matching_report(data_cutoff_period)
        if not report_file:
            raise Exception(f"未找到数据截止期为 {data_cutoff_period} 的分析报告")
        
        # 4. 解析推荐号码
        report_content = robust_file_read(report_file)
        if not report_content:
            raise Exception(f"无法读取报告文件 {report_file}")
        
        play_recommendations, complex_numbers = parse_recommendations_from_report(report_content)
        if not play_recommendations:
            raise Exception("报告中未找到有效的推荐组合")
        
        # 5. 获取实际开奖结果
        actual_result = period_data[eval_period]
        prize_numbers = actual_result['numbers']
        
        # 6. 计算各玩法中奖情况
        total_amount, prize_counts, winning_details = calculate_multi_play_prize(play_recommendations, prize_numbers)
        
        # 7. 格式化结果
        summary = ', '.join([f"{level}: {count}次" for level, count in prize_counts.items()]) if prize_counts else "未中奖"
        detail_lines = format_winning_tickets_for_report(winning_details, prize_numbers)
        
        # 8. 输出结果
        log_message(f"=== 期号 {eval_period} 验证结果 ===", "INFO")
        log_message(f"总奖金: {total_amount} 元", "INFO")
        log_message(f"中奖情况: {summary}", "INFO")
        if winning_details:
            log_message("详细中奖信息:", "INFO")
            for line in detail_lines:
                log_message(line, "INFO")
        
        # 9. 更新报告文件
        report_entry = {
            'period': eval_period,
            'total_amount': total_amount,
            'summary': summary,
            'details': detail_lines,
            'winning_numbers': ' '.join(f'{n:02d}' for n in sorted(prize_numbers))
        }
        manage_report(new_entry=report_entry)
        
        log_message("处理完成", "INFO")
        
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        log_message(error_msg, "ERROR")
        log_message(f"详细错误信息:\n{traceback.format_exc()}", "ERROR")
        
        # 记录错误到报告文件
        manage_report(new_error=error_msg)

if __name__ == "__main__":
    main_process() 
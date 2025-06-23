#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8微信推送模块
用于将分析结果和验证结果推送到微信
"""

import requests
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any

# 微信推送配置
APP_TOKEN = "AT_FInZJJ0mUU8xvQjKRP7v6omvuHN3Fdqw"
USER_UIDS = ["UID_yYObqdMVScIa66DGR2n2PCRFL10w"]
TOPIC_IDS = []  # 暂时不使用话题推送，只使用个人推送

def send_wxpusher_message(content: str, title: str = None, content_type: int = 1) -> Dict[str, Any]:
    """
    发送微信推送消息
    
    Args:
        content: 消息内容
        title: 消息标题
        content_type: 内容类型，1=文本，2=HTML，3=Markdown
    
    Returns:
        推送结果
    """
    url = "https://wxpusher.zjiecode.com/api/send/message"
    headers = {"Content-Type": "application/json"}
    data = {
        "appToken": APP_TOKEN,
        "content": content,
        "uids": USER_UIDS,
        "topicIds": TOPIC_IDS,
        "summary": title,
        "contentType": content_type,
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        result = response.json()
        print(f"微信推送结果: {result}")
        return result
    except Exception as e:
        print(f"微信推送失败: {e}")
        return {"success": False, "error": str(e)}

def extract_analysis_summary(file_path: str) -> str:
    """
    从分析报告中提取关键信息
    
    Args:
        file_path: 分析报告文件路径
    
    Returns:
        格式化的摘要信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取生成时间
        time_match = re.search(r'报告生成时间: ([^\n]+)', content)
        gen_time = time_match.group(1) if time_match else "未知"
        
        # 提取分析期数和预测目标
        period_match = re.search(r'分析基于数据: 截至 (\d+) 期', content)
        current_period = period_match.group(1) if period_match else "未知"
        
        target_match = re.search(r'本次预测目标: 第 (\d+) 期', content)
        target_period = target_match.group(1) if target_match else "未知"
        
        # 提取优化状态
        optuna_status = "未启用"
        if "Optuna 优化摘要" in content:
            status_match = re.search(r'优化状态: ([^\n]+)', content)
            if status_match and "完成" in status_match.group(1):
                score_match = re.search(r'最佳性能得分: ([\d.]+)', content)
                score = score_match.group(1) if score_match else "未知"
                optuna_status = f"已优化 (得分: {score})"
        
        # 提取各玩法推荐号码（新格式）
        play_recommendations = []
        play_section_match = re.search(r'--- 各玩法推荐号码 ---(.*?)(?=---|$)', content, re.DOTALL)
        if play_section_match:
            play_section = play_section_match.group(1)
            # 匹配格式：选一: [03] (得分: 95.2)
            # 中文数字映射
            chinese_numbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
            for play_num in range(1, 11):
                chinese_num = chinese_numbers[play_num - 1]
                play_pattern = f'选{chinese_num}: \\[([^\\]]+)\\]'
                play_match = re.search(play_pattern, play_section)
                if play_match:
                    numbers = play_match.group(1)
                    play_recommendations.append(f"  选{chinese_num}: [{numbers}]")
        
        # 提取复式推荐号码（新格式）
        complex_match = re.search(r'复式推荐: \[([^\]]+)\]', content)
        complex_numbers = complex_match.group(1) if complex_match else "未找到"
        
        # 提取回测ROI
        roi_match = re.search(r'总体ROI: ([-\d.]+)%', content)
        roi = roi_match.group(1) if roi_match else "未知"
        
        summary = f"""📅 生成时间: {gen_time}
📊 分析数据: 截至第{current_period}期
🎯 预测目标: 第{target_period}期
🚀 优化状态: {optuna_status}
📈 历史ROI: {roi}%

🎯 各玩法推荐:
{chr(10).join(play_recommendations) if play_recommendations else "  暂无推荐"}

🎲 复式号码: {complex_numbers}"""
        
        return summary
        
    except Exception as e:
        return f"提取分析摘要失败: {str(e)}"

def extract_calculation_summary(file_path: str) -> str:
    """
    从验证计算报告中提取关键信息
    
    Args:
        file_path: 验证计算报告文件路径
    
    Returns:
        格式化的摘要信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找最新的验证报告
        report_matches = re.findall(r'### 验证报告 #(\d+) \(期号: (\d+)\)', content)
        if not report_matches:
            return "未找到验证报告"
        
        # 获取最新报告编号
        latest_report_num = max(int(num) for num, _ in report_matches)
        latest_period = next(period for num, period in report_matches if int(num) == latest_report_num)
        
        # 提取最新报告的详细信息
        report_pattern = f'### 验证报告 #{latest_report_num} \\(期号: {latest_period}\\)(.*?)(?=### 验证报告|$)'
        report_match = re.search(report_pattern, content, re.DOTALL)
        
        if not report_match:
            return "无法提取最新验证报告"
        
        report_content = report_match.group(1)
        
        # 提取验证时间
        time_match = re.search(r'验证时间: ([^\n]+)', report_content)
        verify_time = time_match.group(1) if time_match else "未知"
        
        # 提取开奖号码
        numbers_match = re.search(r'实际开奖号码: ([^\n]+)', report_content)
        winning_numbers = numbers_match.group(1) if numbers_match else "未知"
        
        # 提取总体验证结果
        total_bets_match = re.search(r'总投注注数: (\d+)', report_content)
        total_bets = total_bets_match.group(1) if total_bets_match else "未知"
        
        total_wins_match = re.search(r'总中奖注数: (\d+)', report_content)
        total_wins = total_wins_match.group(1) if total_wins_match else "0"
        
        total_prize_match = re.search(r'总奖金: ([\d.]+) 元', report_content)
        total_prize = total_prize_match.group(1) if total_prize_match else "0"
        
        # 提取所有玩法中奖情况
        play_results = []
        
        # 查找各玩法验证结果部分
        play_section_match = re.search(r'各玩法验证结果:\s*\n(.*?)(?=\n\n|$)', report_content, re.DOTALL)
        if play_section_match:
            play_section = play_section_match.group(1)
            
            # 解析每个玩法的结果
            for line in play_section.split('\n'):
                line = line.strip()
                if line.startswith('选') and ':' in line:
                    # 提取玩法信息
                    if '未中奖' in line or '命中 0 个' in line:
                        play_num = re.search(r'选(\d+)', line)
                        if play_num:
                            play_results.append(f"  选{play_num.group(1)}: 未中奖")
                    else:
                        # 提取中奖信息
                        prize_match = re.search(r'选(\d+).*?(\d+)\s*元', line)
                        if prize_match:
                            play_num, amount = prize_match.groups()
                            play_results.append(f"  选{play_num}: {amount}元")
                        else:
                            # 如果没有找到金额，直接使用原始行
                            play_results.append(f"  {line}")
        else:
            # 兼容旧格式 - 如果没有找到新格式，尝试旧格式
            for play_num in range(1, 11):
                play_pattern = f'选{play_num}玩法验证:.*?中奖: ([^\\n]+)'
                play_match = re.search(play_pattern, report_content, re.DOTALL)
                if play_match:
                    win_result = play_match.group(1)
                    # 提取中奖注数和奖金
                    win_nums_match = re.search(r'(\d+) 注', win_result)
                    prize_match = re.search(r'奖金 ([\d.]+) 元', win_result)
                    
                    win_nums = win_nums_match.group(1) if win_nums_match else "0"
                    prize = prize_match.group(1) if prize_match else "0"
                    
                    if win_nums == "0":
                        play_results.append(f"  选{play_num}: 未中奖")
                    else:
                        play_results.append(f"  选{play_num}: {win_nums}注 {prize}元")
        
        # 计算中奖率
        hit_rate = "0%"
        if total_bets != "未知" and int(total_bets) > 0:
            rate = (int(total_wins) / int(total_bets)) * 100
            hit_rate = f"{rate:.1f}%"
        
        summary = f"""📈 验证报告 #{latest_report_num}
⏰ 验证时间: {verify_time}
🎯 验证期号: {latest_period}
🎰 开奖号码: {winning_numbers}
📊 投注统计: {total_wins}/{total_bets} 注中奖
💰 中奖率: {hit_rate}
💵 总奖金: {total_prize} 元

🎮 各玩法验证结果:
{chr(10).join(play_results) if play_results else "  暂无数据"}"""
        
        return summary
        
    except Exception as e:
        return f"提取验证摘要失败: {str(e)}"

def send_analysis_notification(analysis_file: str) -> bool:
    """
    发送分析结果通知
    
    Args:
        analysis_file: 分析报告文件路径
    
    Returns:
        是否发送成功
    """
    summary = extract_analysis_summary(analysis_file)
    
    title = "🎯 快乐8分析报告"
    content = f"""🎲 快乐8智能分析报告

{summary}

📝 详细报告请查看GitHub项目
⏰ 推送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    result = send_wxpusher_message(content, title)
    return result.get("success", False) or result.get("code") == 1000

def send_calculation_notification(calculation_file: str) -> bool:
    """
    发送验证结果通知
    
    Args:
        calculation_file: 验证计算报告文件路径
    
    Returns:
        是否发送成功
    """
    summary = extract_calculation_summary(calculation_file)
    
    title = "💰 快乐8验证结果"
    content = f"""💎 快乐8推荐验证结果

{summary}

📊 详细验证请查看GitHub项目
⏰ 推送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    result = send_wxpusher_message(content, title)
    return result.get("success", False) or result.get("code") == 1000

def main():
    """
    主函数，发送分析和验证结果通知
    """
    print("开始发送快乐8微信推送通知...")
    
    # 发送分析结果
    analysis_file = "latest_kl8_analysis.txt"
    if os.path.exists(analysis_file):
        print("发送分析结果通知...")
        if send_analysis_notification(analysis_file):
            print("✅ 分析结果推送成功")
        else:
            print("❌ 分析结果推送失败")
    else:
        print(f"⚠️ 分析文件不存在: {analysis_file}")
    
    # 发送验证结果
    calculation_file = "latest_kl8_calculation.txt"
    if os.path.exists(calculation_file):
        print("发送验证结果通知...")
        if send_calculation_notification(calculation_file):
            print("✅ 验证结果推送成功")
        else:
            print("❌ 验证结果推送失败")
    else:
        print(f"⚠️ 验证文件不存在: {calculation_file}")
    
    print("微信推送通知完成")

if __name__ == "__main__":
    main()
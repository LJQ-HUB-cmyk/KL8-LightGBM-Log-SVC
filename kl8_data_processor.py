# -*- coding: utf-8 -*-
"""
快乐8数据处理器
================

本脚本负责从网络上获取快乐8的历史开奖数据，并将其保存到本地的CSV文件中。

主要功能:
1.  从文本文件 (kl8_asc.txt) 获取完整历史数据。
2.  将数据保存到主CSV文件 ('kuaile8.csv') 中。
3.  具备良好的错误处理和日志记录能力，能应对网络波动和数据格式问题。
"""

import pandas as pd
import sys
import os
import requests
import logging
from datetime import datetime

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 获取脚本所在的目录，确保路径的相对性
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 目标CSV文件的完整路径
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'kuaile8.csv')

# 网络数据源URL
TXT_DATA_URL = 'https://data.17500.cn/kl8_asc.txt'

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('kl8_data_processor')

# ==============================================================================
# --- 数据获取模块 ---
# ==============================================================================

def fetch_kl8_data():
    """
    从指定的文本文件URL下载快乐8历史数据并保存到CSV文件。
    """
    logger.info("开始获取快乐8数据...")
    
    try:
        # 发送HTTP GET请求获取数据
        response = requests.get(TXT_DATA_URL, headers={'User-agent': 'chrome'}, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        logger.info("数据下载成功")
    except requests.RequestException as e:
        logger.error(f"请求数据时出错: {e}")
        return False

    data = []
    lines = sorted(response.text.split('\n'), reverse=True)
    logger.info(f"开始解析 {len(lines)} 行数据...")

    for line_num, line in enumerate(lines, 1):
        if len(line) < 10:
            continue  # 跳过无效行

        # 仅分割第一个逗号，忽略后续数据
        parts = line.split(',', 1)
        if not parts:
            continue  # 跳过空行

        first_part = parts[0].strip()
        fields = first_part.split()

        # 确保有至少 22 个字段（Seq + 日期 + 20 个号码）
        if len(fields) < 22:
            if line_num <= 10:  # 只显示前10行的错误信息
                logger.warning(f"跳过行（字段不足22个）：{line[:50]}...")
            continue

        seq = fields[0]
        date = fields[1]  # 日期字段
        numbers = fields[2:22]  # 提取20个号码

        # 检查号码数量是否正确
        if len(numbers) != 20:
            logger.warning(f"跳过行（号码数量不足20个）：期号 {seq}")
            continue

        # 验证期号是否为数字
        try:
            int(seq)
        except ValueError:
            logger.warning(f"跳过行（期号非数字）：{seq}")
            continue

        # 验证号码是否都在1-80范围内
        try:
            number_list = [int(num) for num in numbers]
            if not all(1 <= num <= 80 for num in number_list):
                logger.warning(f"跳过期号 {seq}（号码超出1-80范围）")
                continue
        except ValueError:
            logger.warning(f"跳过期号 {seq}（号码包含非数字）")
            continue

        # 构建数据字典
        item = {
            '期号': seq,
            '日期': date,
            '号码': ','.join(numbers)
        }
        data.append(item)

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['期号', '日期', '号码'])

    if df.empty:
        logger.error("没有提取到任何有效数据。请检查数据格式或数据源是否可用。")
        return False
    else:
        # 将期号转换为整数以便排序
        try:
            df['期号'] = df['期号'].astype(int)
        except ValueError as e:
            logger.error(f"转换期号为整数时出错: {e}")
            return False

        # 按期号升序排序
        df.sort_values(by='期号', inplace=True)
        df.reset_index(drop=True, inplace=True)

        try:
            # 保存为CSV文件
            df.to_csv(CSV_FILE_PATH, encoding="utf-8", index=False)
            logger.info(f"数据已成功保存到 {CSV_FILE_PATH}")
            logger.info(f"共保存 {len(df)} 期数据")
            if not df.empty:
                logger.info(f"数据范围：第 {df['期号'].min()} 期 到 第 {df['期号'].max()} 期")
            return True
        except Exception as e:
            logger.error(f"保存数据时出错: {e}")
            return False

def main():
    """主函数，执行快乐8数据获取和处理"""
    logger.info("=" * 50)
    logger.info("快乐8数据处理器启动")
    logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    success = fetch_kl8_data()
    
    if success:
        logger.info("快乐8数据处理完成")
    else:
        logger.error("快乐8数据处理失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 
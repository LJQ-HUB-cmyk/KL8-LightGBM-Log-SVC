# -*- coding: utf-8 -*-
"""
快乐8彩票数据分析与推荐系统
================================

本脚本整合了统计分析、机器学习和策略化组合生成，为快乐8彩票提供数据驱动的
号码推荐。脚本支持两种运行模式，由全局变量 `ENABLE_OPTUNA_OPTIMIZATION` 控制：

1.  **分析模式 (默认 `False`)**:
    使用内置的 `DEFAULT_WEIGHTS` 权重，执行一次完整的历史数据分析、策略回测，
    并为下一期生成推荐号码。所有结果会输出到一个带时间戳的详细报告文件中。

2.  **优化模式 (`True`)**:
    在分析前，首先运行 Optuna 框架进行参数搜索，以找到在近期历史数据上
    表现最佳的一组权重。然后，自动使用这组优化后的权重来完成后续的分析、
    回测和推荐。优化过程和结果也会记录在报告中。

版本: 1.0 (KL8 Edition)
"""

# --- 标准库导入 ---
import os
import sys
import json
import time
import datetime
import logging
import io
import random
from collections import Counter
from contextlib import redirect_stdout
from typing import (Union, Optional, List, Dict, Tuple, Any)
from functools import partial

# --- 第三方库导入 ---
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import concurrent.futures

# ==============================================================================
# --- 全局常量与配置 ---
# ==============================================================================

# --------------------------
# --- 路径与模式配置 ---
# --------------------------
# 脚本文件所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始快乐8数据CSV文件路径 (由 kl8_data_processor.py 生成)
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'kuaile8.csv')
# 预处理后的数据缓存文件路径，避免每次都重新计算特征
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'kuaile8_processed.csv')

# 运行模式配置:
# True  -> 运行参数优化，耗时较长，但可能找到更优策略。
# False -> 使用默认权重进行快速分析和推荐。
ENABLE_OPTUNA_OPTIMIZATION = True

# ARM动态调整配置:
# True  -> 总是启用动态调整（原始行为）
# False -> 严格使用配置参数，不进行动态调整
# 'auto' -> 根据优化状态自动决定（推荐）
ARM_DYNAMIC_ADJUSTMENT_MODE = 'auto'

# --------------------------
# --- 策略开关配置 ---
# --------------------------
# 是否启用最终推荐组合层面的"反向思维"策略 (移除得分最高的几注)
ENABLE_FINAL_COMBO_REVERSE = True
# 在启用反向思维并移除组合后，是否从候选池中补充新的组合以达到目标数量
ENABLE_REVERSE_REFILL = True

# --------------------------
# --- 彩票规则配置 ---
# --------------------------
# 快乐8的有效号码范围 (1到80)
BALL_RANGE = range(1, 81)
# 每期开奖号码个数
NUMBERS_PER_DRAW = 20
# 支持的投注玩法 (选一到选十)
PLAY_TYPES = {
    1: "选一", 2: "选二", 3: "选三", 4: "选四", 5: "选五",
    6: "选六", 7: "选七", 8: "选八", 9: "选九", 10: "选十"
}
# 快乐8号码分区定义，用于特征工程和模式分析
NUMBER_ZONES = {
    'Zone1': (1, 20),    # 第一区: 1-20
    'Zone2': (21, 40),   # 第二区: 21-40
    'Zone3': (41, 60),   # 第三区: 41-60
    'Zone4': (61, 80)    # 第四区: 61-80
}
# 复式推荐号码数量
COMPLEX_RECOMMEND_COUNT = 20

# --------------------------
# --- 分析与执行参数配置 ---
# --------------------------
# 机器学习模型使用的滞后特征阶数 (e.g., 使用前1、3、5、10期的数据作为特征)
ML_LAG_FEATURES = [1, 3, 5, 8, 10]
# 用于生成乘积交互特征的特征对 (e.g., 和值 * 奇数个数)
ML_INTERACTION_PAIRS = [('sum', 'odd_count')]
# 用于生成自身平方交互特征的特征 (e.g., 跨度的平方)
ML_INTERACTION_SELF = ['span']
# 计算号码"近期"出现频率时所参考的期数窗口大小
RECENT_FREQ_WINDOW = 20
# 在分析模式下，进行策略回测时所评估的总期数
BACKTEST_PERIODS_COUNT = 100
# 在优化模式下，每次试验用于快速评估性能的回测期数 (数值越小优化越快)
OPTIMIZATION_BACKTEST_PERIODS = 20
# 在优化模式下，Optuna 进行参数搜索的总试验次数
OPTIMIZATION_TRIALS = 100
# 训练机器学习模型时，一个球号在历史数据中至少需要出现的次数 (防止样本过少导致模型不可靠)
MIN_POSITIVE_SAMPLES_FOR_ML = 25

# ==============================================================================
# --- 默认权重配置 (这些参数可被Optuna优化) ---
# ==============================================================================
# 这里的每一项都是一个可调整的策略参数，共同决定了最终的推荐结果。
DEFAULT_WEIGHTS = {
    # --- 反向思维 ---
    # 若启用反向思维，从最终推荐列表中移除得分最高的组合的比例
    'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT': 0.3,

    # --- 组合生成 ---
    # 最终向用户推荐的组合（注数）数量
    'NUM_COMBINATIONS_TO_GENERATE': 10,
    # 构建候选池时，从所有号码中选取分数最高的N个
    'TOP_N_FOR_CANDIDATE': 50,

    # --- 号码评分权重 ---
    # 号码历史总频率得分的权重 (降低默认值，避免过度依赖)
    'FREQ_SCORE_WEIGHT': 15.0,
    # 号码当前遗漏值（与平均遗漏的偏差）得分的权重 (提高权重)
    'OMISSION_SCORE_WEIGHT': 25.0,
    # 号码当前遗漏与其历史最大遗漏比率的得分权重
    'MAX_OMISSION_RATIO_SCORE_WEIGHT': 16.12,
    # 号码近期出现频率的得分权重 (提高权重)
    'RECENT_FREQ_SCORE_WEIGHT': 20.0,
    # 号码的机器学习模型预测出现概率的得分权重 (提高权重)
    'ML_PROB_SCORE_WEIGHT': 30.0,

    # --- 组合属性匹配奖励 ---
    # 推荐组合的奇数个数若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 13.10,
    # 推荐组合的区间分布若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_ZONE_MATCH_BONUS': 13.12,

    # --- 关联规则挖掘(ARM)参数与奖励 ---
    # ARM算法的最小支持度阈值
    'ARM_MIN_SUPPORT': 0.01,
    # ARM算法的最小置信度阈值
    'ARM_MIN_CONFIDENCE': 0.53,
    # ARM算法的最小提升度阈值
    'ARM_MIN_LIFT': 1.53,
    # 推荐组合若命中了某条挖掘出的关联规则，其获得的基础奖励分值
    'ARM_COMBINATION_BONUS_WEIGHT': 18.86,
    # 在计算ARM奖励时，规则的提升度(lift)对此奖励的贡献乘数因子
    'ARM_BONUS_LIFT_FACTOR': 0.48,
    # 在计算ARM奖励时，规则的置信度(confidence)对此奖励的贡献乘数因子
    'ARM_BONUS_CONF_FACTOR': 0.25,

    # --- 组合多样性控制 ---
    # 最终推荐的任意两注组合之间，其号码至少要有几个是不同的
    'DIVERSITY_MIN_DIFFERENT_NUMBERS': 5,
}

# ==============================================================================
# --- 机器学习模型参数配置 ---
# ==============================================================================
# 这些是 LightGBM 机器学习模型的核心超参数。
LGBM_PARAMS = {
    'objective': 'binary',              # 目标函数：二分类问题（预测一个球号是否出现）
    'boosting_type': 'gbdt',            # 提升类型：梯度提升决策树
    'learning_rate': 0.04,              # 学习率：控制每次迭代的步长
    'n_estimators': 100,                # 树的数量：总迭代次数
    'num_leaves': 15,                   # 每棵树的最大叶子节点数：控制模型复杂度
    'min_child_samples': 15,            # 一个叶子节点上所需的最小样本数：防止过拟合
    'lambda_l1': 0.15,                  # L1 正则化
    'lambda_l2': 0.15,                  # L2 正则化
    'feature_fraction': 0.7,            # 特征采样比例：每次迭代随机选择70%的特征
    'bagging_fraction': 0.8,            # 数据采样比例：每次迭代随机选择80%的数据
    'bagging_freq': 5,                  # 数据采样的频率：每5次迭代进行一次
    'seed': 42,                         # 随机种子：确保结果可复现
    'n_jobs': 1,                        # 并行线程数：设为1以在多进程环境中避免冲突
    'verbose': -1,                      # 控制台输出级别：-1表示静默
}

# ==============================================================================
# --- 日志系统配置 ---
# ==============================================================================
# 创建两种格式化器
console_formatter = logging.Formatter('%(message)s')  # 用于控制台的简洁格式
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s') # 用于文件的详细格式

# 主日志记录器
logger = logging.getLogger('kl8_analyzer')
logger.setLevel(logging.DEBUG)
logger.propagate = False # 防止日志向根记录器传递

# 进度日志记录器 (用于回测和Optuna进度条，避免被详细格式污染)
progress_logger = logging.getLogger('progress_logger')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False

# 全局控制台处理器
global_console_handler = logging.StreamHandler(sys.stdout)
global_console_handler.setFormatter(console_formatter)

# 进度专用控制台处理器
progress_console_handler = logging.StreamHandler(sys.stdout)
progress_console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(global_console_handler)
progress_logger.addHandler(progress_console_handler)

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """动态设置主日志记录器在控制台的输出级别和格式。"""
    global_console_handler.setLevel(level)
    global_console_handler.setFormatter(console_formatter if use_simple_formatter else detailed_formatter)

# ==============================================================================
# --- 核心工具函数 ---
# ==============================================================================

class SuppressOutput:
    """一个上下文管理器，用于临时抑制标准输出和捕获标准错误。"""
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout, self.capture_stderr = suppress_stdout, capture_stderr
        self.old_stdout, self.old_stderr, self.stdout_io, self.stderr_io = None, None, None, None
    def __enter__(self):
        if self.suppress_stdout: self.old_stdout, self.stdout_io, sys.stdout = sys.stdout, io.StringIO(), self.stdout_io
        if self.capture_stderr: self.old_stderr, self.stderr_io, sys.stderr = sys.stderr, io.StringIO(), self.stderr_io
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr; captured = self.stderr_io.getvalue(); self.stderr_io.close()
            if captured.strip(): logger.warning(f"在一个被抑制的输出块中捕获到标准错误:\n{captured.strip()}")
        if self.suppress_stdout and self.old_stdout:
            sys.stdout = self.old_stdout; self.stdout_io.close()
        return False # 不抑制异常

def get_prize_level(hit_count: int, play_type: int = 10) -> Optional[str]:
    """
    根据命中的号码个数和玩法类型确定奖级。

    Args:
        hit_count (int): 命中的号码个数。
        play_type (int): 玩法类型（1-10），默认为选十。

    Returns:
        Optional[str]: 奖级名称，如果未中奖则返回 None。
    """
    # 快乐8各玩法中奖规则 - 根据官方规则完整设定
    prize_rules = {
        1: {1: "一等奖"},  # 选一：中1个
        2: {2: "一等奖"},  # 选二：中2个
        3: {3: "一等奖", 2: "二等奖"},  # 选三：中3/2个
        4: {4: "一等奖", 3: "二等奖", 2: "三等奖"},  # 选四：中4/3/2个
        5: {5: "一等奖", 4: "二等奖", 3: "三等奖"},  # 选五：中5/4/3个
        6: {6: "一等奖", 5: "二等奖", 4: "三等奖", 3: "四等奖"},  # 选六：中6/5/4/3个
        7: {7: "一等奖", 6: "二等奖", 5: "三等奖", 4: "四等奖", 0: "五等奖"},  # 选七：中7/6/5/4个或全不中
        8: {8: "一等奖", 7: "二等奖", 6: "三等奖", 5: "四等奖", 4: "五等奖", 0: "六等奖"},  # 选八：中8/7/6/5/4个或全不中
        9: {9: "一等奖", 8: "二等奖", 7: "三等奖", 6: "四等奖", 5: "五等奖", 4: "六等奖", 0: "七等奖"},  # 选九：中9/8/7/6/5/4个或全不中
        10: {10: "一等奖", 9: "二等奖", 8: "三等奖", 7: "四等奖", 6: "五等奖", 5: "六等奖", 0: "七等奖"}  # 选十：中10个一等奖500万元，中9个二等奖8000元，以此类推
    }
    
    return prize_rules.get(play_type, {}).get(hit_count)

def format_time(seconds: float) -> str:
    """将秒数格式化为易于阅读的 HH:MM:SS 字符串。"""
    if seconds < 0: return "00:00:00"
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d}" 

# ==============================================================================
# --- 数据处理模块 ---
# ==============================================================================

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    从CSV文件加载数据，并能自动尝试多种常用编码格式。

    Args:
        file_path (str): CSV文件的路径。

    Returns:
        Optional[pd.DataFrame]: 加载成功的DataFrame，如果文件不存在或无法解码则返回None。
    """
    if not os.path.exists(file_path):
        logger.error(f"数据文件未找到: {file_path}")
        return None
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"使用编码 {enc} 加载 {file_path} 时出错: {e}")
            return None
    logger.error(f"无法使用任何支持的编码打开文件 {file_path}。"); return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    清洗和结构化原始DataFrame，确保数据类型正确，并转换为"一行一期"的格式。

    Args:
        df (pd.DataFrame): 从CSV原始加载的DataFrame。

    Returns:
        Optional[pd.DataFrame]: 清洗和结构化后的DataFrame，如果输入无效或处理失败则返回None。
    """
    if df is None or df.empty: return None
    required_cols = ['期号', '号码']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"输入数据缺少必要列: {required_cols}")
        return None

    df.dropna(subset=required_cols, inplace=True)
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce')
        df.dropna(subset=['期号'], inplace=True)
        df = df.astype({'期号': int})
    except (ValueError, TypeError) as e:
        logger.error(f"转换'期号'为整数时失败: {e}"); return None

    df.sort_values(by='期号', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    parsed_rows = []
    for _, row in df.iterrows():
        try:
            # 解析号码，并进行严格验证
            numbers = sorted([int(n) for n in str(row['号码']).split(',')])
            if len(numbers) != NUMBERS_PER_DRAW or not all(n in BALL_RANGE for n in numbers):
                logger.warning(f"期号 {row['期号']} 的数据无效，已跳过: 号码={numbers}")
                continue
            
            # 构建结构化的记录
            record = {'期号': row['期号'], **{f'num{i+1}': n for i, n in enumerate(numbers)}}
            if '日期' in row and pd.notna(row['日期']):
                record['日期'] = row['日期']
            parsed_rows.append(record)
        except (ValueError, TypeError):
            logger.warning(f"解析期号 {row['期号']} 的号码时失败，已跳过。")
            continue
            
    return pd.DataFrame(parsed_rows) if parsed_rows else None

def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    为DataFrame计算各种衍生特征，如和值、跨度、奇偶比、区间分布等。

    Args:
        df (pd.DataFrame): 经过清洗和结构化后的DataFrame。

    Returns:
        Optional[pd.DataFrame]: 包含新计算特征的DataFrame。
    """
    if df is None or df.empty: return None
    df_fe = df.copy()
    num_cols = [f'num{i+1}' for i in range(NUMBERS_PER_DRAW)]
    
    # 基本统计特征
    df_fe['sum'] = df_fe[num_cols].sum(axis=1)
    df_fe['span'] = df_fe[num_cols].max(axis=1) - df_fe[num_cols].min(axis=1)
    df_fe['odd_count'] = df_fe[num_cols].apply(lambda r: sum(x % 2 != 0 for x in r), axis=1)
    df_fe['mean'] = df_fe[num_cols].mean(axis=1)
    df_fe['std'] = df_fe[num_cols].std(axis=1)
    
    # 区间特征
    for zone, (start, end) in NUMBER_ZONES.items():
        df_fe[f'{zone}_count'] = df_fe[num_cols].apply(lambda r: sum(start <= x <= end for x in r), axis=1)
        
    # 形态特征
    def count_consecutive(row): 
        sorted_nums = sorted(row)
        return sum(1 for i in range(len(sorted_nums)-1) if sorted_nums[i+1] - sorted_nums[i] == 1)
    df_fe['consecutive_count'] = df_fe[num_cols].apply(count_consecutive, axis=1)
    
    # 重号特征 (与上一期的重复个数)
    num_sets = df_fe[num_cols].apply(set, axis=1)
    prev_num_sets = num_sets.shift(1)
    df_fe['repeat_count'] = [len(current.intersection(prev)) if isinstance(prev, set) else 0 for current, prev in zip(num_sets, prev_num_sets)]
    
    return df_fe

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """
    为机器学习模型创建滞后特征（将历史期的特征作为当前期的输入）和交互特征。

    Args:
        df (pd.DataFrame): 包含基础特征的DataFrame。
        lags (List[int]): 滞后阶数列表, e.g., [1, 3, 5]。

    Returns:
        Optional[pd.DataFrame]: 一个只包含滞后和交互特征的DataFrame。
    """
    if df is None or df.empty or not lags: return None
    
    feature_cols = [col for col in df.columns if col in ['sum', 'span', 'odd_count', 'mean', 'std', 'consecutive_count', 'repeat_count'] or 'Zone' in col]
    df_features = df[feature_cols].copy()
    
    # 创建交互特征
    for c1, c2 in ML_INTERACTION_PAIRS:
        if c1 in df_features and c2 in df_features: df_features[f'{c1}_x_{c2}'] = df_features[c1] * df_features[c2]
    for c in ML_INTERACTION_SELF:
        if c in df_features: df_features[f'{c}_sq'] = df_features[c]**2
        
    # 创建滞后特征
    all_feature_cols = df_features.columns.tolist()
    lagged_dfs = [df_features[all_feature_cols].shift(lag).add_suffix(f'_lag{lag}') for lag in lags]
    final_df = pd.concat(lagged_dfs, axis=1)
    final_df.dropna(inplace=True)
    
    return final_df if not final_df.empty else None 

# ==============================================================================
# --- 分析与评分模块 ---
# ==============================================================================

def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """
    分析所有号码的频率、当前遗漏、平均遗漏、最大遗漏和近期频率。

    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。

    Returns:
        dict: 包含各种频率和遗漏统计信息的字典。
    """
    if df is None or df.empty: return {}
    num_cols, total_periods = [f'num{i+1}' for i in range(NUMBERS_PER_DRAW)], len(df)
    most_recent_idx = total_periods - 1
    
    # 频率计算
    all_nums_flat = df[num_cols].values.flatten()
    num_freq = Counter(all_nums_flat)
    
    # 遗漏和近期频率计算
    current_omission, max_hist_omission, recent_N_freq = {}, {}, Counter()
    
    for num in BALL_RANGE:
        app_indices = df.index[(df[num_cols] == num).any(axis=1)].tolist()
        if app_indices:
            current_omission[num] = most_recent_idx - app_indices[-1]
            gaps = np.diff([0] + app_indices) - 1 # 包含从开始到第一次出现的遗漏
            max_hist_omission[num] = max(gaps.max(), current_omission[num])
        else:
            current_omission[num] = max_hist_omission[num] = total_periods
            
    # 计算近期频率
    if total_periods >= RECENT_FREQ_WINDOW:
        recent_N_freq.update(df.tail(RECENT_FREQ_WINDOW)[num_cols].values.flatten())
        
    # 平均间隔（理论遗漏）
    avg_interval = {num: total_periods / (num_freq.get(num, 0) + 1e-9) for num in BALL_RANGE}
        
    return {
        'freq': num_freq,
        'current_omission': current_omission, 
        'average_interval': avg_interval,
        'max_historical_omission': max_hist_omission,
        'recent_N_freq': recent_N_freq
    }

def analyze_patterns(df: pd.DataFrame) -> dict:
    """
    分析历史数据中的常见模式，如最常见的和值、奇偶比、区间分布等。

    Args:
        df (pd.DataFrame): 包含特征工程后历史数据的DataFrame。

    Returns:
        dict: 包含最常见模式的字典。
    """
    if df is None or df.empty: return {}
    res = {}
    def safe_mode(s): return s.mode().iloc[0] if not s.empty and not s.mode().empty else None
    
    for col, name in [('sum', 'sum'), ('span', 'span'), ('odd_count', 'odd_count'), ('mean', 'mean')]:
        if col in df.columns: res[f'most_common_{name}'] = safe_mode(df[col])
        
    zone_cols = [f'{zone}_count' for zone in NUMBER_ZONES.keys()]
    if all(c in df.columns for c in zone_cols):
        dist_counts = df[zone_cols].apply(tuple, axis=1).value_counts()
        if not dist_counts.empty: res['most_common_zone_distribution'] = dist_counts.index[0]
    
    return res

def analyze_associations(df: pd.DataFrame, weights_config: Dict, enable_dynamic_adjustment: bool = True) -> pd.DataFrame:
    """
    使用Apriori算法挖掘号码之间的关联规则（例如，哪些号码倾向于一起出现）。
    添加内存保护机制，防止内存溢出。

    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。
        weights_config (Dict): 包含ARM算法参数(min_support, min_confidence, min_lift)的字典。
        enable_dynamic_adjustment (bool): 是否启用动态调整机制。在优化阶段应设为False。

    Returns:
        pd.DataFrame: 一个包含挖掘出的强关联规则的DataFrame。
    """
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.01)
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.5)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.5)
    num_cols = [f'num{i+1}' for i in range(NUMBERS_PER_DRAW)]
    if df is None or df.empty: return pd.DataFrame()
    
    try:
        # 内存保护：限制数据量
        max_rows = 1000  # 限制最大分析行数
        if len(df) > max_rows:
            df_sample = df.tail(max_rows)  # 使用最近的数据
            logger.info(f"数据量过大，使用最近{max_rows}期数据进行关联规则分析")
        else:
            df_sample = df
            
        transactions = df_sample[num_cols].astype(str).values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_oh = pd.DataFrame(te_ary, columns=te.columns_)
        
        # 动态调整支持度阈值以控制内存使用
        adjusted_min_s = min_s
        num_items = len(te.columns_)
        
        if enable_dynamic_adjustment:
            # 只在实际执行时启用动态调整
            adjusted_min_s = max(min_s, 0.05)  # 至少5%支持度
            if num_items > 60:  # 如果唯一项目过多，进一步提高阈值
                adjusted_min_s = max(adjusted_min_s, 0.1)
                logger.info(f"唯一号码数量({num_items})较多，调整最小支持度至{adjusted_min_s}")
        else:
            # 优化阶段：如果参数过小且项目过多，给出警告但不强制调整
            if min_s < 0.05 and num_items > 60:
                logger.warning(f"警告：支持度{min_s:.4f}较低且唯一号码数量({num_items})较多，可能导致内存不足")
            # 设置一个最小的安全阈值，避免内存溢出
            adjusted_min_s = max(min_s, 0.01)
            
        frequent_itemsets = apriori(df_oh, min_support=adjusted_min_s, use_colnames=True, max_len=3)  # 限制最大长度
        if frequent_itemsets.empty: 
            logger.warning("未找到满足条件的频繁项集")
            return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_l)
        strong_rules = rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
        
        if not strong_rules.empty:
            logger.info(f"成功挖掘到{len(strong_rules)}条关联规则 (支持度阈值: {adjusted_min_s:.4f})")
        return strong_rules
        
    except MemoryError as e:
        logger.error(f"关联规则分析内存不足: {e}，跳过此分析")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"关联规则分析失败: {e}")
        return pd.DataFrame()

def calculate_scores(freq_data: Dict, probabilities: Dict, weights: Dict) -> Dict[str, Dict[int, float]]:
    """
    根据所有分析结果（频率、遗漏、ML预测），使用加权公式计算每个球的最终推荐分数。

    Args:
        freq_data (Dict): 来自 `analyze_frequency_omission` 的频率和遗漏分析结果。
        probabilities (Dict): 来自机器学习模型的预测概率。
        weights (Dict): 包含所有评分权重的配置字典。

    Returns:
        Dict[str, Dict[int, float]]: 包含号码归一化后分数的字典。
    """
    scores = {}
    freq = freq_data.get('freq', {})
    omission = freq_data.get('current_omission', {})
    avg_int = freq_data.get('average_interval', {})
    max_hist_o = freq_data.get('max_historical_omission', {})
    recent_freq = freq_data.get('recent_N_freq', {})
    pred_probs = probabilities.get('numbers', {})
    
    # 号码评分
    for num in BALL_RANGE:
        # 频率分：出现次数越多，得分越高
        freq_s = (freq.get(num, 0)) * weights['FREQ_SCORE_WEIGHT']
        # 遗漏分：当前遗漏接近平均遗漏时得分最高，过冷或过热都会降低分数
        omit_s = np.exp(-0.005 * (omission.get(num, 0) - avg_int.get(num, 0))**2) * weights['OMISSION_SCORE_WEIGHT']
        # 最大遗漏比率分：当前遗漏接近或超过历史最大遗漏时得分高（博冷）
        max_o_ratio = (omission.get(num, 0) / max_hist_o.get(num, 1)) if max_hist_o.get(num, 0) > 0 else 0
        max_o_s = max_o_ratio * weights['MAX_OMISSION_RATIO_SCORE_WEIGHT']
        # 近期频率分：近期出现次数越多，得分越高（追热）
        recent_s = recent_freq.get(num, 0) * weights['RECENT_FREQ_SCORE_WEIGHT']
        # ML预测分
        ml_s = pred_probs.get(num, 0.0) * weights['ML_PROB_SCORE_WEIGHT']
        scores[num] = sum([freq_s, omit_s, max_o_s, recent_s, ml_s])

    # 归一化所有分数到0-100范围，便于比较
    def normalize_scores(scores_dict):
        if not scores_dict: return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: return {k: 50.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) * 100 for k, v in scores_dict.items()}

    return {'number_scores': normalize_scores(scores)} 

# ==============================================================================
# --- 机器学习模块 ---
# ==============================================================================

def train_single_lgbm_model(ball_number: int, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Optional[LGBMClassifier], Optional[str]]:
    """为单个球号训练一个LGBM二分类模型（预测它是否出现）。"""
    if y_train.sum() < MIN_POSITIVE_SAMPLES_FOR_ML or y_train.nunique() < 2:
        return None, None # 样本不足或只有一类，无法训练
        
    model_key = f'lgbm_{ball_number}'
    model_params = LGBM_PARAMS.copy()
    
    # 类别不平衡处理：给样本量较少的类别（中奖）更高的权重
    if (pos_count := y_train.sum()) > 0:
        model_params['scale_pos_weight'] = (len(y_train) - pos_count) / pos_count
        
    try:
        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        return model, model_key
    except Exception as e:
        logger.debug(f"训练LGBM for 球号 {ball_number} 失败: {e}")
        return None, None

def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int]) -> Optional[Dict[str, Any]]:
    """为所有号码并行训练预测模型。"""
    if (X := create_lagged_features(df_train_raw.copy(), ml_lags_list)) is None or X.empty:
        logger.warning("创建滞后特征失败或结果为空，跳过模型训练。")
        return None
        
    if (target_df := df_train_raw.loc[X.index].copy()).empty: return None
    
    num_cols = [f'num{i+1}' for i in range(NUMBERS_PER_DRAW)]
    trained_models = {'numbers': {}, 'feature_cols': X.columns.tolist()}
    
    # 使用进程池并行训练，加快速度
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        # 为每个号码提交训练任务
        for ball_num in BALL_RANGE:
            y = target_df[num_cols].eq(ball_num).any(axis=1).astype(int)
            future = executor.submit(train_single_lgbm_model, ball_num, X, y)
            futures[future] = ball_num
            
        for future in concurrent.futures.as_completed(futures):
            ball_num = futures[future]
            try:
                model, model_key = future.result()
                if model and model_key:
                    trained_models['numbers'][model_key] = model
            except Exception as e:
                logger.error(f"训练球号 {ball_num} 的模型时出现异常: {e}")

    return trained_models if trained_models['numbers'] else None

def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[Dict], ml_lags_list: List[int]) -> Dict[str, Dict[int, float]]:
    """使用训练好的模型预测下一期每个号码的出现概率。"""
    probs = {'numbers': {}}
    if not trained_models or not (feat_cols := trained_models.get('feature_cols')):
        return probs
        
    max_lag = max(ml_lags_list) if ml_lags_list else 0
    if len(df_historical) < max_lag + 1:
        return probs # 数据不足以创建预测所需的特征
        
    if (predict_X := create_lagged_features(df_historical.tail(max_lag + 1), ml_lags_list)) is None:
        return probs
        
    predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
    
    for ball_num in BALL_RANGE:
        if (model := trained_models.get('numbers', {}).get(f'lgbm_{ball_num}')):
            try:
                # 预测类别为1（出现）的概率
                probs['numbers'][ball_num] = model.predict_proba(predict_X)[0, 1]
            except Exception:
                pass
    return probs

# ==============================================================================
# --- 组合生成与策略应用模块 ---
# ==============================================================================

def generate_play_type_recommendations(scores_data: Dict, pattern_data: Dict, arm_rules: pd.DataFrame, weights_config: Dict) -> Tuple[Dict, List[str]]:
    """为所有玩法（选一到选十）生成推荐组合。"""
    num_scores = scores_data.get('number_scores', {})
    if not num_scores: 
        return {}, ["无法生成推荐 (分数数据缺失)。"]

    # 构建候选池
    top_n = int(weights_config['TOP_N_FOR_CANDIDATE'])
    cand_pool = [n for n, _ in sorted(num_scores.items(), key=lambda i: i[1], reverse=True)[:top_n]]
    
    if len(cand_pool) < 10:
        return {}, ["候选池号码不足。"]

    # 为每种玩法生成推荐
    recommendations = {}
    output_strs = []
    
    for play_type in range(1, 11):  # 选一到选十
        play_name = PLAY_TYPES[play_type]
        
        # 根据玩法选择号码数量
        num_to_select = play_type
        
        if len(cand_pool) < num_to_select:
            continue
            
        # 使用加权随机选择生成推荐
        weights_array = np.array([num_scores.get(n, 0) + 1 for n in cand_pool])
        probs = weights_array / weights_array.sum() if weights_array.sum() > 0 else None
        
        # 生成多个候选组合，选择最佳的
        best_combo = None
        best_score = -1
        
        for _ in range(100):  # 尝试100次
            try:
                if probs is not None:
                    numbers = sorted(np.random.choice(cand_pool, size=num_to_select, replace=False, p=probs).tolist())
                else:
                    numbers = sorted(random.sample(cand_pool, num_to_select))
                
                # 计算组合得分
                combo_score = sum(num_scores.get(n, 0) for n in numbers)
                
                if combo_score > best_score:
                    best_score = combo_score
                    best_combo = numbers
                    
            except Exception:
                continue
                
        if best_combo:
            recommendations[play_type] = {
                'play_name': play_name,
                'numbers': best_combo,
                'score': best_score
            }
    
    # 生成复式推荐（前20个高分号码）
    complex_numbers = [n for n, _ in sorted(num_scores.items(), key=lambda i: i[1], reverse=True)[:COMPLEX_RECOMMEND_COUNT]]
    recommendations['complex'] = {
        'play_name': '复式推荐',
        'numbers': complex_numbers,
        'score': sum(num_scores.get(n, 0) for n in complex_numbers)
    }
    
    # 生成输出字符串
    output_strs.append("--- 各玩法推荐号码 ---")
    for play_type in range(1, 11):
        if play_type in recommendations:
            rec = recommendations[play_type]
            num_str = ' '.join(f'{n:02d}' for n in rec['numbers'])
            output_strs.append(f"{rec['play_name']}: [{num_str}] (得分: {rec['score']:.2f})")
    
    output_strs.append(f"\n--- 复式推荐号码池 ---")
    complex_rec = recommendations['complex']
    complex_str = ' '.join(f'{n:02d}' for n in complex_rec['numbers'])
    output_strs.append(f"复式推荐: [{complex_str}]")
    
    return recommendations, output_strs

def generate_combinations(scores_data: Dict, pattern_data: Dict, arm_rules: pd.DataFrame, weights_config: Dict) -> Tuple[List[Dict], List[str]]:
    """为向后兼容保留的旧版本组合生成函数（选十玩法）。"""
    # 调用新函数并返回选十的结果
    recommendations, _ = generate_play_type_recommendations(scores_data, pattern_data, arm_rules, weights_config)
    
    if 10 in recommendations:
        rec = recommendations[10]
        final_recs = [{'combination': {'numbers': rec['numbers']}, 'score': rec['score'], 'number_tuple': tuple(rec['numbers'])}]
        output_strs = [f"选十推荐: [{' '.join(f'{n:02d}' for n in rec['numbers'])}] (得分: {rec['score']:.2f})"]
        return final_recs, output_strs
    else:
        return [], ["无法生成选十推荐。"]

# ==============================================================================
# --- 核心分析与回测流程 ---
# ==============================================================================

def run_analysis_and_recommendation(df_hist: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame) -> Tuple:
    """
    执行一次完整的分析和推荐流程，用于特定一期。

    Returns:
        tuple: 包含推荐组合、输出字符串、分析摘要、训练模型和分数的元组。
    """
    freq_data = analyze_frequency_omission(df_hist)
    patt_data = analyze_patterns(df_hist)
    ml_models = train_prediction_models(df_hist, ml_lags)
    probabilities = predict_next_draw_probabilities(df_hist, ml_models, ml_lags) if ml_models else {'numbers': {}}
    scores = calculate_scores(freq_data, probabilities, weights_config)
    
    # 生成所有玩法的推荐
    all_play_recs, all_play_strings = generate_play_type_recommendations(scores, patt_data, arm_rules, weights_config)
    
    # 为了向后兼容，也生成选十的推荐
    old_recs, old_rec_strings = generate_combinations(scores, patt_data, arm_rules, weights_config)
    
    analysis_summary = {'frequency_omission': freq_data, 'patterns': patt_data}
    return all_play_recs, all_play_strings, analysis_summary, ml_models, scores

def run_backtest(full_df: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame, num_periods: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    在历史数据上执行策略回测，以评估策略表现。支持多种玩法验证。

    Returns:
        tuple: 包含详细回测结果的DataFrame和统计摘要的字典。
    """
    min_data_needed = (max(ml_lags) if ml_lags else 0) + MIN_POSITIVE_SAMPLES_FOR_ML + num_periods
    if len(full_df) < min_data_needed:
        logger.error(f"数据不足以回测{num_periods}期。需要至少{min_data_needed}期，当前有{len(full_df)}期。")
        return pd.DataFrame(), {}

    start_idx = len(full_df) - num_periods
    results, prize_counts_by_play, num_cols = [], {}, [f'num{i+1}' for i in range(NUMBERS_PER_DRAW)]
    best_hits_per_period = []
    
    # 初始化每种玩法的奖级统计
    for play_type in range(1, 11):
        prize_counts_by_play[play_type] = Counter()
    
    logger.info("策略回测已启动...")
    start_time = time.time()
    
    for i in range(num_periods):
        current_iter = i + 1
        current_idx = start_idx + i
        
        # 使用SuppressOutput避免在回测循环中打印大量日志
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
            hist_data = full_df.iloc[:current_idx]
            predicted_combos, _, _, _, _ = run_analysis_and_recommendation(hist_data, ml_lags, weights_config, arm_rules)
            
        actual_outcome = full_df.loc[current_idx]
        actual_num_set = set(actual_outcome[num_cols])
        
        period_best_hits = {}
        
        if predicted_combos:
            # 验证每种玩法的推荐
            for play_type in range(1, 11):
                if play_type in predicted_combos:
                    rec_data = predicted_combos[play_type]
                    recommended_numbers = rec_data['numbers']
                    hits = len(set(recommended_numbers) & actual_num_set)
                    prize = get_prize_level(hits, play_type)
                    
                    if prize:
                        prize_counts_by_play[play_type][prize] += 1
                    
                    results.append({
                        'period': actual_outcome['期号'], 
                        'play_type': play_type,
                        'play_name': PLAY_TYPES[play_type],
                        'hits': hits, 
                        'prize': prize,
                        'recommended': ','.join(map(str, recommended_numbers))
                    })
                    
                    period_best_hits[play_type] = hits

        # 记录本期各玩法的最佳表现
        best_hits_per_period.append({
            'period': actual_outcome['期号'], 
            **{f'best_hits_play_{pt}': period_best_hits.get(pt, 0) for pt in range(1, 11)}
        })

        # 打印进度
        if current_iter == 1 or current_iter % 10 == 0 or current_iter == num_periods:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_iter
            remaining_time = avg_time * (num_periods - current_iter)
            progress_logger.info(f"回测进度: {current_iter}/{num_periods} | 平均耗时: {avg_time:.2f}s/期 | 预估剩余: {format_time(remaining_time)}")
            
    return pd.DataFrame(results), {
        'prize_counts_by_play': {k: dict(v) for k, v in prize_counts_by_play.items()}, 
        'best_hits_per_period': pd.DataFrame(best_hits_per_period)
    }

# ==============================================================================
# --- Optuna 参数优化模块 ---
# ==============================================================================

def objective(trial: optuna.trial.Trial, df_for_opt: pd.DataFrame, ml_lags: List[int], arm_rules: pd.DataFrame) -> float:
    """Optuna 的目标函数，用于评估一组给定的权重参数的好坏。"""
    trial_weights = {}
    
    # 动态地从DEFAULT_WEIGHTS构建搜索空间，但限制某些参数的范围
    for key, value in DEFAULT_WEIGHTS.items():
        if isinstance(value, int):
            if 'NUM_COMBINATIONS' in key: trial_weights[key] = trial.suggest_int(key, 5, 15)
            elif 'TOP_N' in key: trial_weights[key] = trial.suggest_int(key, 30, 60)
            else: trial_weights[key] = trial.suggest_int(key, max(0, value - 2), value + 2)
        elif isinstance(value, float):
            # 对不同类型的浮点数使用不同的搜索范围
            if any(k in key for k in ['PERCENT', 'FACTOR', 'CONFIDENCE']):
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 1.5)
            elif 'SUPPORT' in key:
                # ARM支持度特殊处理：避免优化出无法使用的极小值
                trial_weights[key] = trial.suggest_float(key, max(0.01, value * 0.5), min(0.1, value * 2.0))
            elif key == 'FREQ_SCORE_WEIGHT':
                # 严格限制历史频率权重的上限，防止过度依赖
                trial_weights[key] = trial.suggest_float(key, 5.0, 18.0)  # 进一步限制上限到18
            elif key in ['OMISSION_SCORE_WEIGHT', 'ML_PROB_SCORE_WEIGHT']:
                # 对遗漏和ML权重给予更大的搜索空间
                trial_weights[key] = trial.suggest_float(key, value * 1.0, value * 4.0)  # 扩大搜索范围
            else: # 对其他权重参数使用较宽的搜索范围
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 2.0)

    full_trial_weights = DEFAULT_WEIGHTS.copy()
    full_trial_weights.update(trial_weights)
    
    # 重新生成关联规则以确保使用当前试验的ARM参数
    # 注意：这里禁用动态调整，使用优化的参数
    trial_arm_rules = analyze_associations(df_for_opt, full_trial_weights, enable_dynamic_adjustment=False)
    
    # 在快速回测中评估这组权重
    with SuppressOutput():
        backtest_results_df, backtest_stats = run_backtest(df_for_opt, ml_lags, full_trial_weights, trial_arm_rules, OPTIMIZATION_BACKTEST_PERIODS)
        
        # 同时生成一个推荐样本来评估多样性
        sample_recs, _, _, _, sample_scores = run_analysis_and_recommendation(
            df_for_opt.iloc[:-5], ml_lags, full_trial_weights, trial_arm_rules
        )
        
    # 定义一个分数来衡量表现，根据快乐8真实奖金价值设定权重
    # 权重设计思路：按奖金价值的对数分布设定，避免高额奖项过度影响优化方向
    prize_weights = {
        '一等奖': 100,    # 最高权重，但不过度放大
        '二等奖': 50,     # 中等权重  
        '三等奖': 25,     # 中等权重
        '四等奖': 10,     # 较低权重
        '五等奖': 5,      # 低权重
        '六等奖': 3,      # 最低权重
        '七等奖': 2       # 选九全不中的权重
    }
    
    # 计算基础得分（回测表现）
    base_score = 0
    prize_counts_by_play = backtest_stats.get('prize_counts_by_play', {})
    for play_type, prize_counts in prize_counts_by_play.items():
        for prize_level, count in prize_counts.items():
            base_score += prize_weights.get(prize_level, 0) * count
    
    # 计算多样性惩罚分数 - 强化版本
    diversity_penalty = 0
    if sample_scores and sample_scores.get('number_scores'):
        # 加载历史频率数据
        freq_data = analyze_frequency_omission(df_for_opt)
        historical_freq = freq_data.get('freq', {})
        
        if historical_freq:
            # 获取历史频率TOP 15号码（缩小范围）
            top_freq_numbers = set([n for n, _ in sorted(historical_freq.items(), key=lambda x: x[1], reverse=True)[:15]])
            
            # 检查推荐的选十号码
            if 10 in sample_recs:
                recommended_numbers = set(sample_recs[10]['numbers'])
                overlap = len(recommended_numbers & top_freq_numbers)
                overlap_ratio = overlap / len(recommended_numbers)
                
                # 如果与历史频率TOP15的重合度超过50%，给予重惩罚
                if overlap_ratio > 0.5:
                    diversity_penalty = (overlap_ratio - 0.5) * 1000  # 最大惩罚500分
                # 即使只有40%重合度也要轻微惩罚
                elif overlap_ratio > 0.4:
                    diversity_penalty = (overlap_ratio - 0.4) * 500  # 最大惩罚50分
            
            # 检查复式推荐的前10个号码 - 这是关键！
            if 'complex' in sample_recs:
                complex_top10 = set(sample_recs['complex']['numbers'][:10])
                complex_overlap = len(complex_top10 & top_freq_numbers)
                complex_overlap_ratio = complex_overlap / 10
                
                # 如果复式推荐前10与历史频率TOP15重合度超过60%，给予重惩罚  
                if complex_overlap_ratio > 0.6:
                    diversity_penalty += (complex_overlap_ratio - 0.6) * 1500  # 最大惩罚600分
                # 即使只有50%重合度也要轻微惩罚
                elif complex_overlap_ratio > 0.5:
                    diversity_penalty += (complex_overlap_ratio - 0.5) * 800  # 最大惩罚80分
    
    # 权重平衡奖励：鼓励更平衡的权重分配
    balance_bonus = 0
    freq_weight = full_trial_weights.get('FREQ_SCORE_WEIGHT', 0)
    omission_weight = full_trial_weights.get('OMISSION_SCORE_WEIGHT', 0)
    ml_weight = full_trial_weights.get('ML_PROB_SCORE_WEIGHT', 0)
    
    # 如果遗漏权重或ML权重超过频率权重，给予奖励
    if omission_weight > freq_weight:
        balance_bonus += 50
    if ml_weight > freq_weight:
        balance_bonus += 50
    
    # 如果频率权重过高（超过总权重的40%），给予惩罚
    total_main_weights = freq_weight + omission_weight + ml_weight + full_trial_weights.get('RECENT_FREQ_SCORE_WEIGHT', 0)
    if total_main_weights > 0:
        freq_ratio = freq_weight / total_main_weights
        if freq_ratio > 0.4:
            balance_bonus -= (freq_ratio - 0.4) * 200  # 频率权重占比惩罚
    
    # 最终得分 = 基础得分 - 多样性惩罚 + 平衡奖励
    final_score = base_score - diversity_penalty + balance_bonus
    
    return final_score

def optuna_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial, total_trials: int):
    """Optuna 的回调函数，用于在控制台报告优化进度。"""
    global OPTUNA_START_TIME
    current_iter = trial.number + 1
    if current_iter == 1 or current_iter % 10 == 0 or current_iter == total_trials:
        elapsed = time.time() - OPTUNA_START_TIME
        avg_time = elapsed / current_iter
        remaining_time = avg_time * (total_trials - current_iter)
        best_value = f"{study.best_value:.2f}" if study.best_trial else "N/A"
        progress_logger.info(f"Optuna进度: {current_iter}/{total_trials} | 当前最佳得分: {best_value} | 预估剩余: {format_time(remaining_time)}")

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    # 1. 初始化日志记录器，同时输出到控制台和文件
    log_filename = os.path.join(SCRIPT_DIR, f"kl8_analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    set_console_verbosity(logging.INFO, use_simple_formatter=True)

    logger.info("--- 快乐8数据分析与推荐系统 ---")
    logger.info("启动数据加载和预处理...")

    # 2. 智能数据加载逻辑 - 检查数据是否需要更新
    main_df = None
    need_regenerate = False
    
    # 检查原始数据文件是否存在
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"原始数据文件不存在: {CSV_FILE_PATH}")
        logger.error("请先运行 kl8_data_processor.py 获取最新数据。")
        sys.exit(1)
    
    # 检查是否需要重新生成缓存
    if os.path.exists(PROCESSED_CSV_PATH):
        # 比较文件修改时间
        raw_mtime = os.path.getmtime(CSV_FILE_PATH)
        processed_mtime = os.path.getmtime(PROCESSED_CSV_PATH)
        
        if raw_mtime > processed_mtime:
            logger.info("检测到原始数据已更新，需要重新生成预处理数据...")
            need_regenerate = True
        else:
            # 尝试加载缓存并验证数据完整性
            main_df = load_data(PROCESSED_CSV_PATH)
            if main_df is not None and not main_df.empty:
                # 验证缓存数据是否完整
                raw_df_temp = load_data(CSV_FILE_PATH)
                if raw_df_temp is not None:
                    raw_periods = len(raw_df_temp)
                    processed_periods = len(main_df)
                    
                    if raw_periods != processed_periods:
                        logger.info(f"缓存数据期数({processed_periods})与原始数据期数({raw_periods})不匹配，需要重新生成...")
                        need_regenerate = True
                    else:
                        logger.info(f"从缓存文件加载预处理数据成功，共 {processed_periods} 期数据。")
                else:
                    need_regenerate = True
            else:
                logger.info("缓存文件损坏或为空，需要重新生成...")
                need_regenerate = True
    else:
        logger.info("未找到预处理缓存文件，正在从原始文件生成...")
        need_regenerate = True
    
    # 如果需要重新生成或缓存无效，则重新处理数据
    if need_regenerate or main_df is None or main_df.empty:
        logger.info("正在重新处理原始数据...")
        raw_df = load_data(CSV_FILE_PATH)
        if raw_df is not None and not raw_df.empty:
            logger.info(f"原始数据加载成功，共 {len(raw_df)} 期，开始清洗...")
            cleaned_df = clean_and_structure(raw_df)
            if cleaned_df is not None and not cleaned_df.empty:
                logger.info("数据清洗成功，开始特征工程...")
                main_df = feature_engineer(cleaned_df)
                if main_df is not None and not main_df.empty:
                    logger.info("特征工程成功，保存预处理数据...")
                    try:
                        main_df.to_csv(PROCESSED_CSV_PATH, index=False)
                        logger.info(f"预处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except IOError as e:
                        logger.error(f"保存预处理数据失败: {e}")
                else:
                    logger.error("特征工程失败，无法生成最终数据集。")
            else:
                logger.error("数据清洗失败。")
        else:
            logger.error("原始数据加载失败。")
    
    if main_df is None or main_df.empty:
        logger.critical("数据准备失败，无法继续。请检查 'kl8_data_processor.py' 是否已成功运行并生成 'kuaile8.csv'。程序终止。")
        sys.exit(1)
    
    logger.info(f"数据加载完成，共 {len(main_df)} 期有效数据。")
    last_period = main_df['期号'].iloc[-1]

    # 3. 根据模式执行：优化或直接分析
    active_weights = DEFAULT_WEIGHTS.copy()
    optuna_summary = None

    if ENABLE_OPTUNA_OPTIMIZATION:
        logger.info("\n" + "="*25 + " Optuna 参数优化模式 " + "="*25)
        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        
        # 优化前先进行一次全局关联规则分析（使用默认参数）
        # 注意：优化过程中会重新生成关联规则，这里只是为了初始化
        optuna_arm_rules = analyze_associations(main_df, DEFAULT_WEIGHTS, enable_dynamic_adjustment=False)
        
        study = optuna.create_study(direction="maximize")
        global OPTUNA_START_TIME; OPTUNA_START_TIME = time.time()
        progress_callback_with_total = partial(optuna_progress_callback, total_trials=OPTIMIZATION_TRIALS)
        
        try:
            study.optimize(lambda t: objective(t, main_df, ML_LAG_FEATURES, optuna_arm_rules), n_trials=OPTIMIZATION_TRIALS, callbacks=[progress_callback_with_total])
            logger.info("Optuna 优化完成。")
            active_weights.update(study.best_params)
            optuna_summary = {"status": "完成", "best_value": study.best_value, "best_params": study.best_params}
        except Exception as e:
            logger.error(f"Optuna 优化过程中断: {e}", exc_info=True)
            optuna_summary = {"status": "中断", "error": str(e)}
            logger.warning("优化中断，将使用默认权重继续分析。")
    
    # 4. 切换到报告模式并打印报告头
    report_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(report_formatter)
    global_console_handler.setFormatter(report_formatter)
    
    logger.info("\n\n" + "="*60 + f"\n{' ' * 22}快乐8策略分析报告\n" + "="*60)
    logger.info(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"分析基于数据: 截至 {last_period} 期 (共 {len(main_df)} 期)")
    logger.info(f"本次预测目标: 第 {last_period + 1} 期")
    logger.info(f"日志文件: {os.path.basename(log_filename)}")

    # 5. 决定是否启用动态调整（需要在打印优化摘要前定义）
    # 根据配置决定是否启用动态调整
    if ARM_DYNAMIC_ADJUSTMENT_MODE == 'auto':
        # 自动模式：如果使用优化参数则不调整，使用默认参数则调整
        use_dynamic_adjustment = not ENABLE_OPTUNA_OPTIMIZATION or optuna_summary is None or optuna_summary.get("status") != "完成"
    else:
        # 手动模式：直接使用配置值
        use_dynamic_adjustment = ARM_DYNAMIC_ADJUSTMENT_MODE

    # 6. 打印优化摘要
    if ENABLE_OPTUNA_OPTIMIZATION and optuna_summary:
        logger.info("\n" + "="*25 + " Optuna 优化摘要 " + "="*25)
        logger.info(f"优化状态: {optuna_summary['status']}")
        if optuna_summary['status'] == '完成':
            logger.info(f"最佳性能得分: {optuna_summary['best_value']:.4f}")
            logger.info("--- 本次分析已采用以下优化参数 ---")
            best_params_str = json.dumps(optuna_summary['best_params'], indent=2, ensure_ascii=False)
            logger.info(best_params_str)
            # 显示ARM参数使用状态
            if use_dynamic_adjustment:
                logger.info("注意: ARM参数可能因数据量限制被动态调整")
            else:
                logger.info("ARM参数: 严格按照优化结果执行，未进行动态调整")
        else: logger.info(f"错误信息: {optuna_summary['error']}")
    else:
        logger.info("\n--- 本次分析使用脚本内置的默认权重 ---")

    # 7. 全局分析
    
    full_history_arm_rules = analyze_associations(main_df, active_weights, enable_dynamic_adjustment=use_dynamic_adjustment)
    
    # 8. 回测并打印报告
    logger.info("\n" + "="*25 + " 策 略 回 测 摘 要 " + "="*25)
    backtest_results_df, backtest_stats = run_backtest(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
    
    if not backtest_results_df.empty:
        num_periods_tested = len(backtest_results_df['period'].unique())
        logger.info(f"回测周期: 最近 {num_periods_tested} 期")
        
        # 分玩法统计
        prize_counts_by_play = backtest_stats.get('prize_counts_by_play', {})
        
        # 快乐8真实奖金表 - 按各玩法设定
        play_prize_values = {
            1: {"一等奖": 4.6},  # 选一
            2: {"一等奖": 19},   # 选二
            3: {"一等奖": 53, "二等奖": 3},  # 选三
            4: {"一等奖": 100, "二等奖": 5, "三等奖": 3},  # 选四
            5: {"一等奖": 1000, "二等奖": 21, "三等奖": 3},  # 选五
            6: {"一等奖": 3000, "二等奖": 30, "三等奖": 10, "四等奖": 3},  # 选六
            7: {"一等奖": 10000, "二等奖": 288, "三等奖": 28, "四等奖": 4, "五等奖": 2},  # 选七
            8: {"一等奖": 50000, "二等奖": 800, "三等奖": 88, "四等奖": 10, "五等奖": 3, "六等奖": 2},  # 选八
            9: {"一等奖": 300000, "二等奖": 2000, "三等奖": 200, "四等奖": 20, "五等奖": 5, "六等奖": 3, "七等奖": 2},  # 选九
            10: {"一等奖": 5000000, "二等奖": 8000, "三等奖": 800, "四等奖": 80, "五等奖": 5, "六等奖": 3, "七等奖": 2}  # 选十：中10个一等奖500万元，中9个二等奖8000元，以此类推
        }
        
        logger.info("\n--- 各玩法中奖统计 ---")
        total_revenue_all_plays = 0
        total_cost_all_plays = 0
        
        for play_type in range(1, 11):
            play_name = PLAY_TYPES[play_type]
            if play_type in prize_counts_by_play:
                prize_dist = prize_counts_by_play[play_type]
                total_bets_this_play = num_periods_tested  # 每期每种玩法投注1注
                total_cost = total_bets_this_play * 2  # 每注2元
                
                # 使用对应玩法的奖金表计算收益
                prize_values_for_play = play_prize_values.get(play_type, {})
                total_revenue = sum(prize_values_for_play.get(p, 0) * c for p, c in prize_dist.items())
                
                roi = (total_revenue - total_cost) * 100 / total_cost if total_cost > 0 else 0
                
                total_revenue_all_plays += total_revenue
                total_cost_all_plays += total_cost
                
                logger.info(f"\n{play_name}:")
                logger.info(f"  - 投注: {total_bets_this_play} 注 | 成本: {total_cost:,.0f} 元 | 收益: {total_revenue:,.0f} 元 | ROI: {roi:.2f}%")
                
                if prize_dist:
                    logger.info("  - 中奖分布:")
                    for prize in prize_values_for_play.keys():
                        if prize in prize_dist: 
                            logger.info(f"    - {prize}: {prize_dist[prize]} 次")
                else:
                    logger.info("  - 未命中任何奖级")
        
        # 总体统计
        overall_roi = (total_revenue_all_plays - total_cost_all_plays) * 100 / total_cost_all_plays if total_cost_all_plays > 0 else 0
        logger.info(f"\n--- 总体统计 ---")
        logger.info(f"  - 总投入: {total_cost_all_plays:,.0f} 元")
        logger.info(f"  - 总收益: {total_revenue_all_plays:,.0f} 元")
        logger.info(f"  - 总体ROI: {overall_roi:.2f}%")
        logger.info(f"  - 平均命中数: {backtest_results_df['hits'].mean():.3f}")
        
        # 各玩法命中率统计
        logger.info("\n--- 各玩法中奖率 ---")
        for play_type in range(1, 11):
            play_name = PLAY_TYPES[play_type]
            play_results = backtest_results_df[backtest_results_df['play_type'] == play_type]
            if not play_results.empty:
                total_plays = len(play_results)
                win_plays = len(play_results[play_results['prize'].notna()])
                win_rate = win_plays * 100 / total_plays if total_plays > 0 else 0
                avg_hits = play_results['hits'].mean()
                logger.info(f"  - {play_name}: 中奖率 {win_rate:.1f}% ({win_plays}/{total_plays}) | 平均命中 {avg_hits:.2f}")
    else: 
        logger.warning("回测未产生有效结果，可能是数据量不足。")
    
    # 9. 最终推荐
    logger.info("\n" + "="*25 + f" 第 {last_period + 1} 期 号 码 推 荐 " + "="*25)
    final_recs, final_rec_strings, _, _, final_scores = run_analysis_and_recommendation(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules)
    
    # 输出各玩法推荐
    for line in final_rec_strings: 
        logger.info(line)
    
    # 额外显示号码得分排序
    logger.info("\n--- 号码得分排序 (Top 30) ---")
    if final_scores and final_scores.get('number_scores'):
        top_30_with_scores = sorted(final_scores['number_scores'].items(), key=lambda x: x[1], reverse=True)[:30]
        score_lines = []
        for i, (num, score) in enumerate(top_30_with_scores, 1):
            score_lines.append(f"{num:02d}({score:.1f})")
            if i % 10 == 0:  # 每10个号码换行
                logger.info(f"  {' '.join(score_lines)}")
                score_lines = []
        if score_lines:  # 输出剩余的号码
            logger.info(f"  {' '.join(score_lines)}")
    
    logger.info("\n" + "="*60 + f"\n--- 报告结束 (详情请查阅: {os.path.basename(log_filename)}) ---\n") 

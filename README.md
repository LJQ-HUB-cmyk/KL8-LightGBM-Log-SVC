# 快乐8数据处理与分析系统

## 项目仓库地址
**GitHub仓库**: https://github.com/LJQ-HUB-cmyk/KL8-LightGBM-Log-SVC

## 项目部署与同步

### 首次克隆项目

如果您还没有本地项目副本，可以通过以下命令克隆：

```bash
git clone https://github.com/LJQ-HUB-cmyk/KL8-LightGBM-Log-SVC.git
cd KL8-LightGBM-Log-SVC
```

### 将现有项目同步到远程仓库

如果您已有本地项目代码，需要同步到远程仓库，请按以下步骤操作：

#### 1. 初始化Git仓库（如果还未初始化）
```bash
git init
```

#### 2. 添加远程仓库地址
```bash
git remote add origin https://github.com/LJQ-HUB-cmyk/KL8-LightGBM-Log-SVC.git
```

#### 3. 检查远程仓库配置
```bash
git remote -v
```
应该显示：
```
origin  https://github.com/LJQ-HUB-cmyk/KL8-LightGBM-Log-SVC.git (fetch)
origin  https://github.com/LJQ-HUB-cmyk/KL8-LightGBM-Log-SVC.git (push)
```

#### 4. 拉取远程仓库内容（如果远程仓库不为空）
```bash
git pull origin main --allow-unrelated-histories
```

#### 5. 添加所有文件到Git跟踪
```bash
git add .
```

#### 6. 提交本地更改
```bash
git commit -m "Initial commit: 修改NameError: name 'use_dynamic_adjustment' is not defined问题"
```

#### 7. 将分支重命名为main（如果当前是master分支）
```bash
git branch -M main
```

#### 8. 推送到远程仓库
```bash
git push -u origin main
```

**注意**：首次推送可能需要GitHub身份验证：
- 使用GitHub用户名和Personal Access Token
- 或配置SSH密钥进行身份验证

### 验证同步状态

同步完成后，可以通过以下命令验证：

```bash
# 检查远程仓库配置
git remote -v

# 查看分支状态
git branch -a

# 查看提交历史
git log --oneline -5
```

### 日常更新操作

项目同步到远程仓库后，日常更新操作：

#### 推送本地更改到远程
```bash
git add .
git commit -m "修改奖金规则"
git push origin main
```

#### 拉取远程更新到本地
```bash
git pull origin main
```

#### 查看文件状态
```bash
git status
```
如果需要强制推送（谨慎使用）：
git push -f origin main

### GitHub Actions自动化

项目已配置GitHub Actions自动化工作流，推送到远程仓库后：

1. **自动触发条件**：
   - 每天北京时间7:00自动运行
   - 手动触发（在GitHub Actions页面）

2. **自动执行流程**：
   - 清理缓存文件
   - 获取最新快乐8数据
   - 运行智能分析
   - 验证推荐结果
   - 发送微信推送
   - 自动提交更新

3. **查看运行结果**：
   - 访问：https://github.com/LJQ-HUB-cmyk/KL8-LightGBM-Log-SVC/actions
   - 查看工作流执行状态和日志

### 微信推送配置

如需使用微信推送功能，请：

1. 注册WxPusher账号：https://wxpusher.zjiecode.com/
2. 获取APP_TOKEN和UID
3. 修改 `kl8_wxpusher.py` 中的配置：
   ```python
   APP_TOKEN = "你的APP_TOKEN"
   USER_UIDS = ["你的UID"]
   ```

---

## 1. kl8_data_processor.py - 数据获取与预处理

### 主要功能
这个脚本负责从网络获取快乐8历史开奖数据，并将其整合到一个标准化的CSV文件中。

### 核心逻辑流程
1. **数据获取**：
   - 从网络文本文件获取数据（`fetch_kl8_data`）- 主要数据源，包含日期信息
   - 数据源：https://data.17500.cn/kl8_asc.txt

2. **数据解析**：
   - 解析文本数据 - 将原始文本转换为结构化数据
   - 验证号码格式（确保每期20个号码，号码范围1-80）

3. **数据整合与存储**：
   - 将数据保存到CSV文件（`kuaile8.csv`）
   - 按期号升序排序，确保数据有序性

4. **错误处理**：
   - 多种编码尝试（utf-8, gbk, latin-1）
   - 网络连接错误处理
   - 数据格式验证

### 技术特点
- 使用requests库进行网络请求
- 使用pandas进行数据处理和CSV操作
- 实现了日志系统和进度显示
- 数据验证确保质量

## 2. kl8_analyzer.py - 数据分析与预测

### 主要功能
这个脚本负责分析快乐8历史数据，识别模式，训练机器学习模型，并生成下一期的推荐号码组合。

### 核心逻辑流程
1. **数据加载与预处理**：
   - 加载CSV数据（`load_data`）
   - 数据清理和结构化（`clean_and_structure`）
   - 特征工程（`feature_engineer`）- 创建和值、跨度、奇偶计数等特征

2. **历史统计分析**：
   - 频率和遗漏分析（`analyze_frequency_omission`）
   - 模式分析（`analyze_patterns`）- 分析奇偶比、区域分布等
   - 关联规则挖掘（`analyze_associations`）- 使用Apriori算法

3. **机器学习模型训练**：
   - 创建滞后特征（`create_lagged_features`）
   - 训练LightGBM分类器模型（`train_prediction_models`）
   - 预测下一期号码概率（`predict_next_draw_probabilities`）

4. **号码评分与组合生成**：
   - 计算综合得分（`calculate_scores`）- 结合频率、遗漏和ML预测概率
   - 生成推荐组合（`generate_combinations`）- 基于得分和历史模式

5. **回测与验证**：
   - 历史数据回测（`run_backtest`）- 评估预测方法的有效性
   - 结果分析与统计

6. **结果输出**：
   - 生成分析报告
   - 推荐单式组合（每注20个号码）
   - 推荐号码池参考

### 技术特点
- 专门适配快乐8游戏规则（80选20）
- 使用LightGBM机器学习算法
- 实现了特征工程和滞后特征创建
- 使用关联规则挖掘（Apriori算法）
- 实现了回测系统评估预测效果
- 采用模块化设计，功能分离清晰
- 支持Optuna参数优化

## 3. kl8_bonus_calculation.py - 奖金计算与验证

### 主要功能
这个脚本负责验证推荐结果的实际表现，计算中奖情况和奖金。

### 核心逻辑流程
1. **数据匹配**：
   - 读取历史开奖数据
   - 匹配分析报告和实际开奖结果

2. **中奖验证**：
   - 计算每注推荐与开奖号码的命中数
   - 根据快乐8中奖规则确定奖级

3. **奖金计算**：
   - 按照奖级计算奖金
   - 统计总收益和投资回报率

4. **报告生成**：
   - 生成详细的验证报告
   - 记录历史验证结果

### 快乐8中奖规则
- 一等奖：命中15个以上号码
- 二等奖：命中13-14个号码  
- 三等奖：命中11-12个号码
- 四等奖：命中9-10个号码
- 五等奖：命中7-8个号码
- 六等奖：命中5-6个号码

## 系统整体效果

这三个脚本共同构成了一个完整的快乐8数据分析与预测系统：

1. **数据流向**：
   - `kl8_data_processor.py` 负责获取和预处理数据，生成标准化CSV
   - `kl8_analyzer.py` 读取CSV，进行分析和预测
   - `kl8_bonus_calculation.py` 验证预测结果的实际表现

2. **系统优势**：
   - **数据获取的健壮性**：自动数据源获取，多编码支持，错误处理
   - **分析的全面性**：结合统计分析和机器学习
   - **预测的科学性**：基于历史数据模式和机器学习算法
   - **可验证性**：通过回测和实际验证评估预测效果

3. **实际应用效果**：
   - 系统能够基于历史数据识别快乐8号码模式
   - 使用机器学习预测单个号码的出现概率
   - 生成综合评分较高的号码组合
   - 通过回测评估预测方法的有效性
   - 提供详细的分析报告和验证结果

## 自动化运行

通过GitHub Actions实现每日自动运行：
- 每天北京时间7:00自动执行
- 自动获取最新数据
- 自动生成分析报告
- 自动验证历史预测结果
- 自动提交更新到仓库

## 快乐8游戏特点

快乐8是一种数字型彩票游戏：
- 号码池：1-80
- 每期开奖：20个号码
- 投注方式：选择若干个号码进行投注
- 中奖规则：根据命中开奖号码的个数确定奖级
- 开奖频率：通常每天多期

本系统专门针对快乐8的这些特点进行了算法优化和策略设计。

## 使用说明

### 手动运行方式

1. **环境配置**：安装requirements.txt中的依赖包
2. **数据获取**：运行 `python kl8_data_processor.py`
3. **分析预测**：运行 `python kl8_analyzer.py`
4. **结果验证**：运行 `python kl8_bonus_calculation.py`
5. **微信推送**：运行 `python kl8_wxpusher.py`

### 一键运行方式

运行 `python update_and_analyze.py` 执行完整的数据更新、分析和推送流程。

### 智能数据更新机制

系统采用了智能的数据缓存更新机制：

1. **自动检测数据更新**：
   - 比较原始数据文件和缓存文件的修改时间
   - 验证缓存数据的完整性（期数匹配检查）
   - 当检测到新数据时自动重新生成缓存

2. **缓存优化策略**：
   - 首次运行或缓存不存在时：自动生成预处理缓存
   - 数据无更新时：直接使用缓存，提高运行效率
   - 检测到数据更新时：自动清理缓存并重新处理

3. **GitHub Actions自动化**：
   - 每次运行前自动清理缓存文件
   - 确保每日7:00运行时使用最新数据
   - 包含完整的微信推送通知

### 微信推送功能

系统集成了微信推送通知：
- **分析结果推送**：包含各玩法推荐号码、优化状态、历史ROI
- **验证结果推送**：包含上期推荐的中奖情况、奖金统计
- **推送时机**：每日分析完成后自动发送
- **消息格式**：结构化展示，便于快速查看关键信息

### 自动化运行

通过GitHub Actions实现完全自动化：
- 每天北京时间7:00自动执行
- 智能检测并获取最新数据
- 自动生成分析报告和验证结果
- 自动发送微信推送通知
- 自动提交更新到仓库

# stock-ai-system
# 📈 基于LSTM与本地大模型的A股智能投研系统

## 项目简介
一个端到端的A股智能投研系统，输入任意股票代码，自动输出次日涨跌预测与AI投资建议。

**核心价值**：
- 🔮 LSTM模型预测次日涨跌方向（准确率 ~57%）
- 🤖 本地部署Qwen2.5大模型生成投资建议
- 🌐 Streamlit交互式Web界面
- 💻 全本地化部署，数据安全无隐私泄露风险

## 技术架构
用户输入股票代码
        ↓
BaoStock数据采集（个股 + 大盘指数）
        ↓
21项技术指标特征工程 → LSTM预测 → Qwen2.5大模型分析
        ↓
Streamlit Web界面展示（预测结果 + AI投资建议）


## 技术栈
| 模块 | 技术 |
|------|------|
| 数据采集 | BaoStock (A股数据) |
| 数据处理 | Pandas, NumPy, scikit-learn |
| 深度学习 | TensorFlow/Keras, LSTM |
| 大模型 | Ollama, Qwen2.5-7B |
| 前端展示 | Streamlit |
| 开发语言 | Python |

## 核心功能
1. **自动数据采集**：输入任意A股代码，系统自动下载历史数据（含大盘指数）
2. **智能预测**：基于LSTM模型预测次日涨跌方向及概率
3. **AI投资建议**：调用本地Qwen2.5大模型，将预测结果转化为专业投资分析
4. **交互式界面**：Streamlit Web应用，操作简单直观

## 项目亮点
- ✅ 从数据采集到模型部署的完整AI开发链路
- ✅ 本地大模型部署实战经验
- ✅ 模块化设计，代码结构清晰，易于扩展
- ✅ 解决类别不平衡、模型崩溃等真实问题

## 效果展示
| 股票代码 | 预测方向 | 上涨概率 |
|---------|---------|---------|
| sh.600036（招商银行） | 涨 | 58.3% |
| sh.601318（中国平安） | 涨 | 59.5% |
| sh.600030（中信证券） | 涨 | 58.3% |

```
## 项目结构
stock-ai-system/
├── app.py # Streamlit Web界面
├── llm_advisor.py # 预测 + AI建议核心逻辑
├── preprocess.py # 数据预处理、特征工程
├── data_loader_with_index.py # 数据采集（含大盘指数）
├── lstm_model.py # LSTM模型训练
├── data/ # 股票数据存储
├── *.keras # 训练好的模型文件
└── requirements.txt # 依赖包列表

## 如何运行
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动Ollama服务（需提前安装Ollama）
ollama pull qwen2.5:7b
ollama serve

# 3. 启动Web应用
streamlit run app.py
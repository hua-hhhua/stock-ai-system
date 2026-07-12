import requests
import json
import numpy as np
import os
import traceback
from tensorflow.keras.models import load_model
from preprocess import prepare_data

class LLMAdvisor:
    """本地大模型投研助手"""
    
    def __init__(self, model_name="qwen2.5:7b", host="localhost", port=11434):
        self.model_name = model_name
        self.base_url = f"http://{host}:{port}/api/generate"
    
    def ask(self, prompt, temperature=0.7):
        """向本地大模型发送提问"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "无响应")
        except Exception as e:
            return f"调用失败: {str(e)}\n提示：请确保 Ollama 正在运行 (ollama serve)"
    
    def generate_investment_advice(self, ticker, prediction, probability, threshold):
        """生成投资建议"""
        pred_text = "涨" if prediction == 1 else "跌"
        
        prompt = f"""
你是一位专业的量化投资分析师。请基于以下LSTM模型的预测结果，生成一份简洁、专业的投资建议。

【股票代码】{ticker}
【预测方向】{pred_text}
【上涨概率】{probability*100:.1f}%
【决策阈值】{threshold:.2f}

请从以下维度给出分析（每条1-2句话）：
1. 信号强度评估（概率是否显著高于/低于阈值）
2. 短期操作建议（仓位建议、入场/观望）
3. 风险提示（需要关注的因素）

注意：
- 回答要专业、理性
- 明确提示"本建议仅供参考，不构成投资操作依据"
- 总字数控制在200字以内
"""
        return self.ask(prompt)


def predict_next_day(ticker, model_path=None):
    """
    预测下一个交易日涨跌
    """
    try:
        # 将 ticker 中的点号替换为下划线，匹配实际文件名
        ticker_clean = ticker.replace('.', '_')
        
        # ✅ 加载模型
        if model_path is None:
            # 1. 尝试该股票的专属模型
            model_path = f"lstm_model_{ticker_clean}_horizon5.keras"
            if not os.path.exists(model_path):
                print(f"⚠️ 未找到专属模型 {model_path}")
                # 2. 尝试招商银行模型（作为通用模型）
                model_path = "lstm_model_sh_600036_horizon5.keras"
                if not os.path.exists(model_path):
                    print(f"⚠️ 未找到招商银行模型，使用默认模型...")
                    model_path = "lstm_stock_model.keras"
                else:
                    print(f"✅ 使用招商银行模型作为通用模型")
            else:
                print(f"✅ 使用专属模型: {model_path}")
        
        try:
            model = load_model(model_path)
            print(f"✅ 加载模型: {model_path}")
            # 获取模型期望的输入特征数
            expected_features = model.input_shape[-1]
            print(f"📊 模型期望特征数: {expected_features}")
        except Exception as e:
            print(f"❌ 无法加载模型: {e}")
            return None, None, None
        
        # ✅ 加载数据：直接使用原始特征（不做PCA）
        # 招商银行模型是用 use_pca=False 训练的
        print(f"📊 正在加载数据: {ticker}")
        X, y, scaler, df, feature_cols = prepare_data(
            ticker,
            time_steps=60,
            horizon=5,
            use_pca=False,      # ✅ 不做 PCA，保持原始 23 个特征
            n_components=None
        )
        
        if X is None:
            print("❌ prepare_data 返回 None")
            return None, None, None
        
        print(f"📊 数据加载成功: {len(X)} 个样本, 特征数: {X.shape[2]}")
        
        # 取最后一条数据预测
        last_seq = X[-1]
        X_input = last_seq.reshape(1, 60, -1)
        
        # 预测概率
        prob = model.predict(X_input, verbose=0)[0][0]
        print(f"📊 预测概率: {prob:.4f}")
        
        threshold = 0.58
        pred = 1 if prob > threshold else 0
        
        return pred, prob, threshold
        
    except Exception as e:
        print(f"❌ predict_next_day 发生异常: {e}")
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    import sys
    
    # 获取股票代码
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "sh.600036"
    
    print("=" * 60)
    print(f"🔮 {ticker} 智能投研系统")
    print("=" * 60)
    
    # 1. LSTM预测
    print("\n📊 正在加载模型并预测...")
    pred, prob, threshold = predict_next_day(ticker)
    
    if pred is None:
        print("❌ 预测失败，请检查数据")
        sys.exit(1)
    
    pred_text = "涨" if pred == 1 else "跌"
    print(f"\n📈 LSTM预测结果:")
    print(f"  方向: {pred_text}")
    print(f"  概率: {prob*100:.1f}%")
    print(f"  阈值: {threshold:.2f}")
    
    # 2. 大模型生成建议
    print("\n🤖 正在调用本地大模型生成投资建议...")
    advisor = LLMAdvisor()
    advice = advisor.generate_investment_advice(ticker, pred, prob, threshold)
    
    print("\n" + "=" * 60)
    print("💡 投资建议")
    print("=" * 60)
    print(advice)
    print("\n" + "=" * 60)
    print("⚠️ 免责声明：以上内容由AI生成，仅供参考，不构成投资建议。")
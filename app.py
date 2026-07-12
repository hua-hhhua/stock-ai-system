import streamlit as st
import os
import sys
import traceback
import numpy as np
from tensorflow.keras.models import load_model
from data_loader_with_index import fetch_data_with_index
from preprocess import prepare_data

st.set_page_config(page_title="智能投研系统", page_icon="📈", layout="centered")

st.title("📈 智能投研系统")
st.caption("输入任意A股代码，AI帮你预测涨跌并生成投资建议")

ticker = st.text_input("股票代码", value="sh.600036", placeholder="例如: sh.600036, sh.600519, sz.000858")

if st.button("🔮 预测", type="primary"):
    if not ticker:
        st.warning("请输入股票代码")
    else:
        with st.spinner("正在准备数据并预测..."):
            try:
                # 1. 检查数据文件
                file_path = f"data/{ticker}_with_index.csv"
                st.info(f"📂 1. 检查文件: {file_path}")
                
                if not os.path.exists(file_path):
                    st.info(f"📥 首次使用 {ticker}，正在自动下载数据...")
                    df = fetch_data_with_index(ticker, "sh.000001")
                    if df is None:
                        st.error(f"❌ 下载失败")
                        st.stop()
                    st.success("✅ 数据下载完成！")
                else:
                    st.success("✅ 数据文件已存在")
                
                # 2. 加载模型
                st.info("🔮 2. 加载模型...")
                ticker_clean = ticker.replace('.', '_')
                model_path = f"lstm_model_{ticker_clean}_horizon5.keras"
                if not os.path.exists(model_path):
                    model_path = "lstm_model_sh_600036_horizon5.keras"
                if not os.path.exists(model_path):
                    model_path = "lstm_stock_model.keras"
                
                model = load_model(model_path)
                st.info(f"✅ 模型加载成功: {model_path}")
                
                # 3. 加载数据
                st.info("📊 3. 加载数据...")
                X, y, scaler, df, feature_cols = prepare_data(
                    ticker,
                    time_steps=60,
                    horizon=5,
                    use_pca=False,
                    n_components=None
                )
                
                if X is None:
                    st.error("❌ 数据加载失败")
                    st.stop()
                
                st.info(f"✅ 数据加载成功: {len(X)} 个样本")
                
                # 4. 预测
                st.info("🔮 4. 预测中...")
                last_seq = X[-1]
                X_input = last_seq.reshape(1, 60, -1)
                prob = model.predict(X_input, verbose=0)[0][0]
                threshold = 0.58
                pred = 1 if prob > threshold else 0
                
                st.info(f"📊 预测结果: pred={pred}, prob={prob:.4f}")
                
                # 5. 显示结果
                col1, col2, col3 = st.columns(3)
                pred_text = "📈 涨" if pred == 1 else "📉 跌"
                col1.metric("预测方向", pred_text)
                col2.metric("上涨概率", f"{prob*100:.1f}%")
                col3.metric("决策阈值", f"{threshold:.2f}")
                
                # 6. 大模型建议
                st.info("🤖 5. 生成投资建议...")
                from llm_advisor import LLMAdvisor
                advisor = LLMAdvisor(model_name="qwen2.5:7b")
                advice = advisor.generate_investment_advice(ticker, pred, prob, threshold)
                
                st.divider()
                st.subheader("💡 AI 投资建议")
                st.markdown(advice)
                st.caption("⚠️ 免责声明：以上内容由AI生成，仅供参考，不构成投资建议。")
                
            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")
                st.code(traceback.format_exc())
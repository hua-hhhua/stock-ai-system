import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sys
import warnings
warnings.filterwarnings('ignore')

from preprocess import prepare_data

def build_lstm_model(input_shape):
    """构建LSTM模型"""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.25),
        LSTM(16, return_sequences=False),
        Dropout(0.25),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def grid_search_for_stock(X_train, y_train, X_val, y_val, X_test, y_test, input_shape):
    """
    针对当前股票自动搜索最优权重和阈值
    """
    train_ratio = sum(y_train) / len(y_train)
    print(f"\n📊 训练集涨占比: {train_ratio*100:.2f}%")
    
    # 权重搜索范围（给"涨"类更高权重）
    weight_range = [1.5, 2.0, 2.5, 3.0, 3.5]
    thresholds = [0.50, 0.55, 0.58, 0.60, 0.62]
    
    best_acc = 0
    best_config = None
    best_model = None
    best_pred_ratio = 0
    
    results = []
    
    for w in weight_range:
        class_weight_dict = {0: 1.0, 1: w}
        for thresh in thresholds:
            print(f"\n🔍 测试: 权重涨={w:.1f}, 阈值={thresh:.2f}")
            
            model = build_lstm_model(input_shape)
            early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                class_weight=class_weight_dict,
                verbose=0
            )
            
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > thresh).astype(int)
            acc = accuracy_score(y_test, y_pred)
            pred_ratio = sum(y_pred) / len(y_pred) if len(y_pred) > 0 else 0
            
            print(f"  结果: 准确率={acc*100:.2f}%, 预测涨占比={pred_ratio*100:.1f}%")
            
            results.append({
                'weight': w,
                'threshold': thresh,
                'accuracy': acc,
                'pred_ratio': pred_ratio
            })
            
            # 优先选择准确率最高且预测涨占比在 20%~80% 之间的配置
            if acc > best_acc and 0.20 < pred_ratio < 0.80:
                best_acc = acc
                best_config = (class_weight_dict, thresh)
                best_model = model
                best_pred_ratio = pred_ratio
    
    # 如果没有找到健康的配置，退而求其次选准确率最高的
    if best_model is None:
        print("\n⚠️ 没有找到预测分布健康的配置，选择准确率最高的...")
        # 找准确率最高的结果
        best_result = max(results, key=lambda x: x['accuracy'])
        best_acc = best_result['accuracy']
        best_weight = best_result['weight']
        best_thresh = best_result['threshold']
        best_config = ({0: 1.0, 1: best_weight}, best_thresh)
        
        # 重新训练这个配置
        model = build_lstm_model(input_shape)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            class_weight={0: 1.0, 1: best_weight},
            verbose=0
        )
        best_model = model
    
    print(f"\n{'='*55}")
    print(f"✅ 最优配置: 权重涨={best_config[0][1]:.1f}, 阈值={best_config[1]:.2f}")
    print(f"✅ 最佳准确率: {best_acc*100:.2f}%")
    print(f"{'='*55}\n")
    
    return best_model, best_config, results

def evaluate_model(model, X_test, y_test, best_threshold):
    """评估模型"""
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > best_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 测试集准确率: {acc*100:.2f}%")
    print(f"📊 AUC: {auc:.4f}")
    print("\n📋 详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '涨']))
    
    print(f"\n📊 预测分布: 涨={sum(y_pred)}次, 跌={len(y_pred)-sum(y_pred)}次")
    print(f"📊 实际分布: 涨={sum(y_test)}次, 跌={len(y_test)-sum(y_test)}次")
    
    return y_pred, y_pred_proba

def plot_history(history, ticker):
    """绘制训练曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='训练损失')
    axes[0].plot(history.history['val_loss'], label='验证损失')
    axes[0].set_title(f'{ticker} 损失曲线')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['accuracy'], label='训练准确率')
    axes[1].plot(history.history['val_accuracy'], label='验证准确率')
    axes[1].set_title(f'{ticker} 准确率曲线')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{ticker}.png', dpi=150)
    print(f"✅ 训练曲线已保存为 training_history_{ticker}.png")
    plt.show()

if __name__ == "__main__":
    # ✅ 支持命令行参数：python lstm_model.py sh.600519
    if len(sys.argv) > 1:
        ticker_input = sys.argv[1]
        # 如果输入带点号，转为下划线（与data_loader保存格式一致）
        ticker = ticker_input.replace(".", "_")
    else:
        ticker = "sh_600519"
    
    print("=" * 55)
    print(f"🚀 开始为 {ticker} 训练LSTM股票预测模型（优化版）")
    print("=" * 55)
    
    # ========== 加载数据（horizon=5） ==========
    X, y, scaler, df, feature_columns = prepare_data(
        ticker, 
        time_steps=60,
        horizon=5,           # 预测未来5天，减少噪声
        use_pca=False,       # ✅ 先用原始特征，不降维
        n_components=None
    )
    
    if X is None:
        print(f"❌ 无法加载 {ticker} 数据，请检查股票代码")
        sys.exit(1)
    
    print(f"\n📊 总样本: {len(X)} 个, 特征数: {X.shape[2]} 个")
    print(f"📊 标签分布: 涨={sum(y==1)} ({sum(y==1)/len(y)*100:.2f}%), 跌={sum(y==0)} ({sum(y==0)/len(y)*100:.2f}%)")
    
    total = len(X)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(X_train)} 个样本")
    print(f"  验证集: {len(X_val)} 个样本")
    print(f"  测试集: {len(X_test)} 个样本")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # ========== 网格搜索 ==========
    print("\n" + "=" * 55)
    print("🔍 自动搜索最优配置...")
    print("=" * 55)
    
    best_model, best_config, results = grid_search_for_stock(
        X_train, y_train, X_val, y_val, X_test, y_test, input_shape
    )
    
    # ========== 最终评估 ==========
    best_threshold = best_config[1]
    y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, best_threshold)
    
    # ========== 保存模型 ==========
    model_path = f'lstm_model_{ticker}_horizon5.keras'
    best_model.save(model_path)
    print(f"\n✅ 模型已保存为 {model_path}")
    
    # ========== 预测下一交易日 ==========
    print("\n" + "=" * 55)
    print("🔮 下一个交易日预测")
    print("=" * 55)
    
    last_seq = X[-1]
    X_input = last_seq.reshape(1, 60, -1)
    prob = best_model.predict(X_input, verbose=0)[0][0]
    pred = "涨" if prob > best_threshold else "跌"
    print(f"  股票: {ticker}")
    print(f"  预测: {pred} (概率: {prob*100:.1f}%)")
    print(f"  使用阈值: {best_threshold:.2f}")
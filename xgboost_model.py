import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from preprocess import prepare_data

if __name__ == "__main__":
    print("=" * 55)
    print("🚀 快速 XGBoost 训练")
    print("=" * 55)
    
    ticker = "sh_600519"
    
    # 加载数据
    X, y, scaler, df, feature_columns = prepare_data(
        ticker, 
        time_steps=60,
        horizon=5,
        use_pca=False,
        n_components=None
    )
    
    # 展平：将 60×21 变成 1260 个特征
    X_flat = X.reshape(X.shape[0], -1)
    print(f"📊 展平后 X 形状: {X_flat.shape}")
    
    # 划分数据
    total = len(X_flat)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    X_train = X_flat[:train_size]
    y_train = y[:train_size]
    X_val = X_flat[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X_flat[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"📊 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # ✅ 直接用一组固定参数，不做网格搜索
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        scale_pos_weight=1.4,  # 手动设置，类似你的 LSTM 权重
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    
    print("\n🔍 开始训练...")
    model.fit(X_train, y_train)
    
    # 验证集评估（找最优阈值）
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # 测试不同阈值
    thresholds = [0.48, 0.50, 0.52, 0.55, 0.58]
    best_acc = 0
    best_th = 0.5
    
    print("\n📊 验证集不同阈值的表现:")
    for th in thresholds:
        y_pred = (y_val_proba > th).astype(int)
        acc = accuracy_score(y_val, y_pred)
        pred_ratio = sum(y_pred) / len(y_pred)
        print(f"  阈值 {th:.2f}: 准确率={acc*100:.2f}%, 预测涨占比={pred_ratio*100:.1f}%")
        if acc > best_acc and 0.20 < pred_ratio < 0.80:
            best_acc = acc
            best_th = th
    
    if best_acc == 0:
        # 如果没有找到健康的配置，选准确率最高的
        for th in thresholds:
            y_pred = (y_val_proba > th).astype(int)
            acc = accuracy_score(y_val, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_th = th
    
    print(f"\n✅ 最优阈值: {best_th:.2f}")
    
    # 测试集评估
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba > best_th).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n{'='*55}")
    print(f"📊 测试集准确率: {acc*100:.2f}%")
    print(f"📊 AUC: {auc:.4f}")
    print(f"{'='*55}")
    
    print("\n📋 分类报告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '涨']))
    
    print(f"\n📊 预测分布: 涨={sum(y_pred)}次, 跌={len(y_pred)-sum(y_pred)}次")
    print(f"📊 实际分布: 涨={sum(y_test)}次, 跌={len(y_test)-sum(y_test)}次")
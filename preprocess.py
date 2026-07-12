import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import os

def add_technical_features(df):
    """
    在 OHLCV 基础上添加技术指标特征
    """
    df = df.copy()
    
    # 1. 价格衍生特征
    df['High_Low_ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-6)
    df['Close_Open_ratio'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-6)
    df['Return_1d'] = df['Close'].pct_change()
    
    # 2. 均线特征 (MA)
    for period in [5, 10, 20, 60]:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'MA_ratio_{period}'] = df['Close'] / (df[f'MA_{period}'] + 1e-6) - 1
    
    # 3. 波动率
    df['Volatility'] = df['Return_1d'].rolling(window=10).std()
    
    # 4. 成交量相关
    df['Volume_ratio'] = df['Volume'] / (df['Volume'].rolling(window=10).mean() + 1e-6)
    
    # 5. RSI (14日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 删除所有 NaN 行
    df = df.dropna()
    
    return df

def create_labels(df, horizon=1):
    """创建标签：预测未来 horizon 天的涨跌"""
    df = df.copy()
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    df = df[:-horizon] if horizon > 0 else df
    return df

def prepare_data(ticker, time_steps=60, horizon=1, test_size=0.2, val_size=0.1, use_pca=True, n_components=8):
    """
    准备LSTM训练数据
    """
    # ✅ 统一格式：直接使用 ticker，保持和 data_loader 一致
    # 例如 sh.600036 → data/sh.600036_with_index.csv
    file_path = f"data/{ticker}_with_index.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ 文件 {file_path} 不存在，请先运行 data_loader_with_index.py 下载数据")
        return None, None, None, None, None
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"📊 原始数据: {len(df)} 条")
    
    # 添加技术指标
    df = add_technical_features(df)
    print(f"📊 添加技术指标后: {len(df)} 条")
    
    # 构造标签
    df = create_labels(df, horizon=horizon)
    print(f"📊 构造标签后: {len(df)} 条")
    
    # 选择特征（所有技术指标 + 大盘特征）
    feature_columns = [
        'Open', 'High', 'Low', 'Close',
        'High_Low_ratio', 'Close_Open_ratio', 'Return_1d',
        'MA_5', 'MA_10', 'MA_20', 'MA_60',
        'MA_ratio_5', 'MA_ratio_10', 'MA_ratio_20', 'MA_ratio_60',
        'Volatility', 'Volume_ratio',
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        # 大盘相关特征
        'Idx_Return',
        'Relative_Strength'
    ]
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"📊 原始特征数: {len(feature_columns)} 个")
    
    # 归一化
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    
    # 可选：PCA降维
    if use_pca:
        n_components_actual = n_components if n_components else 0.95
        pca = PCA(n_components=n_components_actual)
        scaled_features = pca.fit_transform(scaled_features)
        print(f"📊 PCA降维后特征数: {scaled_features.shape[1]} 个（原 {len(feature_columns)} 个）")
        print(f"📊 PCA累计方差贡献: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        scaler.pca = pca
        feature_columns = [f'PC{i+1}' for i in range(scaled_features.shape[1])]
    
    # 获取标签
    target_values = df['Target'].values
    
    # 滑动窗口构造样本
    X, y = [], []
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i - time_steps:i])
        y.append(target_values[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"📊 构造样本完成: {len(X)} 个样本")
    print(f"📊 每个样本: {time_steps} 天 × {X.shape[2]} 个特征")
    
    # 按时间顺序划分数据集
    total = len(X)
    train_size = int(total * (1 - test_size - val_size))
    val_size_actual = int(total * val_size)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size_actual]
    y_val = y[train_size:train_size + val_size_actual]
    X_test = X[train_size + val_size_actual:]
    y_test = y[train_size + val_size_actual:]
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(X_train)} 个样本")
    print(f"  验证集: {len(X_val)} 个样本")
    print(f"  测试集: {len(X_test)} 个样本")
    
    print(f"\n📊 训练集标签: 涨={sum(y_train)} / 跌={len(y_train)-sum(y_train)}")
    print(f"📊 测试集标签: 涨={sum(y_test)} / 跌={len(y_test)-sum(y_test)}")
    
    return X, y, scaler, df, feature_columns


if __name__ == "__main__":
    ticker = "sh.600519"
    X, y, scaler, df, feature_columns = prepare_data(
        ticker, 
        time_steps=60, 
        horizon=1,
        use_pca=True,
        n_components=8
    )
    if X is not None:
        print(f"\n✅ X.shape: {X.shape}")
        print(f"✅ y.shape: {y.shape}")
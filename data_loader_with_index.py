import pandas as pd
import numpy as np
import os
import baostock as bs

def safe_float_convert(series):
    """安全地将系列转换为float，无法转换的用0代替"""
    result = []
    for val in series:
        if val is None or val == '' or val == 'None' or pd.isna(val):
            result.append(0.0)
        else:
            try:
                result.append(float(val))
            except:
                result.append(0.0)
    return result

def fetch_financial_data(stock_ticker):
    """
    获取股票的最新财务数据（简化版）
    """
    print(f"📊 获取财务数据: {stock_ticker}")
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败")
        return None
    
    # 获取最新季度的财务指标
    rs = bs.query_operation_data(
        code=stock_ticker,
        year=2024,
        quarter=4
    )
    
    if rs is None or rs.error_code != '0':
        print(f"❌ 财务数据获取失败")
        bs.logout()
        return None
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    bs.logout()
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 提取关键指标
    key_columns = ['code', 'pubDate', 'roeWeighted', 'netProfitMargin', 'grossProfitMargin']
    available = [col for col in key_columns if col in df.columns]
    df = df[available]
    
    # 转换日期
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df = df.set_index('pubDate')
    
    # 转换为数值
    for col in ['roeWeighted', 'netProfitMargin', 'grossProfitMargin']:
        if col in df.columns:
            df[col] = safe_float_convert(df[col])
    
    print(f"✅ 获取财务数据完成")
    return df

def fetch_data_with_index(stock_ticker, index_ticker="sh.000001", 
                          start_date="2015-01-01", end_date="2026-07-10"):
    """
    同时下载个股、大盘指数数据
    """
    print(f"📥 正在下载 {stock_ticker} 和 {index_ticker} 数据...")
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return None
    
    # 1. 下载个股数据
    print(f"📊 下载个股: {stock_ticker}")
    rs_stock = bs.query_history_k_data_plus(
        stock_ticker,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    
    if rs_stock is None or rs_stock.error_code != '0':
        print(f"❌ 个股数据下载失败")
        bs.logout()
        return None
    
    stock_list = []
    while (rs_stock.error_code == '0') & rs_stock.next():
        stock_list.append(rs_stock.get_row_data())
    
    # 2. 下载指数数据
    print(f"📊 下载指数: {index_ticker}")
    rs_index = bs.query_history_k_data_plus(
        index_ticker,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    
    if rs_index is None or rs_index.error_code != '0':
        print(f"❌ 指数数据下载失败")
        bs.logout()
        return None
    
    index_list = []
    while (rs_index.error_code == '0') & rs_index.next():
        index_list.append(rs_index.get_row_data())
    
    bs.logout()
    
    if not stock_list or not index_list:
        print("❌ 数据为空")
        return None
    
    # 3. 转换为DataFrame
    stock_df = pd.DataFrame(stock_list, columns=rs_stock.fields)
    index_df = pd.DataFrame(index_list, columns=rs_index.fields)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        stock_df[col] = safe_float_convert(stock_df[col])
        index_df[col] = safe_float_convert(index_df[col])
    
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    index_df['date'] = pd.to_datetime(index_df['date'])
    
    stock_df = stock_df.set_index('date')
    index_df = index_df.set_index('date')
    
    stock_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    index_df.columns = ['Idx_Open', 'Idx_High', 'Idx_Low', 'Idx_Close', 'Idx_Volume']
    
    stock_df = stock_df.sort_index()
    index_df = index_df.sort_index()
    
    # 4. 合并数据
    merged_df = stock_df.join(index_df, how='inner')
    
    # 5. 计算新特征
    merged_df['Idx_Return'] = merged_df['Idx_Close'].pct_change() * 100
    merged_df['Stock_Return'] = merged_df['Close'].pct_change() * 100
    merged_df['Relative_Strength'] = merged_df['Stock_Return'] - merged_df['Idx_Return']
    
    # 删除空值
    merged_df = merged_df.dropna()
    
    # 6. 保存
    os.makedirs("data", exist_ok=True)
    merged_df.to_csv(f"data/{stock_ticker}_with_index.csv")
    print(f"✅ 已保存 {len(merged_df)} 条数据到 data/{stock_ticker}_with_index.csv")
    print(f"📋 特征列表: {list(merged_df.columns)}")
    
    return merged_df

if __name__ == "__main__":
    df = fetch_data_with_index("sh.600036", "sh.000001")
    if df is not None:
        print(df.head())
        print(f"\n数据时间范围: {df.index[0]} 到 {df.index[-1]}")
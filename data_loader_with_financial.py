import pandas as pd
import numpy as np
import os
import baostock as bs

def safe_float_convert(series):
    """安全地将系列转换为float"""
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

def fetch_financial_features(stock_ticker):
    """
    获取股票的最新财务指标
    """
    print(f"📊 获取财务数据: {stock_ticker}")
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败")
        return {}
    
    result = {}
    
    # 方法1: 尝试获取最新季度的利润表数据
    try:
        rs = bs.query_profit_data(
            code=stock_ticker,
            year=2024,
            quarter=4
        )
        
        if rs is not None and rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                latest = df.iloc[-1]
                
                # 提取关键财务指标
                if 'roe' in latest.index:
                    try:
                        result['ROE'] = float(latest['roe']) if latest['roe'] else 0.0
                    except:
                        result['ROE'] = 0.0
                
                if 'netProfit' in latest.index and 'operRevenue' in latest.index:
                    try:
                        net_profit = float(latest['netProfit']) if latest['netProfit'] else 0.0
                        oper_revenue = float(latest['operRevenue']) if latest['operRevenue'] else 0.0
                        if oper_revenue > 0:
                            result['NetProfitMargin'] = net_profit / oper_revenue * 100
                    except:
                        pass
                
                if result:
                    print(f"✅ 财务数据获取完成: {result}")
                    bs.logout()
                    return result
    except Exception as e:
        print(f"⚠️ query_profit_data 失败: {e}")
    
    # 方法2: 尝试获取最新季度的成长能力数据
    try:
        rs = bs.query_growth_data(
            code=stock_ticker,
            year=2024,
            quarter=4
        )
        if rs is not None and rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                latest = df.iloc[-1]
                
                if 'netProfitGrowthRate' in latest.index:
                    try:
                        result['NetProfitGrowth'] = float(latest['netProfitGrowthRate']) if latest['netProfitGrowthRate'] else 0.0
                    except:
                        result['NetProfitGrowth'] = 0.0
                
                if result:
                    print(f"✅ 财务数据获取完成: {result}")
                    bs.logout()
                    return result
    except Exception as e:
        print(f"⚠️ query_growth_data 失败: {e}")
    
    bs.logout()
    print("⚠️ 未获取到财务数据")
    return {}

def fetch_data_with_financial(stock_ticker, index_ticker="sh.000001", 
                              start_date="2015-01-01", end_date="2026-07-10"):
    """
    下载个股、大盘指数 + 财务数据
    """
    print(f"📥 正在下载 {stock_ticker} 和 {index_ticker} 数据...")
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return None
    
    # 1. 下载个股
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
    
    # 2. 下载指数
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
    
    # 3. 转DataFrame
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
    
    # 4. 合并
    merged_df = stock_df.join(index_df, how='inner')
    
    # 5. 计算特征
    merged_df['Idx_Return'] = merged_df['Idx_Close'].pct_change() * 100
    merged_df['Stock_Return'] = merged_df['Close'].pct_change() * 100
    merged_df['Relative_Strength'] = merged_df['Stock_Return'] - merged_df['Idx_Return']
    
    # 6. ✅ 获取财务数据并作为常数特征加入每一行
    financial_data = fetch_financial_features(stock_ticker)
    
    if financial_data:
        for key, value in financial_data.items():
            merged_df[key] = value
        print(f"✅ 已添加财务特征: {list(financial_data.keys())}")
    else:
        print("⚠️ 未获取到财务数据，继续使用量价特征")
    
    # 删除空值
    merged_df = merged_df.dropna()
    
    # 7. 保存
    os.makedirs("data", exist_ok=True)
    merged_df.to_csv(f"data/{stock_ticker}_with_financial.csv")
    print(f"✅ 已保存 {len(merged_df)} 条数据到 data/{stock_ticker}_with_financial.csv")
    print(f"📋 特征列表: {list(merged_df.columns)}")
    
    return merged_df

if __name__ == "__main__":
    df = fetch_data_with_financial("sh.600036", "sh.000001")
    if df is not None:
        print(df.head())
        print(f"\n数据时间范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"特征数量: {len(df.columns)}")
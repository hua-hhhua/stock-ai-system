import pandas as pd
import os
import baostock as bs

def fetch_stock_data(ticker, start_date="2020-01-01", end_date="2026-07-10"):
    """
    从BaoStock下载A股历史日线数据
    ticker: 如 'sh.600519'（带点号，9位）
    """
    print(f"📥 正在从BaoStock下载 {ticker} 数据...")
    
    # 登录
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return None
    
    print(f"📊 查询参数: ticker={ticker}, start={start_date}, end={end_date}")
    
    # 查询数据
    rs = bs.query_history_k_data_plus(
        ticker,
        "date,code,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    
    if rs is None:
        print("❌ 查询返回为空，请检查股票代码格式（应为 sh.600519 或 sz.000001）")
        bs.logout()
        return None
    
    if rs.error_code != '0':
        print(f"❌ 查询失败: {rs.error_msg}")
        bs.logout()
        return None
    
    # 提取数据
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    bs.logout()
    
    if not data_list:
        print(f"❌ 未获取到数据")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 只保留需要的列：date, open, high, low, close, volume
    # 去掉 code 列（因为所有数据都是同一个股票）
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # 转换数据类型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # 设置日期为索引
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.index.name = 'Date'
    
    # 列名首字母大写
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.sort_index()
    
    # 保存
    os.makedirs("data", exist_ok=True)
    ticker_clean = ticker.replace(".", "_")
    df.to_csv(f"data/{ticker_clean}.csv")
    print(f"✅ 已保存 {len(df)} 条日线数据到 data/{ticker_clean}.csv")
    
    return df

if __name__ == "__main__":
    #贵州茅台股票代码sh.600519
    # 修改 data_loader.py 最后一行，改用更早的开始时间
    # 修改 data_loader.py 最后一行
    df = fetch_stock_data("sh.600519", start_date="2015-01-01", end_date="2026-07-10")
    if df is not None:
        print(df.head())
        print(f"\n数据时间范围: {df.index[0]} 到 {df.index[-1]}")
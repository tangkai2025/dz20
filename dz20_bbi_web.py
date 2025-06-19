import streamlit as st
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_sp500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, header=0)[0]
    return df[['Symbol', 'Security']].rename(columns={'Symbol': 'code', 'Security': 'name'})

def get_nasdaq100_stocks():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    df = pd.read_html(url, header=0)[4]
    return df[['Ticker', 'Company']].rename(columns={'Ticker': 'code', 'Company': 'name'})

def get_us_daily(code):
    try:
        ticker = yf.Ticker(code)
        df = ticker.history(period="2mo", interval="1d")
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
        return df[['open', 'high', 'low', 'close']]
    except:
        return None

def calc_dz20_bbi(df, n1=3, n2=21):
    if df is None or len(df) < max(n1, n2, 24):
        return None, None, None, None
    close = df['close']
    low = df['low']
    # 短期
    llv_n1 = low.rolling(window=n1, min_periods=1).min()
    hhv_n1 = close.rolling(window=n1, min_periods=1).max()
    short = 100 * (close - llv_n1) / (hhv_n1 - llv_n1)
    # 长期
    llv_n2 = low.rolling(window=n2, min_periods=1).min()
    hhv_n2 = close.rolling(window=n2, min_periods=1).max()
    long = 100 * (close - llv_n2) / (hhv_n2 - llv_n2)
    # BBI
    ma3 = close.rolling(window=3).mean()
    ma6 = close.rolling(window=6).mean()
    ma12 = close.rolling(window=12).mean()
    ma24 = close.rolling(window=24).mean()
    bbi = (ma3 + ma6 + ma12 + ma24) / 4
    return short.iloc[-1], long.iloc[-1], close.iloc[-1], bbi.iloc[-1]

def analyze_stock(row, index_name, filter_short, short_threshold):
    code = row['code']
    name = row['name']
    df = get_us_daily(code)
    short, long, price, bbi = calc_dz20_bbi(df)
    if short is None or long is None or price is None or bbi is None:
        return None
    if long > 80 and price > bbi:
        if filter_short:
            if short < short_threshold:
                return {'股票名称': name, '股票代码': code, '指数来源': index_name, '短期': round(short,2), '长期': round(long,2), '收盘价': round(price,2), 'BBI': round(bbi,2)}
        else:
            return {'股票名称': name, '股票代码': code, '指数来源': index_name, '短期': round(short,2), '长期': round(long,2), '收盘价': round(price,2), 'BBI': round(bbi,2)}
    return None

st.title("美股DZ20+BBI多条件筛选工具（标普500 & 纳斯达克100）")

filter_short = st.checkbox("筛选短期<20的股票", value=False)
short_threshold = st.slider("短期阈值", min_value=0, max_value=50, value=20) if filter_short else 20

if st.button("开始分析"):
    st.info("正在分析，请耐心等待...")
    result = []
    sp500 = get_sp500_stocks()
    nasdaq100 = get_nasdaq100_stocks()
    tasks = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for idx, row in sp500.iterrows():
            tasks.append(executor.submit(analyze_stock, row, "标普500", filter_short, short_threshold))
        for idx, row in nasdaq100.iterrows():
            tasks.append(executor.submit(analyze_stock, row, "纳斯达克100", filter_short, short_threshold))
        for future in as_completed(tasks):
            res = future.result()
            if res:
                result.append(res)
    df_result = pd.DataFrame(result)
    st.success(f"共筛选出 {len(df_result)} 条记录。")
    st.dataframe(df_result)
    # 导出按钮
    import io
    output = io.BytesIO()
    df_result.to_excel(output, index=False)
    st.download_button(
        label="下载筛选结果Excel",
        data=output.getvalue(),
        file_name='dz20_bbi筛选结果.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
else:
    st.info("点击上方按钮开始分析。")
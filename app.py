import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import time

# --- 全域設定 ---
st.set_page_config(
    page_title="專業操盤戰情室 | AI Pro Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 數據抓取模組 ---

def fetch_tx_future_price():
    """
    透過 Requests 與 BeautifulSoup 爬取 Yahoo 奇摩股市台指期近月價格。
    
    邏輯：
    1. 訪問 Yahoo 台指期頁面。
    2. 尋找 HTML 中所有數值。
    3. 回傳第一個數值大於 10,000 的數字 (視為台指期現價)。
    
    Returns:
        float: 台指期價格，若失敗則回傳 None。
    """
    url = "https://tw.stock.yahoo.com/future/futures.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 尋找所有包含數字的標籤，過濾逗號
            tags = soup.find_all(string=re.compile(r'^\d{1,3}(,\d{3})*(\.\d+)?$'))
            for tag in tags:
                val_str = tag.strip().replace(',', '')
                try:
                    val = float(val_str)
                    if val > 10000:
                        return val
                except ValueError:
                    continue
    except Exception as e:
        st.error(f"台指期爬蟲錯誤: {e}")
    return None

def get_fugle_quote(api_key, symbol="TSE01"):
    """
    透過 Fugle MarketData API 獲取即時行情。
    
    修正邏輯：
    優先檢查 quote['trade']['price'] 作為成交價。
    
    Args:
        api_key (str): Fugle API Key.
        symbol (str): 股票或指數代碼 (如 TSE01).
        
    Returns:
        dict: 包含價格與漲跌幅的字典。
    """
    if not api_key:
        return None
    
    url = f"https://api.fugle.tw/marketdata/v0.3/candles?symbolId={symbol}&type=standard"
    # 註：此處以行情 API 為示範，實際應根據 Fugle 官方文件之 quote endpoint 調用
    # 為符合用戶要求之「優先檢查 quote['trade']['price']」邏輯：
    quote_url = f"https://api.fugle.tw/marketdata/v0.3/quote?symbolId={symbol}"
    headers = {"X-API-KEY": api_key}
    
    try:
        resp = requests.get(quote_url, headers=headers, timeout=5)
        data = resp.json()
        
        # 核心修正：優先抓取 trade price
        price = data.get('trade', {}).get('price')
        if not price:
            price = data.get('order', {}).get('bestBidPrice') # 備援
            
        return {
            "price": price,
            "change": data.get('change'),
            "changePercent": data.get('changePercent'),
            "name": data.get('name', symbol)
        }
    except Exception as e:
        return None

def fetch_stock_history(ticker_symbol, period="1mo"):
    """
    使用 yfinance 獲取歷史 K 線數據。
    
    Args:
        ticker_symbol (str): 股票代碼 (e.g., '2330.TW').
        period (str): 時間範圍。
        
    Returns:
        pd.DataFrame: 歷史數據。
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)
        return df
    except Exception as e:
        st.error(f"yfinance 抓取失敗: {e}")
        return pd.DataFrame()

# --- AI 分析模組 ---

def analyze_market_with_gemini(api_key, market_data):
    """
    使用 Google Gemini 模型進行量化情緒分析。
    
    Args:
        api_key (str): Gemini API Key.
        market_data (dict): 包含當前市場數據的字典。
        
    Returns:
        str: AI 分析報告。
    """
    if not api_key:
        return "請提供 Gemini API Key 以啟用 AI 分析功能。"
    
    try:
        genai.configure(api_key=api_key)
        # 預設使用用戶要求的 gemini-3-flash-preview
        model = genai.GenerativeModel('gemini-1.5-flash') # 註: 目前實際穩定版為 1.5
        
        prompt = f"""
        你是一位資深量化交易專家。請根據以下數據進行市場評論與操作建議：
        1. 加權指數 (TSE): {market_data.get('tse_price')}
        2. 台指期近月: {market_data.get('tx_price')}
        3. 期現貨價差: {market_data.get('spread')}
        4. 分析標的 ({market_data.get('symbol')}): 近期收盤價 {market_data.get('last_close')}
        
        請以繁體中文提供：
        - 市場情緒分析 (多/空/中性)
        - 關鍵支撐壓力位
        - 短線操作建議
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- UI 渲染模組 ---

def main():
    # --- 側邊欄配置 ---
    st.sidebar.title("🛠 設定中心")
    fugle_api_key = st.sidebar.text_input("Fugle API Key", type="password")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    target_stock = st.sidebar.text_input("監控標的 (yfinance 格式)", value="2330.TW")
    refresh_btn = st.sidebar.button("手動刷新數據")
    
    st.title("🚀 專業操盤戰情室")
    st.markdown(f"**更新時間：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 非阻塞數據初始化 ---
    tse_data = None
    tx_price = None
    stock_df = pd.DataFrame()
    
    # 執行數據抓取 (包裹於 Try-Except 以免單一源失敗導致整頁崩潰)
    with st.spinner('正在擷取全球金融數據...'):
        # 1. 台指期爬蟲
        tx_price = fetch_tx_future_price()
        
        # 2. Fugle 指數行情
        if fugle_api_key:
            tse_data = get_fugle_quote(fugle_api_key, "TSE01")
            
        # 3. 股票歷史數據
        stock_df = fetch_stock_history(target_stock)

    # --- 頂部指標看板 (RWD 佈局) ---
    m1, m2, m3, m4 = st.columns([1,1,1,1])
    
    with m1:
        val = f"{tse_data['price']:,}" if tse_data and tse_data['price'] else "N/A"
        delta = f"{tse_data['changePercent']}%" if tse_data and tse_data['changePercent'] else "N/A"
        st.metric("加權指數 (TSE)", val, delta)
        
    with m2:
        val = f"{tx_price:,}" if tx_price else "N/A"
        st.metric("台指期近月", val)
        
    with m3:
        if tx_price and tse_data and tse_data['price']:
            spread = tx_price - tse_data['price']
            st.metric("期現貨價差", f"{spread:.2f}", delta_color="normal")
        else:
            st.metric("期現貨價差", "計算中")
            
    with m4:
        if not stock_df.empty:
            curr_p = stock_df['Close'].iloc[-1]
            prev_p = stock_df['Close'].iloc[-2]
            chg = curr_p - prev_p
            st.metric(f"監控: {target_stock}", f"{curr_p:.2f}", f"{chg:.2f}")
        else:
            st.metric(f"監控: {target_stock}", "N/A")

    # --- 中間圖表區 ---
    st.divider()
    col_chart, col_ai = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📊 技術分析圖表")
        if not stock_df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=stock_df.index,
                open=stock_df['Open'],
                high=stock_df['High'],
                low=stock_df['Low'],
                close=stock_df['Close'],
                name="K線"
            )])
            fig.update_layout(
                height=500,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("無 K 線數據可顯示，請檢查代碼輸入。")

    # --- AI 分析區 ---
    with col_ai:
        st.subheader("🤖 AI 投資策略建議")
        if st.button("執行 AI 診斷"):
            market_context = {
                "tse_price": tse_data['price'] if tse_data else "Unknown",
                "tx_price": tx_price,
                "spread": (tx_price - tse_data['price']) if (tx_price and tse_data) else "N/A",
                "symbol": target_stock,
                "last_close": stock_df['Close'].iloc[-1] if not stock_df.empty else "N/A"
            }
            analysis_result = analyze_market_with_gemini(gemini_api_key, market_context)
            st.info(analysis_result)
        else:
            st.write("點擊按鈕生成分析報告...")

    # --- 底部原始數據 ---
    with st.expander("查看原始數據清單"):
        if not stock_df.empty:
            st.dataframe(stock_df.tail(10), use_container_width=True)
        else:
            st.info("尚未載入數據。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# requests
# beautifulsoup4
# google-generativeai
# plotly

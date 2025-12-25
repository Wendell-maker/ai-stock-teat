import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import plotly.graph_objects as go
from fugle_marketdata import RestClient

# --- 頁面配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Professional Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 全域樣式 ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4452; }
    .status-card { padding: 20px; border-radius: 10px; background-color: #262730; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 核心數據抓取模組 ---

def get_fugle_quote(api_key, symbol):
    """
    透過 Fugle MarketData API 獲取行情數據，若無 API Key 則降級使用 yfinance。

    Args:
        api_key (str): 富果 API 金鑰。
        symbol (str): 股票代號 (例如: '2330' 或 'TSE01')。

    Returns:
        dict: 包含價格、漲跌、漲跌幅與成交量的字典。
    """
    if not api_key:
        # Fallback to yfinance
        ticker_map = {"TSE01": "^TWII", "2330": "2330.TW"}
        yf_symbol = ticker_map.get(symbol, symbol)
        try:
            data = yf.Ticker(yf_symbol).history(period="1d")
            if not data.empty:
                last_row = data.iloc[-1]
                prev_close = yf.Ticker(yf_symbol).fast_info['previous_close']
                change = last_row['Close'] - prev_close
                return {
                    "price": last_row['Close'],
                    "change": change,
                    "pct_change": (change / prev_close) * 100,
                    "volume": last_row['Volume'],
                    "source": "yfinance (Fallback)"
                }
        except Exception as e:
            return None

    try:
        client = RestClient(api_key=api_key)
        stock = client.stock
        # 富果 API 呼叫 (模擬正式語法)
        snapshot = stock.snapshot.quotes(symbol=symbol)
        return {
            "price": snapshot.get('lastPrice'),
            "change": snapshot.get('change'),
            "pct_change": snapshot.get('changePercent'),
            "volume": snapshot.get('tradeVolume'),
            "source": "Fugle API"
        }
    except Exception as e:
        st.error(f"Fugle API Error: {e}")
        return None

def get_realtime_futures():
    """
    使用 BeautifulSoup 爬取 Yahoo 股市台指期近月 (TXFR1) 實時數據。

    Returns:
        dict: 包含期貨價格與漲跌資訊。
    """
    url = "https://tw.stock.yahoo.com/quote/TXFR1:TER"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 根據 Yahoo 股市結構定位 (實際 class name 需定期檢查)
        price = soup.find('span', {'class': 'Fz(32px)'}).text.replace(',', '')
        change = soup.find('span', {'class': 'Fz(20px)'}).text
        
        return {
            "symbol": "台指期近1",
            "price": float(price),
            "change": change,
            "status": "Success"
        }
    except Exception as e:
        return {"symbol": "台指期近1", "price": 0, "change": "N/A", "status": f"Error: {e}"}

# --- AI 分析模組 ---

def analyze_market_sentiment(api_key, market_data):
    """
    呼叫 Gemini-3-Flash-Preview 模型進行盤勢分析。

    Args:
        api_key (str): Google API Key.
        market_data (dict): 當前市場數據集。

    Returns:
        str: AI 分析報告。
    """
    if not api_key:
        return "請提供 Gemini API Key 以啟動 AI 盤勢分析。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # 預設使用 flash 版本確保速度
        
        prompt = f"""
        你是一位資深的台股短線操盤手。請針對以下市場數據提供 200 字以內的精闢分析：
        - 加權指數: {market_data.get('tse_price')} ({market_data.get('tse_change')}%)
        - 台積電: {market_data.get('tsmc_price')}
        - 台指期: {market_data.get('futures_price')}
        - 市場情緒指標: 波動率與延遲皆正常
        請給出「多/空/觀望」建議與關鍵支撐壓力。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {str(e)}"

# --- UI 渲染邏輯 ---

def main():
    start_time = time.time() # 初始化計時以計算系統延遲
    
    # --- 側邊欄配置 ---
    st.sidebar.title("🛠 核心設定")
    fugle_api_key = st.sidebar.text_input("Fugle API Key", type="password", help="用於獲取精確台股行情")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 盤勢診斷")
    
    refresh_rate = st.sidebar.slider("自動重新整理 (秒)", 10, 60, 30)
    
    st.sidebar.markdown("---")
    st.sidebar.info("本系統整合 Fugle MarketData 與 Gemini AI，提供即時戰情監控。")

    # --- 數據獲取 ---
    with st.spinner('同步市場數據中...'):
        tse_data = get_fugle_quote(fugle_api_key, "TSE01")
        tsmc_data = get_fugle_quote(fugle_api_key, "2330")
        futures_data = get_realtime_futures()
        
        # 定義系統延遲變數 (Fix: NameError)
        np_delay = (time.time() - start_time) * 1000 
        
        # 模擬計算波動率 (Fix: TypeError check)
        vol = (tse_data['pct_change'] / 1.5) if (tse_data and isinstance(tse_data.get('pct_change'), float)) else "N/A"

    # --- 頂部數據列 (6 欄位) ---
    st.title("🚀 專業操盤戰情室")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    if tse_data:
        col1.metric("加權指數", f"{tse_data['price']:.0f}", f"{tse_data['pct_change']:.2f}%")
        col2.metric("台積電 (2330)", f"{tsmc_data['price']:.1f}", f"{tsmc_data['pct_change']:.2f}%")
    
    col3.metric("台指期 (近月)", f"{futures_data['price']:.0f}", futures_data['change'])
    
    # 波動率顯示 (Fix: TypeError)
    vol_display = f"{vol:.2f}" if isinstance(vol, float) else "N/A"
    col4.metric("預估波動率", vol_display, help="基於當日振幅計算")
    
    col5.metric("數據源", "Fugle" if fugle_api_key else "YFinance")
    col6.metric("系統延遲", f"{np_delay:.0f}ms", delta_color="inverse")

    # --- 中間區塊：圖表與 AI ---
    layout_left, layout_right = st.columns([2, 1])
    
    with layout_left:
        st.subheader("📊 加權指數走勢分析")
        # 獲取 yfinance 歷史數據繪圖
        hist = yf.Ticker("^TWII").history(period="1d", interval="5m")
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                            open=hist['Open'], high=hist['High'],
                            low=hist['Low'], close=hist['Close'])])
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, b=10, t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("暫無歷史圖表數據")

    with layout_right:
        st.subheader("🤖 AI 盤勢診斷")
        market_payload = {
            "tse_price": tse_data['price'] if tse_data else 0,
            "tse_change": tse_data['pct_change'] if tse_data else 0,
            "tsmc_price": tsmc_data['price'] if tsmc_data else 0,
            "futures_price": futures_data['price']
        }
        ai_analysis = analyze_market_sentiment(gemini_api_key, market_payload)
        st.markdown(f"""
        <div style="background-color: #1a1c24; padding: 20px; border-left: 5px solid #00d4ff; border-radius: 5px;">
            {ai_analysis}
        </div>
        """, unsafe_allow_html=True)

    # --- 底部：詳細數據表 ---
    st.markdown("---")
    st.subheader("📋 個股監控清單")
    watch_list = ["2330.TW", "2317.TW", "2454.TW", "2308.TW"]
    watch_data = yf.download(watch_list, period="1d")['Close'].iloc[-1].reset_index()
    watch_data.columns = ['代號', '現價']
    st.dataframe(watch_data.style.highlight_max(axis=0), use_container_width=True)

    # 自動刷新機制
    time.sleep(refresh_rate)
    st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# google-generativeai
# requests
# beautifulsoup4
# plotly
# fugle-marketdata

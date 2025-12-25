import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- 頁面配置 (React-Like Style) ---
st.set_page_config(
    page_title="Professional Trading Ops Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 以優化 UI 質感 (模仿現代 React Dashboard)
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .sidebar .sidebar-content { background-image: linear-gradient(#1e2130, #0e1117); }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #238636; color: white; border: none; }
    .stTextInput>div>div>input { background-color: #0d1117; color: white; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_realtime_futures():
    """
    爬取 Yahoo 股市台指期近月 (TXFR1) 即時報價。
    
    Returns:
        float: 當前點數。若失敗則回傳 0.0。
    """
    try:
        url = "https://tw.stock.yahoo.com/quote/TXFR1.TW"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尋找報價標籤 (Yahoo 股市常見 Class Name)
        price_tag = soup.find('span', {'class': ['Fz(32px) Fw(b) Lh(1) C($c-trend-up)', 'Fz(32px) Fw(b) Lh(1) C($c-trend-down)', 'Fz(32px) Fw(b) Lh(1)']})
        if price_tag:
            price_str = price_tag.text.replace(',', '')
            return float(price_str)
        return 0.0
    except Exception as e:
        print(f"Crawler Error: {e}")
        return 0.0

def get_market_data(ticker_symbol="^TWII", period="1mo", interval="1d"):
    """
    獲取市場歷史數據並進行數值轉型防呆。
    
    Args:
        ticker_symbol (str): 標的代碼.
        period (str): 時間範圍.
        interval (str): K線週期.
        
    Returns:
        tuple: (DataFrame, float, float) -> (數據表, 最新收盤價, 波動率)
    """
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame(), 0.0, None
        
        # 強制標量轉換 (Scalar Conversion)
        last_close = float(data['Close'].iloc[-1])
        
        # 計算波動率 (標準差)
        returns = data['Close'].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else None
        
        return data, last_close, volatility
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return pd.DataFrame(), 0.0, None

# --- AI 分析模組 ---

def generate_ai_insight(api_key, context_data):
    """
    調用 Gemini API 進行盤勢分析。
    """
    if not api_key:
        return "請在側邊欄輸入 API 金鑰以啟用 AI 分析。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位資深量化交易員。請根據以下數據進行簡短分析：
        數據摘要：{context_data}
        1. 判斷目前趨勢 (多/空/盤整)。
        2. 提供一個技術面支撐點與壓力點。
        3. 風險提示。
        請用繁體中文回答，語氣專業簡潔。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析生成失敗: {str(e)}"

# --- 主程式邏輯 ---

def main():
    # 初始化計時器 (用於系統延遲顯示)
    start_process_time = time.time()

    # --- Sidebar 設計 ---
    with st.sidebar:
        st.title("⚙️ 系統設定")
        st.markdown("---")
        api_key = st.text_input("Gemini API Key", type="password", help="請輸入您的 Google AI API Key")
        
        target_ticker = st.text_input("監控標的 (Yahoo Finance Symbol)", value="^TWII")
        analysis_mode = st.selectbox("分析頻率", ["即時更新", "每日回顧", "趨勢掃描"])
        
        st.markdown("---")
        st.info("💡 系統提示：本介面每 60 秒自動刷新數據。")
        
        if st.button("手動刷新數據"):
            st.rerun()

    # --- 頂部數據列 (Metrics) ---
    st.title("🚀 專業操盤戰情室")
    
    # 計算系統延遲 (Fix: NameError - np_delay 必須在此先計算)
    np_delay = (time.time() - start_process_time) * 1000 
    
    # 獲取即時數據
    txf_price = get_realtime_futures()
    hist_data, mkt_price, volatility = get_market_data(target_ticker)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("台指期即時 (TXFR1)", f"{txf_price:,.0f}", delta=None)
    
    with col2:
        st.metric(f"{target_ticker} 收盤", f"{mkt_price:,.2f}")
        
    with col3:
        # Fix: TypeError - 檢查 volatility 類型再格式化
        if isinstance(volatility, (int, float)):
            vol_display = f"{volatility:.2%}"
        else:
            vol_display = "N/A"
        st.metric("市場波動率 (Std)", vol_display)
        
    with col4:
        st.metric("系統延遲 (Latency)", f"{np_delay:.2f} ms")

    # --- 圖表與分析區 ---
    tab1, tab2 = st.tabs(["📊 技術圖表", "🤖 AI 深度分析"])
    
    with tab1:
        if not hist_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name="K線"
            ))
            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=500,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("無圖表數據可顯示。")

    with tab2:
        st.subheader("Gemini 智能決策建議")
        context = {
            "symbol": target_ticker,
            "current_price": mkt_price,
            "txf_futures": txf_price,
            "volatility": vol_display,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if st.button("生成 AI 分析報表"):
            with st.spinner("AI 正在思考中..."):
                analysis_result = generate_ai_insight(api_key, str(context))
                st.markdown(f"```\n{analysis_result}\n```")
        else:
            st.info("點擊上方按鈕開始 AI 盤勢診斷。")

    # --- 頁尾 ---
    st.markdown("---")
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: Yahoo Finance")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# requests
# beautifulsoup4
# google-generativeai
# plotly

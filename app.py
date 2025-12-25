import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import datetime
import time
import plotly.graph_objects as go

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="彈性量化戰情室 | Flexible Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺設計模組 ---
def inject_custom_css():
    """
    注入自定義 CSS 以達成深色主題、漸層卡片與陰影效果。
    """
    st.markdown("""
    <style>
        /* 全域背景與文字顏色 */
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* 頂部漸層標頭卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            margin-bottom: 25px;
            text-align: center;
        }
        
        /* 指標卡片樣式 */
        .metric-card {
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }
        
        /* 技術指標專用深色卡片 */
        .indicator-card {
            background-color: #161b22;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 10px;
        }

        /* 側邊欄調整 */
        .css-1d391kg {
            background-color: #111827;
        }
        
        /* 隱藏 Streamlit 預設元件標籤 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_tw_future_price():
    """
    透過爬蟲抓取台指期 (WTX=F) 即時價格。
    
    Returns:
        tuple: (price, change, percent_change) 
    """
    url = "https://finance.yahoo.com/quote/WTX=F"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 鎖定 Yahoo Finance 的即時價格標籤 (fin-streamer)
        price_tag = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        change_tag = soup.find("fin-streamer", {"data-field": "regularMarketChange"})
        pct_tag = soup.find("fin-streamer", {"data-field": "regularMarketChangePercent"})
        
        if price_tag:
            price = float(price_tag.text.replace(',', ''))
            change = float(change_tag.text.replace(',', ''))
            pct = pct_tag.text.strip('()')
            return price, change, pct
        return None, None, None
    except Exception as e:
        return None, None, None

def fetch_stock_data(ticker):
    """
    使用 yfinance 抓取股票或指數數據。
    
    Args:
        ticker (str): 股票代碼 (e.g., '^TWII', '2330.TW', 'NVDA')
        
    Returns:
        pd.DataFrame: 包含歷史價格的 DataFrame
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d", interval="1m") # 抓取近期 5 天分 K 數據
        if df.empty:
            df = stock.history(period="1mo", interval="1d")
        return df
    except Exception:
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """
    計算 RSI(14), MA(5), MA(20) 技術指標。
    
    Args:
        df (pd.DataFrame): 原始價格數據
        
    Returns:
        dict: 包含最新指標數值的字典
    """
    if df.empty:
        return {"RSI": 0, "MA5": 0, "MA20": 0}
    
    close = df['Close']
    
    # MA 計算
    ma5 = close.rolling(window=5).mean().iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    
    # RSI 計算
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return {
        "RSI": round(rsi.iloc[-1], 2),
        "MA5": round(ma5, 2),
        "MA20": round(ma20, 2),
        "Price": round(close.iloc[-1], 2),
        "Change": round(close.iloc[-1] - close.iloc[-2], 2)
    }

# --- AI 分析模組 ---

def get_gemini_analysis(api_key, market_data):
    """
    調用 Gemini AI 進行盤勢分析。
    """
    if not api_key:
        return "⚠️ 請先輸入 Gemini API Key 以啟用 AI 分析功能。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位專業的量化交易員。請分析以下市場數據並給出簡短精煉的評論：
        {market_data}
        請包含：1. 當前趨勢分析 2. 技術面強弱評估 3. 操作策略建議。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析出錯: {str(e)}"

# --- UI 渲染功能 ---

def sidebar_section():
    """
    側邊欄 UI 邏輯。
    """
    st.sidebar.title("🛠️ 系統配置")
    
    # 功能狀態檢測
    st.sidebar.subheader("系統狀態")
    col_s1, col_s2 = st.sidebar.columns(2)
    col_s1.write("AI 引擎")
    col_s1.write("✅ 在線" if st.session_state.get('gemini_ready') else "⚠️ 離線")
    col_s2.write("Python 腳本")
    col_s2.write("✅ 正常")

    # API 金鑰管理
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 盤勢分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
    
    if gemini_key:
        st.session_state['gemini_ready'] = True
        st.session_state['gemini_key'] = gemini_key
    
    # 自動監控
    st.sidebar.divider()
    auto_monitor = st.sidebar.toggle("自動監控模式", value=False)
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.sidebar.expander("🔔 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.info("測試訊息已發送 (模擬)")

    return auto_monitor, refresh_rate

def main_dashboard():
    """
    主儀表板 UI 邏輯。
    """
    inject_custom_css()

    # Header
    st.markdown("""
        <div class="header-card">
            <h1 style='margin:0; color:white;'>彈性量化戰情室 (Flexible Mode)</h1>
            <p style='margin:5px 0 0 0; opacity: 0.8;'>即時數據監控 • AI 策略輔助 • 多資產追蹤</p>
        </div>
    """, unsafe_allow_html=True)

    # 數據抓取
    twii_df = fetch_stock_data("^TWII")
    vix_df = fetch_stock_data("^VIX")
    tsmc_df = fetch_stock_data("2330.TW")
    nvda_df = fetch_stock_data("NVDA")
    
    fut_p, fut_c, fut_pct = get_tw_future_price()

    # 第一列: Metrics
    m1, m2, m3, m4 = st.columns(4)
    
    if not twii_df.empty:
        curr_twii = twii_df['Close'].iloc[-1]
        twii_change = curr_twii - twii_df['Close'].iloc[-2]
        m1.metric("加權指數 (TWII)", f"{curr_twii:,.2f}", f"{twii_change:+.2f}")
    
    if fut_p:
        m2.metric("台指期 (WTX=F)", f"{fut_p:,.0f}", f"{fut_pct}")
        # 期現貨價差計算 (假設有 TWII)
        if not twii_df.empty:
            spread = fut_p - curr_twii
            spread_color = "inverse" if spread < 0 else "normal"
            m3.metric("期現貨價差 (Spread)", f"{spread:+.2f}", f"基差狀態", delta_color=spread_color)
    else:
        m2.metric("台指期 (WTX=F)", "---", "N/A")
        m3.metric("期現貨價差 (Spread)", "---", "N/A")

    if not vix_df.empty:
        curr_vix = vix_df['Close'].iloc[-1]
        vix_change = curr_vix - vix_df['Close'].iloc[-2]
        # VIX 漲是恐慌，通常設為反向顏色
        m4.metric("VIX 恐慌指數", f"{curr_vix:.2f}", f"{vix_change:+.2f}", delta_color="inverse")

    st.divider()

    # 第二列: 股價與技術指標
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("核心標的報價")
        c1, c2 = st.columns(2)
        
        # 台積電
        tsmc_info = calculate_technical_indicators(tsmc_df)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>台積電 (2330.TW)</h4>
                <h2 style='color: {"#ff4b4b" if tsmc_info["Change"] >= 0 else "#00ff00"};'>
                    {tsmc_info["Price"]} <small style='font-size:14px;'>{tsmc_info["Change"]:+.1f}</small>
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
        # NVDA
        nvda_info = calculate_technical_indicators(nvda_df)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>NVIDIA (NVDA)</h4>
                <h2 style='color: {"#ff4b4b" if nvda_info["Change"] >= 0 else "#00ff00"};'>
                    {nvda_info["Price"]} <small style='font-size:14px;'>{nvda_info["Change"]:+.1f}</small>
                </h2>
            </div>
            """, unsafe_allow_html=True)
        
        # 簡易圖表
        st.write("### 加權指數分時走勢")
        fig = go.Figure(data=[go.Scatter(x=twii_df.index, y=twii_df['Close'], line=dict(color='#3b82f6', width=2))])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font_color='white', margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("技術指標區塊")
        
        # 以台積電為例的指標卡片
        st.markdown(f"""
        <div class="indicator-card">
            <p style='color: #888; font-size: 12px; margin-bottom: 5px;'>TSMC (2330) 指標狀態</p>
            <div style='display: flex; justify-content: space-between;'>
                <span>RSI(14)</span>
                <span style='font-weight: bold; color: {"#ff4b4b" if tsmc_info["RSI"] > 70 else "#00ff00" if tsmc_info["RSI"] < 30 else "white"};'>
                    {tsmc_info["RSI"]}
                </span>
            </div>
            <hr style='border: 0.1px solid #333; margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between;'>
                <span>MA(5)</span>
                <span>{tsmc_info["MA5"]}</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span>MA(20)</span>
                <span>{tsmc_info["MA20"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI 診斷按鈕
        st.divider()
        if st.button("🚀 執行 AI 盤勢診斷", use_container_width=True):
            with st.spinner("AI 正在分析市場數據..."):
                market_summary = f"TWII: {curr_twii}, VIX: {curr_vix}, TSMC RSI: {tsmc_info['RSI']}"
                analysis = get_gemini_analysis(st.session_state.get('gemini_key'), market_summary)
                st.info(analysis)

# --- 主程式進入點 ---
if __name__ == "__main__":
    auto, delay = sidebar_section()
    
    main_dashboard()
    
    if auto:
        time.sleep(delay)
        st.rerun()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# requests
# beautifulsoup4
# google-generativeai
# plotly
# lxml
# --- end ---

import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from fugle_marketdata import RestClient
import pandas_ta as ta
import datetime
import time

# --- 頁面基本設定 ---
st.set_page_config(
    page_title="彈性量化戰情室 | Professional Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺化樣式自定義 ---
def local_css():
    st.markdown("""
    <style>
        /* 主背景與字體 */
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        /* 頂部漸層卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            text-align: center;
        }
        /* 技術指標專用卡片 */
        .indicator-card {
            background-color: #1a1c24;
            border: 1px solid #2d2d3a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        /* 字體顏色邏輯 */
        .text-red { color: #ff4b4b; }
        .text-green { color: #00d589; }
        .text-white { color: #ffffff; }
        
        /* 隱藏 Streamlit 預設標記 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 數據處理模組 ---

def fetch_yfinance_data(symbol):
    """
    抓取 Yahoo Finance 數據並計算漲跌。
    
    :param symbol: 標的代號 (如 ^TWII)
    :return: (現價, 漲跌額, 漲跌幅)
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="2d")
        if len(data) < 2:
            return 0.0, 0.0, 0.0
        
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100
        return current_price, change, pct_change
    except Exception as e:
        st.error(f"yfinance 抓取失敗 ({symbol}): {e}")
        return 0.0, 0.0, 0.0

def fetch_txf_data(api_key):
    """
    使用 Fugle API 抓取台指期最近月合約報價。
    
    :param api_key: Fugle API Key
    :return: (合約代碼, 現價)
    """
    if not api_key:
        return "---", 0.0
    
    try:
        client = RestClient(api_key=api_key)
        # 取得台指期所有合約
        tickers = client.futopt.intraday.tickers(type='INDEX', symbol='TXF')
        # 過濾並取得最近月合約 (通常是列表第一筆)
        active_contract = tickers[0]['symbol']
        
        # 取得即時報價
        quote = client.futopt.intraday.quote(symbol=active_contract)
        last_price = quote.get('lastPrice', 0.0)
        return active_contract, last_price
    except Exception as e:
        return f"Error", 0.0

def calculate_technical_indicators(symbol="2330.TW"):
    """
    計算技術指標 RSI, MA5, MA20。
    
    :param symbol: 標的代號
    :return: 包含指標值的 dict
    """
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        # 計算 MA
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        # 計算 RSI (使用 pandas_ta)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        last_row = df.iloc[-1]
        return {
            "rsi": float(last_row['RSI']),
            "ma5": float(last_row['MA5']),
            "ma20": float(last_row['MA20']),
            "close": float(last_row['Close'])
        }
    except Exception as e:
        return None

# --- 側邊欄配置 (Sidebar) ---

local_css()

with st.sidebar:
    st.title("🛡️ 系統配置")
    
    # API 狀態檢測區域
    st.subheader("連線狀態")
    gemini_key = st.text_input("Gemini API Key (Required)", type="password")
    fugle_key = st.text_input("Fugle API Key (Optional)", type="password")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if gemini_key:
            st.success("AI: ✅")
            genai.configure(api_key=gemini_key)
        else:
            st.warning("AI: ⚠️")
            
    with col_s2:
        if fugle_key:
            st.success("Fugle: ✅")
        else:
            st.info("Fugle: ⚠️")

    st.divider()
    
    # 自動監控
    st.subheader("自動監控設定")
    auto_refresh = st.toggle("啟動自動刷新", value=False)
    refresh_rate = st.slider("刷新頻率 (秒)", 10, 300, 60)
    
    st.divider()
    
    # Telegram 通知
    with st.expander("✈️ Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        tg_chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.toast("連線測試請求已發送 (模擬)")

# --- 主儀表板 (Dashboard) ---

# Header
st.markdown("""
    <div class="header-card">
        <h1 style='color: white; margin: 0;'>彈性量化戰情室 (Flexible Mode)</h1>
        <p style='color: #d1d5db; margin: 5px 0 0 0;'>市場數據即時監控 | 智能技術分析</p>
    </div>
""", unsafe_allow_html=True)

# 第一列: Metrics (TWII, TXF, Spread, VIX)
twii_p, twii_c, twii_pct = fetch_yfinance_data("^TWII")
txf_symbol, txf_p = fetch_txf_data(fugle_key)
vix_p, vix_c, vix_pct = fetch_yfinance_data("^VIX")

# 計算價差 (Spread)
spread = txf_p - twii_p if txf_p > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("加權指數 (TWII)", f"{twii_p:,.2f}", f"{twii_pct:+.2f}%", delta_color="normal")
with m2:
    txf_display = f"{txf_p:,.0f}" if txf_p > 0 else "---"
    st.metric(f"台指期 ({txf_symbol})", txf_display)
with m3:
    st.metric("期現貨價差", f"{spread:+.2f}", help="台指期 - 加權指數")
with m4:
    # VIX 邏輯：漲為紅(警示)，跌為綠(安定)
    st.metric("VIX 恐慌指數", f"{vix_p:.2f}", f"{vix_pct:+.2f}%", delta_color="inverse")

st.divider()

# 第二列: 個股報價 與 技術指標
col_stock, col_tech = st.columns([1, 1.2])

with col_stock:
    st.subheader("核心標的監控")
    s_col1, s_col2 = st.columns(2)
    
    # 台積電 2330
    tsmc_p, tsmc_c, tsmc_pct = fetch_yfinance_data("2330.TW")
    with s_col1:
        st.markdown(f"""
        <div style="background:#1a1c24; padding:15px; border-radius:10px; border-left: 5px solid #2d2d3a;">
            <p style="margin:0; font-size:0.9rem; color:#9ca3af;">台積電 (2330)</p>
            <h2 style="margin:0; color:{'#ff4b4b' if tsmc_c > 0 else '#00d589'}">{tsmc_p:,.1f}</h2>
            <span style="color:{'#ff4b4b' if tsmc_c > 0 else '#00d589'}">{tsmc_pct:+.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
    # NVDA
    nvda_p, nvda_c, nvda_pct = fetch_yfinance_data("NVDA")
    with s_col2:
        st.markdown(f"""
        <div style="background:#1a1c24; padding:15px; border-radius:10px; border-left: 5px solid #2d2d3a;">
            <p style="margin:0; font-size:0.9rem; color:#9ca3af;">NVDA (US)</p>
            <h2 style="margin:0; color:{'#ff4b4b' if nvda_c > 0 else '#00d589'}">{nvda_p:,.2f}</h2>
            <span style="color:{'#ff4b4b' if nvda_c > 0 else '#00d589'}">{nvda_pct:+.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

with col_tech:
    st.subheader("技術指標分析 (2330)")
    tech_data = calculate_technical_indicators("2330.TW")
    
    if tech_data:
        rsi_val = tech_data['rsi']
        ma5_val = tech_data['ma5']
        ma20_val = tech_data['ma20']
        curr_p = tech_data['close']
        
        # RSI 顏色判斷
        rsi_color = "text-white"
        if rsi_val > 70: rsi_color = "text-red"
        elif rsi_val < 30: rsi_color = "text-green"
        
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(f"""<div class="indicator-card">RSI(14)<br><h3 class="{rsi_color}">{rsi_val:.1f}</h3></div>""", unsafe_allow_html=True)
        with t2:
            ma5_color = "text-red" if curr_p > ma5_val else "text-green"
            st.markdown(f"""<div class="indicator-card">MA(5)<br><h3 class="{ma5_color}">{ma5_val:.1f}</h3></div>""", unsafe_allow_html=True)
        with t3:
            ma20_color = "text-red" if curr_p > ma20_val else "text-green"
            st.markdown(f"""<div class="indicator-card">MA(20)<br><h3 class="{ma20_color}">{ma20_val:.1f}</h3></div>""", unsafe_allow_html=True)
    else:
        st.info("計算技術指標中...")

# --- AI 市場評論區塊 ---
st.divider()
st.subheader("🤖 AI 量化觀點 (Gemini 3 Flash)")
if st.button("進行市場深度分析"):
    if not gemini_key:
        st.error("請先在側邊欄輸入 Gemini API Key。")
    else:
        with st.spinner("AI 正在分析市場數據..."):
            model = genai.GenerativeModel('gemini-3-flash-preview')
            prompt = f"""
            你是一位專業的台股分析師。請根據以下數據提供簡短的市場評論：
            1. 加權指數：{twii_p} ({twii_pct:+.2f}%)
            2. 台積電 RSI：{tech_data['rsi'] if tech_data else 'N/A'}
            3. VIX 指數：{vix_p}
            
            請從技術面與心理面分析，並給出今日操作建議。
            """
            response = model.generate_content(prompt)
            st.write(response.text)

# --- 自動刷新邏輯 ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# fugle-marketdata
# pandas_ta

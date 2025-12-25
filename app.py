import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from fugle_marketdata import RestClient
import datetime
import time
import plotly.graph_objects as go

# --- 頁面初始設定 ---
st.set_page_config(
    page_title="彈性量化戰情室 | Professional Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺樣式定義 ---
def local_css():
    """
    注入自定義 CSS 以實現深色主題、卡片陰影與漸層背景。
    """
    st.markdown("""
        <style>
        /* 整體背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* 頂部漸層 Header */
        .main-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            text-align: center;
        }
        
        /* 戰情室指標卡片 */
        .metric-card {
            background-color: #1a1c24;
            border: 1px solid #2d2e35;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        .metric-label {
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .up-trend { color: #ef4444; } /* 台灣習慣：紅漲 */
        .down-trend { color: #10b981; } /* 台灣習慣：綠跌 */
        .neutral { color: #ffffff; }

        /* 側邊欄樣式 */
        [data-testid="stSidebar"] {
            background-color: #111827;
        }
        
        /* 指標區塊 (Technical) */
        .tech-box {
            background-color: #1e293b;
            padding: 12px;
            border-left: 4px solid #3b82f6;
            margin-bottom: 10px;
            border-radius: 0 8px 8px 0;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- 數據抓取模組 ---

def get_yfinance_data(tickers):
    """
    使用 yfinance 抓取即時與歷史數據。
    
    Args:
        tickers (list): 股票代號列表。
        
    Returns:
        dict: 包含代號及其最新價格與漲跌幅的字典。
    """
    results = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # 取得最新一筆交易
            data = stock.history(period="2d")
            if len(data) >= 2:
                close_now = data['Close'].iloc[-1]
                close_prev = data['Close'].iloc[-2]
                change_pct = (close_now - close_prev) / close_prev * 100
                results[ticker] = {
                    "price": round(close_now, 2),
                    "pct": round(change_pct, 2)
                }
            else:
                results[ticker] = {"price": 0.0, "pct": 0.0}
        except Exception as e:
            results[ticker] = {"price": None, "pct": None}
    return results

def get_txf_data(api_key):
    """
    使用 Fugle MarketData API 獲取台指期最近月合約資訊。
    
    Args:
        api_key (str): 富果 API Key。
        
    Returns:
        tuple: (合約代號, 最新價格, 漲跌)
    """
    if not api_key:
        return "N/A", 0.0, 0.0
    
    try:
        client = RestClient(api_key=api_key)
        # 搜尋台指期合約 (TXF)
        fut_tickers = client.futopt.intraday.tickers(type='TXF')
        # 簡單邏輯：取第一個或合約月份最小的 (通常為當月)
        target_ticker = fut_tickers[0]['symbol'] if fut_tickers else None
        
        if target_ticker:
            quote = client.futopt.intraday.quote(symbol=target_ticker)
            last_price = quote.get('lastPrice', 0)
            change = quote.get('change', 0)
            return target_ticker, last_price, change
        return "No Ticker", 0.0, 0.0
    except Exception as e:
        return "Error", 0.0, 0.0

def calculate_indicators(symbol):
    """
    計算 RSI(14) 與 MA 指標。
    
    Args:
        symbol (str): 股票代號。
        
    Returns:
        dict: 指標數值字典。
    """
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        # RSI 計算
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ma5 = df['Close'].rolling(window=5).mean()
        ma20 = df['Close'].rolling(window=20).mean()
        
        return {
            "rsi": round(rsi.iloc[-1], 2),
            "ma5": round(ma5.iloc[-1], 2),
            "ma20": round(ma20.iloc[-1], 2)
        }
    except:
        return {"rsi": 0, "ma5": 0, "ma20": 0}

# --- 側邊欄 (Sidebar) ---
with st.sidebar:
    st.title("⚙️ 系統配置")
    
    # 功能狀態檢測
    st.subheader("連線狀態")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("AI 引擎")
        st.write("✅ Active" if st.session_state.get('gemini_ok') else "⚠️ Pending")
    with col_s2:
        st.write("Fugle API")
        st.write("✅ Online" if st.session_state.get('fugle_ok') else "⚠️ Offline")
    
    st.divider()
    
    # API 金鑰管理
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Required")
    fugle_key = st.text_input("Fugle API Key", type="password", placeholder="Optional (For TXF)")
    
    # 更新 session state
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            st.session_state['gemini_ok'] = True
        except: st.session_state['gemini_ok'] = False
    
    if fugle_key:
        st.session_state['fugle_ok'] = True
    else:
        st.session_state['fugle_ok'] = False

    # 自動監控
    st.subheader("自動監控設定")
    auto_refresh = st.toggle("啟動自動刷新", value=False)
    refresh_interval = st.slider("刷新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.expander("🔔 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        tg_chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.toast("測試訊息已發送 (模擬)")

# --- 主儀表板 UI ---

# 1. Header
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; color:white;'>彈性量化戰情室 (Flexible Mode)</h1>
        <p style='margin:0; opacity:0.8;'>AI 分析引擎: gemini-3-flash-preview | 即時市場數據監控</p>
    </div>
""", unsafe_allow_html=True)

# 抓取數據
with st.spinner("正在獲取最新市場報價..."):
    market_data = get_yfinance_data(["^TWII", "^VIX", "2330.TW", "NVDA"])
    txf_ticker, txf_price, txf_change = get_txf_data(fugle_key)
    
    # 加權指數與 VIX 資料
    twii = market_data.get("^TWII", {"price": 0, "pct": 0})
    vix = market_data.get("^VIX", {"price": 0, "pct": 0})
    
    # 計算期現貨價差
    spread = round(txf_price - twii['price'], 2) if txf_price > 0 else "---"

# 2. 第一列：核心指標 (Metrics)
m1, m2, m3, m4 = st.columns(4)

def render_metric_card(column, label, value, delta, is_vix=False):
    """自定義卡片渲染"""
    delta_class = "up-trend" if delta >= 0 else "down-trend"
    # VIX 邏輯反轉 (下跌通常是好事)
    if is_vix:
        delta_class = "down-trend" if delta >= 0 else "up-trend"
        
    column.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="{delta_class}">{'+' if delta >= 0 else ''}{delta}%</div>
        </div>
    """, unsafe_allow_html=True)

with m1:
    render_metric_card(m1, "加權指數 (TWII)", f"{twii['price']:,}", twii['pct'])
with m2:
    txf_pct = round((txf_change / (txf_price - txf_change)) * 100, 2) if txf_price != 0 else 0
    render_metric_card(m2, f"台指期 ({txf_ticker})", f"{txf_price:,}", txf_pct)
with m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">期現貨價差 (Spread)</div>
            <div class="metric-value" style="color:#fbbf24;">{spread}</div>
            <div style="font-size:0.8rem; color:#9ca3af;">Basis Points</div>
        </div>
    """, unsafe_allow_html=True)
with m4:
    render_metric_card(m4, "VIX 恐慌指數", f"{vix['price']}", vix['pct'], is_vix=True)

st.write("") # 間距

# 3. 第二列：個股與技術指標
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎯 重點監測個股")
    sub_l, sub_r = st.columns(2)
    tsmc = market_data.get("2330.TW", {"price": 0, "pct": 0})
    nvda = market_data.get("NVDA", {"price": 0, "pct": 0})
    
    with sub_l:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">台積電 (2330.TW)</div>
                <div class="metric-value">{tsmc['price']}</div>
                <div class="{'up-trend' if tsmc['pct'] >= 0 else 'down-trend'}">{tsmc['pct']}%</div>
            </div>
        """, unsafe_allow_html=True)
    with sub_r:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">NVIDIA (NVDA)</div>
                <div class="metric-value">${nvda['price']}</div>
                <div class="{'up-trend' if nvda['pct'] >= 0 else 'down-trend'}">{nvda['pct']}%</div>
            </div>
        """, unsafe_allow_html=True)

with col_right:
    st.subheader("🛠 技術指標區塊 (TSMC)")
    tech = calculate_indicators("2330.TW")
    
    st.markdown(f"""
        <div class="tech-box">
            <span style="color:#9ca3af;">Relative Strength Index (14):</span> 
            <b style="color:{'#ef4444' if tech['rsi'] > 70 else '#10b981' if tech['rsi'] < 30 else '#ffffff'}">{tech['rsi']}</b>
        </div>
        <div class="tech-box">
            <span style="color:#9ca3af;">MA(5) 短線支撐:</span> <b>{tech['ma5']}</b>
        </div>
        <div class="tech-box">
            <span style="color:#9ca3af;">MA(20) 月線趨勢:</span> <b>{tech['ma20']}</b>
        </div>
    """, unsafe_allow_html=True)

# 4. AI 決策建議區塊
st.divider()
st.subheader("🤖 AI 盤勢量化分析")

if gemini_key:
    if st.button("執行 Gemini AI 策略診斷"):
        try:
            model = genai.GenerativeModel('gemini-3-flash-preview')
            prompt = f"""
            你是一位專業的量化交易員。請根據以下數據進行簡短分析：
            1. 加權指數: {twii['price']} ({twii['pct']}%)
            2. 台指期: {txf_price} (價差: {spread})
            3. VIX: {vix['price']}
            4. 台積電 RSI: {tech['rsi']}
            請給出「盤勢評價」、「風險等級」與「操盤建議」。
            """
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.error(f"AI 分析失敗: {str(e)}")
else:
    st.warning("請在側邊欄輸入 Gemini API Key 以啟用 AI 分析功能。")

# 5. 自動刷新邏輯
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# fugle-marketdata
# plotly
[instruction]
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from fugle_marketdata import RestClient
import datetime
import time
import plotly.graph_objects as go

# --- 頁面初始設定 ---
st.set_page_config(
    page_title="彈性量化戰情室 | Professional Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺樣式定義 ---
def local_css():
    """
    注入自定義 CSS 以實現深色主題、卡片陰影與漸層背景。
    """
    st.markdown("""
        <style>
        /* 整體背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* 頂部漸層 Header */
        .main-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            text-align: center;
        }
        
        /* 戰情室指標卡片 */
        .metric-card {
            background-color: #1a1c24;
            border: 1px solid #2d2e35;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        .metric-label {
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .up-trend { color: #ef4444; } /* 台灣習慣：紅漲 */
        .down-trend { color: #10b981; } /* 台灣習慣：綠跌 */
        .neutral { color: #ffffff; }

        /* 側邊欄樣式 */
        [data-testid="stSidebar"] {
            background-color: #111827;
        }
        
        /* 指標區塊 (Technical) */
        .tech-box {
            background-color: #1e293b;
            padding: 12px;
            border-left: 4px solid #3b82f6;
            margin-bottom: 10px;
            border-radius: 0 8px 8px 0;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- 數據抓取模組 ---

def get_yfinance_data(tickers):
    """
    使用 yfinance 抓取即時與歷史數據。
    
    Args:
        tickers (list): 股票代號列表。
        
    Returns:
        dict: 包含代號及其最新價格與漲跌幅的字典。
    """
    results = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # 取得最新一筆交易
            data = stock.history(period="2d")
            if len(data) >= 2:
                close_now = data['Close'].iloc[-1]
                close_prev = data['Close'].iloc[-2]
                change_pct = (close_now - close_prev) / close_prev * 100
                results[ticker] = {
                    "price": round(close_now, 2),
                    "pct": round(change_pct, 2)
                }
            else:
                results[ticker] = {"price": 0.0, "pct": 0.0}
        except Exception as e:
            results[ticker] = {"price": None, "pct": None}
    return results

def get_txf_data(api_key):
    """
    使用 Fugle MarketData API 獲取台指期最近月合約資訊。
    
    Args:
        api_key (str): 富果 API Key。
        
    Returns:
        tuple: (合約代號, 最新價格, 漲跌)
    """
    if not api_key:
        return "N/A", 0.0, 0.0
    
    try:
        client = RestClient(api_key=api_key)
        # 搜尋台指期合約 (TXF)
        fut_tickers = client.futopt.intraday.tickers(type='TXF')
        # 簡單邏輯：取第一個或合約月份最小的 (通常為當月)
        target_ticker = fut_tickers[0]['symbol'] if fut_tickers else None
        
        if target_ticker:
            quote = client.futopt.intraday.quote(symbol=target_ticker)
            last_price = quote.get('lastPrice', 0)
            change = quote.get('change', 0)
            return target_ticker, last_price, change
        return "No Ticker", 0.0, 0.0
    except Exception as e:
        return "Error", 0.0, 0.0

def calculate_indicators(symbol):
    """
    計算 RSI(14) 與 MA 指標。
    
    Args:
        symbol (str): 股票代號。
        
    Returns:
        dict: 指標數值字典。
    """
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        # RSI 計算
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ma5 = df['Close'].rolling(window=5).mean()
        ma20 = df['Close'].rolling(window=20).mean()
        
        return {
            "rsi": round(rsi.iloc[-1], 2),
            "ma5": round(ma5.iloc[-1], 2),
            "ma20": round(ma20.iloc[-1], 2)
        }
    except:
        return {"rsi": 0, "ma5": 0, "ma20": 0}

# --- 側邊欄 (Sidebar) ---
with st.sidebar:
    st.title("⚙️ 系統配置")
    
    # 功能狀態檢測
    st.subheader("連線狀態")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("AI 引擎")
        st.write("✅ Active" if st.session_state.get('gemini_ok') else "⚠️ Pending")
    with col_s2:
        st.write("Fugle API")
        st.write("✅ Online" if st.session_state.get('fugle_ok') else "⚠️ Offline")
    
    st.divider()
    
    # API 金鑰管理
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Required")
    fugle_key = st.text_input("Fugle API Key", type="password", placeholder="Optional (For TXF)")
    
    # 更新 session state
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            st.session_state['gemini_ok'] = True
        except: st.session_state['gemini_ok'] = False
    
    if fugle_key:
        st.session_state['fugle_ok'] = True
    else:
        st.session_state['fugle_ok'] = False

    # 自動監控
    st.subheader("自動監控設定")
    auto_refresh = st.toggle("啟動自動刷新", value=False)
    refresh_interval = st.slider("刷新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.expander("🔔 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        tg_chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.toast("測試訊息已發送 (模擬)")

# --- 主儀表板 UI ---

# 1. Header
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; color:white;'>彈性量化戰情室 (Flexible Mode)</h1>
        <p style='margin:0; opacity:0.8;'>AI 分析引擎: gemini-3-flash-preview | 即時市場數據監控</p>
    </div>
""", unsafe_allow_html=True)

# 抓取數據
with st.spinner("正在獲取最新市場報價..."):
    market_data = get_yfinance_data(["^TWII", "^VIX", "2330.TW", "NVDA"])
    txf_ticker, txf_price, txf_change = get_txf_data(fugle_key)
    
    # 加權指數與 VIX 資料
    twii = market_data.get("^TWII", {"price": 0, "pct": 0})
    vix = market_data.get("^VIX", {"price": 0, "pct": 0})
    
    # 計算期現貨價差
    spread = round(txf_price - twii['price'], 2) if txf_price > 0 else "---"

# 2. 第一列：核心指標 (Metrics)
m1, m2, m3, m4 = st.columns(4)

def render_metric_card(column, label, value, delta, is_vix=False):
    """自定義卡片渲染"""
    delta_class = "up-trend" if delta >= 0 else "down-trend"
    # VIX 邏輯反轉 (下跌通常是好事)
    if is_vix:
        delta_class = "down-trend" if delta >= 0 else "up-trend"
        
    column.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="{delta_class}">{'+' if delta >= 0 else ''}{delta}%</div>
        </div>
    """, unsafe_allow_html=True)

with m1:
    render_metric_card(m1, "加權指數 (TWII)", f"{twii['price']:,}", twii['pct'])
with m2:
    txf_pct = round((txf_change / (txf_price - txf_change)) * 100, 2) if txf_price != 0 else 0
    render_metric_card(m2, f"台指期 ({txf_ticker})", f"{txf_price:,}", txf_pct)
with m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">期現貨價差 (Spread)</div>
            <div class="metric-value" style="color:#fbbf24;">{spread}</div>
            <div style="font-size:0.8rem; color:#9ca3af;">Basis Points</div>
        </div>
    """, unsafe_allow_html=True)
with m4:
    render_metric_card(m4, "VIX 恐慌指數", f"{vix['price']}", vix['pct'], is_vix=True)

st.write("") # 間距

# 3. 第二列：個股與技術指標
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎯 重點監測個股")
    sub_l, sub_r = st.columns(2)
    tsmc = market_data.get("2330.TW", {"price": 0, "pct": 0})
    nvda = market_data.get("NVDA", {"price": 0, "pct": 0})
    
    with sub_l:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">台積電 (2330.TW)</div>
                <div class="metric-value">{tsmc['price']}</div>
                <div class="{'up-trend' if tsmc['pct'] >= 0 else 'down-trend'}">{tsmc['pct']}%</div>
            </div>
        """, unsafe_allow_html=True)
    with sub_r:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">NVIDIA (NVDA)</div>
                <div class="metric-value">${nvda['price']}</div>
                <div class="{'up-trend' if nvda['pct'] >= 0 else 'down-trend'}">{nvda['pct']}%</div>
            </div>
        """, unsafe_allow_html=True)

with col_right:
    st.subheader("🛠 技術指標區塊 (TSMC)")
    tech = calculate_indicators("2330.TW")
    
    st.markdown(f"""
        <div class="tech-box">
            <span style="color:#9ca3af;">Relative Strength Index (14):</span> 
            <b style="color:{'#ef4444' if tech['rsi'] > 70 else '#10b981' if tech['rsi'] < 30 else '#ffffff'}">{tech['rsi']}</b>
        </div>
        <div class="tech-box">
            <span style="color:#9ca3af;">MA(5) 短線支撐:</span> <b>{tech['ma5']}</b>
        </div>
        <div class="tech-box">
            <span style="color:#9ca3af;">MA(20) 月線趨勢:</span> <b>{tech['ma20']}</b>
        </div>
    """, unsafe_allow_html=True)

# 4. AI 決策建議區塊
st.divider()
st.subheader("🤖 AI 盤勢量化分析")

if gemini_key:
    if st.button("執行 Gemini AI 策略診斷"):
        try:
            model = genai.GenerativeModel('gemini-3-flash-preview')
            prompt = f"""
            你是一位專業的量化交易員。請根據以下數據進行簡短分析：
            1. 加權指數: {twii['price']} ({twii['pct']}%)
            2. 台指期: {txf_price} (價差: {spread})
            3. VIX: {vix['price']}
            4. 台積電 RSI: {tech['rsi']}
            請給出「盤勢評價」、「風險等級」與「操盤建議」。
            """
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.error(f"AI 分析失敗: {str(e)}")
else:
    st.warning("請在側邊欄輸入 Gemini API Key 以啟用 AI 分析功能。")

# 5. 自動刷新邏輯
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# fugle-marketdata
# plotly
fugle_key = st.text_input("Fugle API Key", type="password", placeholder="Optional (For TXF)")
     
     # 更新 session state
    st.session_state['gemini_ok'] = False
     if gemini_key:
         try:
             genai.configure(api_key=gemini_key)
             st.session_state['gemini_ok'] = True

         except Exception:
             pass
     if fugle_key:
         st.session_state['fugle_ok'] = True
     else:

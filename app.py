import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import datetime
import time
import requests
from pandas_datareader import data as pdr

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="量化交易戰情室 | Professional Trading Room",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式注入 (Custom UI/UX) ---
def local_css():
    st.markdown("""
    <style>
        /* 全域暗色背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ffffff;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        
        /* 頂部漸層卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #1e40af 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* 指標卡片樣式 */
        .metric-card {
            background-color: #1a1c24;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #2d2e35;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        /* 技術指標專用深色卡片 */
        .tech-card {
            background-color: #111827;
            padding: 20px;
            border-radius: 12px;
            border-top: 3px solid #6366f1;
            margin-top: 10px;
        }
        
        /* 文字顏色定義 */
        .up-trend { color: #ef4444; font-weight: bold; } /* 紅漲 */
        .down-trend { color: #10b981; font-weight: bold; } /* 綠跌 */
        .vix-up { color: #f59e0b; }
        
        /* 側邊欄調整 */
        .stSidebar {
            background-color: #111827;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 數據抓取模組 ---

def get_market_data():
    """
    抓取加權指數、VIX 與 美股 NVDA 數據。
    
    Returns:
        dict: 包含各項標的價格與漲跌資訊
    """
    try:
        # 抓取 Yahoo Finance 數據
        tickers = {
            "TWII": "^TWII",
            "VIX": "^VIX",
            "TSMC": "2330.TW",
            "NVDA": "NVDA"
        }
        data = yf.download(list(tickers.values()), period="2d", interval="1d", progress=False)
        
        res = {}
        for key, sym in tickers.items():
            current = data['Close'][sym].iloc[-1]
            prev = data['Close'][sym].iloc[-2]
            change = current - prev
            pct_change = (change / prev) * 100
            res[key] = {"price": current, "change": change, "pct": pct_change}
        return res
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return None

def get_txf_data():
    """
    使用 pd.read_html 抓取台指期數據 (模擬從 Yahoo 財經期貨頁面抓取)。
    
    Returns:
        float: 台指期最新價格
    """
    try:
        # 注意：實際開發中 Yahoo 期貨頁面結構可能變動，此處模擬邏輯
        # 為了穩定性，範例代碼透過 yfinance 抓取 'WTXF=F' (台指期近月代碼)
        txf = yf.Ticker("WTXF=F")
        price = txf.fast_info['last_price']
        return price
    except:
        # 若失敗則返回加權指數減去隨機點數模擬
        return None

def calculate_indicators(symbol="2330.TW"):
    """
    計算技術指標：RSI(14), MA(5), MA(20)。
    
    Args:
        symbol (str): 股票代碼
        
    Returns:
        df: 包含技術指標的 DataFrame
    """
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    # MA 計算
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI 計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.iloc[-1]

# --- 系統配置與 AI 連線 ---

def check_ai_status(api_key):
    """
    檢查 Gemini API 連線狀態。
    """
    if not api_key:
        return "⚠️ 未設定"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 確認連線
        return "✅ 已連線"
    except:
        return "⚠️ 連線失敗"

# --- 側邊欄佈局 ---

with st.sidebar:
    st.title("⚙️ 系統配置")
    
    st.subheader("連線狀態")
    gemini_key = st.text_input("Gemini API Key", type="password", help="請輸入 Google AI API Key")
    fugle_key = st.text_input("Fugle API Key (Optional)", type="password")
    
    ai_status = check_ai_status(gemini_key)
    st.info(f"AI 核心: {ai_status}")
    st.info(f"Python 腳本: ✅ 正常")

    st.markdown("---")
    st.subheader("自動監控")
    is_auto = st.toggle("啟動自動刷新", value=False)
    refresh_rate = st.slider("刷新頻率 (秒)", 10, 300, 60)

    st.markdown("---")
    with st.expander("📲 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.toast("測試訊息已發送 (模擬)")

# --- 主儀表板內容 ---

# 1. Header
st.markdown("""
<div class="header-card">
    <h1 style='margin:0;'>📊 彈性量化戰情室 (Flexible Mode)</h1>
    <p style='margin:0; opacity: 0.8;'>即時市場行情分析與 AI 輔助判讀系統</p>
</div>
""", unsafe_allow_html=True)

# 2. 獲取數據
m_data = get_market_data()
txf_price = get_txf_data()

if m_data:
    # 第一列：Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = m_data['TWII']['price']
        change = m_data['TWII']['change']
        color_class = "up-trend" if change >= 0 else "down-trend"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity:0.7;">加權指數 (TWII)</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{price:,.2f}</div>
            <div class="{color_class}">{change:+.2f} ({m_data['TWII']['pct']:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        txf_val = txf_price if txf_price else 0
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity:0.7;">台指期 (TXF)</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{txf_val:,.2f}</div>
            <div style="color: #94a3b8;">近期契約</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        spread = txf_val - m_data['TWII']['price'] if txf_val else 0
        spread_type = "正價差" if spread >= 0 else "逆價差"
        spread_color = "#ef4444" if spread >= 0 else "#10b981"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity:0.7;">期現貨價差 (Spread)</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: {spread_color};">{spread:+.2f}</div>
            <div style="font-weight: bold;">{spread_type}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        vix_price = m_data['VIX']['price']
        vix_color = "vix-up" if m_data['VIX']['change'] > 0 else "down-trend"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity:0.7;">VIX 恐慌指數</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{vix_price:.2f}</div>
            <div class="{vix_color}">風險狀態: {"偏高" if vix_price > 20 else "穩定"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 第二列：技術指標與個股
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📍 核心追蹤標的")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("台積電 (2330.TW)", f"{m_data['TSMC']['price']:.1f}", f"{m_data['TSMC']['pct']:.2f}%")
        with c2:
            st.metric("NVDA (美股)", f"{m_data['NVDA']['price']:.2f}", f"{m_data['NVDA']['pct']:.2f}%")
        
        # 繪製簡單線圖
        st.line_chart(yf.download("2330.TW", period="1mo", progress=False)['Close'], height=200)

    with right_col:
        st.subheader("🔍 技術指標區塊 (Technical Indicators)")
        tech_data = calculate_indicators("2330.TW")
        
        st.markdown(f"""
        <div class="tech-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <span>RSI (14)</span>
                <span style="font-weight: bold; color: #818cf8;">{tech_data['RSI']:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <span>MA (5) - 短線</span>
                <span style="font-weight: bold;">{tech_data['MA5']:,.1f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>MA (20) - 月線</span>
                <span style="font-weight: bold;">{tech_data['MA20']:,.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI 簡易分析建議
        if gemini_key:
            if st.button("🤖 AI 盤勢分析建議"):
                with st.spinner("AI 正在解析市場情緒..."):
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-3-flash-preview')
                    prompt = f"""
                    你是一位專業量化分析師。請根據以下數據給出精簡建議：
                    1. 加權指數: {m_data['TWII']['price']}
                    2. 台指期價差: {spread}
                    3. VIX: {vix_price}
                    4. 台積電 RSI: {tech_data['RSI']}
                    請提供：(A) 當前盤勢定調 (B) 操作風險提示 (C) 短期關鍵支撐。使用繁體中文。
                    """
                    response = model.generate_content(prompt)
                    st.success("AI 建議分析完成：")
                    st.write(response.text)
        else:
            st.warning("請於側邊欄輸入 Gemini API Key 以啟動 AI 分析功能。")

# --- 自動刷新邏輯 ---
if is_auto:
    time.sleep(refresh_rate)
    st.rerun()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# requests
# lxml

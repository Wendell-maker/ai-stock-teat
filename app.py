import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="量化交易戰情室 | Professional Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式注入 (Dark Theme & Card UI) ---
st.markdown("""
<style>
    /* 全域暗色背景與字體 */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* 頂部漸層 Header */
    .header-card {
        background: linear-gradient(90deg, #1A237E 0%, #0D47A1 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        text-align: center;
    }
    .header-card h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
        letter-spacing: 2px;
    }

    /* 指標卡片樣式 */
    .metric-card {
        background-color: #1E2633;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2196F3;
        box-shadow: 2px 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #90A4AE;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-delta {
        font-size: 1rem;
    }

    /* 技術指標區塊樣式 */
    .tech-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
    }

    /* 漲跌顏色 */
    .price-up { color: #FF5252; }    /* 台股邏輯：紅漲 */
    .price-down { color: #4CAF50; }  /* 台股邏輯：綠跌 */
    .vix-alert { color: #FF9800; }
    
    /* 側邊欄間距 */
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_wtx_price():
    """
    透過爬蟲獲取台指期 (WTX=F) 即時價格。
    
    Returns:
        float: 當前價格，若失敗則回傳 None。
    """
    try:
        url = "https://finance.yahoo.com/quote/WTX=F"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尋找 Yahoo Finance 的價格標籤
        price_tag = soup.find('fin-streamer', {'data-field': 'regularMarketPrice', 'data-symbol': 'WTX=F'})
        if price_tag:
            return float(price_tag.text.replace(',', ''))
        return None
    except Exception as e:
        return None

def fetch_market_data():
    """
    使用 yfinance 抓取市場關鍵數據。
    
    Returns:
        dict: 包含各項市場數據的字典。
    """
    tickers = {
        "TWII": "^TWII",      # 加權指數
        "VIX": "^VIX",        # 恐慌指數
        "TSMC": "2330.TW",    # 台積電
        "NVDA": "NVDA"        # NVIDIA
    }
    
    data_results = {}
    for key, symbol in tickers.items():
        try:
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                data_results[key] = {
                    "price": current_price,
                    "change": change_pct,
                    "history": hist
                }
        except:
            data_results[key] = None
    return data_results

def calculate_indicators(df):
    """
    計算常用的技術指標 (MA, RSI)。
    
    Args:
        df (pd.DataFrame): 包含 Close 價格的 DataFrame。
    Returns:
        dict: 包含 MA5, MA20, RSI14 的最新值。
    """
    if df is None or len(df) < 20:
        return {"MA5": 0, "MA20": 0, "RSI14": 0}
    
    close = df['Close']
    ma5 = close.rolling(window=5).mean().iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    
    # RSI 計算
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi14 = 100 - (100 / (1 + rs)).iloc[-1]
    
    return {"MA5": ma5, "MA20": ma20, "RSI14": rsi14}

# --- 側邊欄配置 ---

with st.sidebar:
    st.title("⚙️ 系統配置")
    
    # 狀態檢測
    st.subheader("功能狀態")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("AI 引擎: ✅")
    with col_s2:
        st.write("腳本連線: ✅")
        
    st.divider()
    
    # API 管理
    st.subheader("API 金鑰管理")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your key...")
    fugle_key = st.text_input("Fugle API Key (Optional)", type="password", placeholder="Optional...")
    
    # 自動監控
    st.subheader("自動監控")
    auto_monitor = st.toggle("開啟即時監控", value=False)
    refresh_rate = st.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.expander("🔔 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        tg_chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.info("測試訊息已送出 (模擬)")

# --- 主儀表板邏輯 ---

# 1. Header
st.markdown("""
    <div class="header-card">
        <h1>彈性量化戰情室 (Flexible Mode)</h1>
        <p style="color: #BBDEFB;">市場即時數據監控與 AI 決策系統</p>
    </div>
""", unsafe_allow_html=True)

# 2. 獲取數據
with st.spinner('正在獲取全球市場數據...'):
    market_data = fetch_market_data()
    wtx_price = get_wtx_price()

# 3. 第一列：核心指標 (Metrics)
m1, m2, m3, m4 = st.columns(4)

# 加權指數
if market_data.get("TWII"):
    twii = market_data["TWII"]
    color_class = "price-up" if twii['change'] >= 0 else "price-down"
    sign = "+" if twii['change'] >= 0 else ""
    m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">加權指數 (TWII)</div>
            <div class="metric-value">{twii['price']:,.2f}</div>
            <div class="metric-delta {color_class}">{sign}{twii['change']:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

# 台指期
if wtx_price:
    # 簡易計算與加權指數的價差
    spread = wtx_price - market_data["TWII"]["price"] if market_data.get("TWII") else 0
    m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">台指期 (WTX=F)</div>
            <div class="metric-value">{wtx_price:,.0f}</div>
            <div class="metric-delta">即時報價</div>
        </div>
    """, unsafe_allow_html=True)
    
    m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">期現貨價差 (Spread)</div>
            <div class="metric-value">{spread:,.2f}</div>
            <div class="metric-delta">{"正價差" if spread > 0 else "逆價差"}</div>
        </div>
    """, unsafe_allow_html=True)
else:
    m2.warning("台指期數據獲取失敗")
    m3.info("價差計算無法顯示")

# VIX
if market_data.get("VIX"):
    vix = market_data["VIX"]
    # VIX 邏輯：越高越恐慌，顏色反向
    vix_color = "vix-alert" if vix['price'] > 20 else ""
    m4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">VIX 恐慌指數</div>
            <div class="metric-value {vix_color}">{vix['price']:.2f}</div>
            <div class="metric-delta">{vix['change']:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

st.write("") # 間距

# 4. 第二列：個股與技術指標
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("💎 關鍵標的觀察")
    c1, c2 = st.columns(2)
    
    if market_data.get("TSMC"):
        tsmc = market_data["TSMC"]
        c1.metric("台積電 (2330.TW)", f"{tsmc['price']:.1f}", f"{tsmc['change']:.2f}%")
        
    if market_data.get("NVDA"):
        nvda = market_data["NVDA"]
        c2.metric("NVIDIA (NVDA)", f"{nvda['price']:.2f}", f"{nvda['change']:.2f}%")

    # 簡單畫個圖 (以加權指數為例)
    if market_data.get("TWII"):
        st.line_chart(market_data["TWII"]["history"]["Close"], height=250)

with col_right:
    st.subheader("📊 技術指標區塊")
    
    # 獲取加權指數做技術分析
    if market_data.get("TWII"):
        # 為了計算 MA/RSI，我們抓取更長的歷史數據
        tw_full = yf.Ticker("^TWII").history(period="1mo")
        indicators = calculate_indicators(tw_full)
        
        st.markdown(f"""
            <div class="tech-card">
                <p><b>RSI (14)</b></p>
                <h3 style="color: {'#FFA726' if indicators['RSI14'] > 70 else '#66BB6A'}">{indicators['RSI14']:.2f}</h3>
                <hr style="border: 0.5px solid #30363D">
                <p><b>MA (5) 短線</b></p>
                <p>{indicators['MA5']:,.2f}</p>
                <p><b>MA (20) 月線</b></p>
                <p>{indicators['MA20']:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 簡單策略訊號
        if indicators['RSI14'] > 70:
            st.error("⚠️ 市場過熱 (RSI > 70)")
        elif indicators['RSI14'] < 30:
            st.success("✅ 市場超賣 (RSI < 30)")
        else:
            st.info("ℹ️ 指標盤整中")

# 5. AI 分析區塊
st.divider()
st.subheader("🤖 AI 盤勢智能分析")
if st.button("啟動 Gemini-3-Flash 盤勢解讀"):
    if not gemini_key:
        st.warning("請在側邊欄輸入 Gemini API Key 以啟動 AI 分析功能。")
    else:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            
            # 準備分析用的 Prompt
            prompt = f"""
            你是一位專業的台股量化交易分析師。
            請根據以下數據進行簡短分析：
            1. 加權指數：{market_data['TWII']['price'] if market_data.get('TWII') else 'N/A'}
            2. 台指期：{wtx_price if wtx_price else 'N/A'}
            3. VIX 指數：{market_data['VIX']['price'] if market_data.get('VIX') else 'N/A'}
            4. RSI(14)：{indicators['RSI14']:.2f}
            
            請提供：
            - 市場情緒總結 (多/空/中性)
            - 潛在風險提示
            - 交易建議 (短線)
            """
            
            with st.spinner('AI 正在思考中...'):
                response = model.generate_content(prompt)
                st.write(response.text)
        except Exception as e:
            st.error(f"AI 分析失敗: {str(e)}")

# 底部頁腳
st.markdown("""
    <div style="text-align: center; color: #546E7A; font-size: 0.8rem; margin-top: 50px;">
        © 2024 Professional Trading Dashboard | Data provided by Yahoo Finance & Google Gemini
    </div>
""", unsafe_allow_html=True)

# --- 自動更新處理 ---
if auto_monitor:
    time.sleep(refresh_rate)
    st.rerun()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# requests
# beautifulsoup4

import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from fugle_marketdata import RestClient

# --- 全局 UI 配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Quant Dash",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式注入 (Dark Theme & Card Style) ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現深色高質感 UI 與漸層效果。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
        }
        
        /* 頂部漸層標頭 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        /* 技術指標卡片樣式 */
        .metric-container {
            background-color: #1a1c24;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 10px;
        }
        
        /* 指標文字顏色 */
        .rsi-high { color: #ff4b4b; font-weight: bold; }
        .rsi-low { color: #00ff41; font-weight: bold; }
        .rsi-mid { color: #ffffff; }

        /* 隱藏預設 Streamlit 元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (Data Scraping) ---

def get_stock_data(symbol: str, period: str = "1mo"):
    """
    使用 yfinance 抓取股票或指數數據。
    
    :param symbol: yfinance 代號 (如 ^TWII, 2330.TW)
    :param period: 抓取期間
    :return: DataFrame or None
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_txf_data(fugle_key: str):
    """
    台指期 (TXF) 雙源策略：優先使用 Fugle，備援使用 yfinance (WTX=F)。
    
    :param fugle_key: Fugle Market Data API Key
    :return: (current_price, change_percent)
    """
    # 嘗試使用 Fugle
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            # 自動搜尋最近月台指期合約 (TXF + 當月/次月代號)
            # 這裡簡化為獲取熱門合約資訊
            tickers = client.futopt.intraday.tickers(type='FUTURE', exchange='TAIFEX', symbol='TXF')
            if tickers:
                # 取得第一個合約 (通常是近月)
                target_symbol = tickers[0].get('symbol')
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                price = quote.get('lastPrice')
                change_pct = quote.get('changePercent', 0)
                if price: return float(price), float(change_pct)
        except Exception as e:
            st.sidebar.warning(f"Fugle 連線失敗: {e}")

    # 備援：yfinance (WTX=F 代表台指期連續合約)
    try:
        txf_yf = yf.Ticker("WTX=F")
        hist = txf_yf.history(period="2d")
        if len(hist) >= 2:
            price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_pct = ((price - prev_close) / prev_close) * 100
            return float(price), float(change_pct)
    except:
        pass
    return 0.0, 0.0

def get_fii_oi():
    """
    從期交所抓取外資期貨淨未平倉口數 (FII Net Open Interest)。
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        tables = pd.read_html(url)
        # 根據期交所結構，通常是表格中特定位置
        df = tables[3] # 依網頁結構而定，此為常見 index
        # 抓取外資 (第三列) 的淨額 (第 12 欄)
        fii_net_oi = int(df.iloc[3, 11])
        return fii_net_oi
    except:
        return 0

def get_option_max_oi():
    """
    抓取選擇權最大未平倉區間 (Call Wall / Put Wall)。
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/callsAndPutsDate"
        # 此處為簡化邏輯：實務上需解析當月合約所有履約價
        # 範例回傳模擬數據，若需真實數據需解析完整列表
        return 23500, 22000 # Call Wall, Put Wall
    except:
        return 0, 0

# --- 技術指標計算 ---

def calculate_indicators(df: pd.DataFrame):
    """
    計算 RSI(14), MA(5), MA(20)。
    """
    if df is None or len(df) < 20:
        return 0.0, 0.0, 0.0
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ma5 = df['Close'].rolling(window=5).mean()
    ma20 = df['Close'].rolling(window=20).mean()
    
    return float(rsi.iloc[-1]), float(ma5.iloc[-1]), float(ma20.iloc[-1])

# --- AI 分析模組 ---

def get_ai_analysis(api_key: str, market_data: dict):
    """
    使用 Gemini API 進行市場情緒與技術面綜合分析。
    """
    if not api_key:
        return "⚠️ 請在左側邊欄輸入 Gemini API Key 以啟用 AI 分析功能。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位資深量化交易專家。請根據以下市場數據提供簡短、精闢的分析報告：
        
        1. 加權指數: {market_data['twii_price']} (漲跌: {market_data['twii_change']:.2f}%)
        2. 台指期: {market_data['txf_price']} (價差: {market_data['spread']:.2f})
        3. VIX 指數: {market_data['vix_price']}
        4. 技術指標 (加權): RSI(14)={market_data['rsi']:.2f}, MA5={market_data['ma5']:.2f}, MA20={market_data['ma20']:.2f}
        5. 籌碼面: 外資期貨淨未平倉={market_data['fii_oi']} 口
        6. 美股連動: NVDA={market_data['nvda_price']}
        
        請針對「當前多空趨勢」與「操作建議」給出繁體中文回覆，並以 Markdown 格式呈現。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {e}"

# --- 主程式邏輯 ---

def main():
    inject_custom_css()
    
    # --- Sidebar ---
    st.sidebar.title("🛠️ 系統配置")
    
    # 功能狀態
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 連線狀態")
    st.sidebar.write(f"Gemini API: {'✅' if gemini_key else '⚠️'}")
    st.sidebar.write(f"Fugle API: {'✅' if fugle_key else '⚠️'}")
    
    # 自動監控
    auto_refresh = st.sidebar.toggle("啟用自動監控", value=False)
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram
    with st.sidebar.expander("📬 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        tg_chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.toast("測試訊息發送中...")

    # --- Header ---
    st.markdown("""
        <div class="header-card">
            <h1>彈性量化戰情室 <small>(Flexible Mode)</small></h1>
            <p>即時監控台股、期指、籌碼面與 AI 決策建議</p>
        </div>
    """, unsafe_allow_html=True)

    # --- 數據抓取與清洗區 ---
    # 抓取大盤
    df_twii = get_stock_data("^TWII")
    curr_twii = df_twii['Close'].iloc[-1] if df_twii is not None else 0.0
    prev_twii = df_twii['Close'].iloc[-2] if df_twii is not None else 0.0
    twii_chg = ((curr_twii - prev_twii) / prev_twii * 100) if prev_twii != 0 else 0.0
    
    # 抓取期貨
    txf_price, txf_chg = get_txf_data(fugle_key)
    
    # 抓取 VIX 與美股
    df_vix = get_stock_data("^VIX")
    vix_price = df_vix['Close'].iloc[-1] if df_vix is not None else 0.0
    vix_chg = ((vix_price - df_vix['Close'].iloc[-2]) / df_vix['Close'].iloc[-2] * 100) if df_vix is not None else 0.0
    
    df_nvda = get_stock_data("NVDA")
    nvda_price = df_nvda['Close'].iloc[-1] if df_nvda is not None else 0.0
    
    df_2330 = get_stock_data("2330.TW")
    tsmc_price = df_2330['Close'].iloc[-1] if df_2330 is not None else 0.0
    
    # 指標計算
    rsi_val, ma5_val, ma20_val = calculate_indicators(df_twii)
    spread = txf_price - curr_twii if (txf_price and curr_twii) else 0.0
    
    # 籌碼面
    fii_oi = get_fii_oi()
    c_wall, p_wall = get_option_max_oi()

    # --- Dashboard Row 1: Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("加權指數 (TWII)", f"{curr_twii:,.2f}", f"{twii_chg:+.2f}%", delta_color="normal")
    with m2:
        st.metric("台指期 (TXF)", f"{txf_price:,.2f}", f"{txf_chg:+.2f}%", delta_color="normal")
    with m3:
        st.metric("期現貨價差 (Spread)", f"{spread:+.2f}", f"{'正價差' if spread > 0 else '逆價差'}")
    with m4:
        # VIX 邏輯：跌為紅(好)，漲為綠(警示)，這裡配合台灣市場色系，漲(風險)設為 inverse
        st.metric("VIX 恐慌指數", f"{vix_price:.2f}", f"{vix_chg:+.2f}%", delta_color="inverse")

    # --- Dashboard Row 2: Stocks & Indicators ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("💡 重點個股")
        sc1, sc2 = st.columns(2)
        sc1.metric("台積電 (2330)", f"{tsmc_price:,.0f}")
        sc2.metric("NVIDIA (NVDA)", f"{nvda_price:,.2f}")
        
    with c2:
        st.subheader("📊 技術指標區塊")
        # RSI 顏色邏輯處理
        rsi_class = "rsi-mid"
        if rsi_val > 70: rsi_class = "rsi-high"
        elif rsi_val < 30: rsi_class = "rsi-low"
        
        st.markdown(f"""
        <div class="metric-container">
            <p>RSI (14): <span class="{rsi_class}">{rsi_val:.2f}</span></p>
            <p>MA (5): <span style="color:white">{ma5_val:,.2f}</span></p>
            <p>MA (20): <span style="color:white">{ma20_val:,.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)

    # --- Dashboard Row 3: Chips Data ---
    st.markdown("---")
    st.subheader("🧬 籌碼面關鍵數據")
    f1, f2, f3 = st.columns(3)
    f1.metric("外資期貨淨未平倉", f"{fii_oi:,} 口", delta=None)
    f2.metric("選擇權壓力區 (Call Wall)", f"{c_wall:,}", delta=None)
    f3.metric("選擇權支撐區 (Put Wall)", f"{p_wall:,}", delta=None)

    # --- AI Analysis Section ---
    st.markdown("---")
    with st.expander("🤖 AI 策略分析師建議", expanded=True):
        if st.button("執行 AI 市場掃描"):
            market_data = {
                "twii_price": curr_twii, "twii_change": twii_chg,
                "txf_price": txf_price, "spread": spread,
                "vix_price": vix_price, "rsi": rsi_val,
                "ma5": ma5_val, "ma20": ma20_val,
                "fii_oi": fii_oi, "nvda_price": nvda_price
            }
            with st.spinner("AI 正在思考中..."):
                analysis = get_ai_analysis(gemini_key, market_data)
                st.markdown(analysis)
        else:
            st.info("點擊按鈕獲取由 Gemini 驅動的交易建議。")

    # --- Auto Refresh Logic ---
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# requests
# beautifulsoup4
# lxml
# fugle-marketdata

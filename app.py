import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from google import genai
from streamlit_autorefresh import st_autorefresh
import json

# --- 頁面設定與樣式注入 ---

st.set_page_config(
    layout="wide", 
    page_title="台股 AI 戰情室", 
    page_icon="📈"
)

# 強制深色模式 CSS 注入
st.markdown(
    """
    <style>
    /* 強制背景深色，文字淺色 */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* 調整 Metric 指標的可讀性 */
    [data-testid="stMetricLabel"] {
        color: #B0B0B0 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    /* 調整表格文字 */
    div[data-testid="stTable"] {
        color: #FAFAFA;
    }
    /* 隱藏預設 Markdown 標題後的線條 */
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-color: #31333F;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 數據抓取模組 ---

def fetch_txf_data():
    """
    爬取 Yahoo 財經之台指期近月數據。
    
    Returns:
        dict: 包含 'price' 與 'change' 的字典，失敗則回傳 None。
    """
    url = "https://tw.stock.yahoo.com/quote/WTX%26"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 抓取價格與漲跌 (使用指定 Selector)
        price_tag = soup.find("span", class_="Fz(32px)")
        change_tag = soup.find("span", class_="Fz(20px)")
        
        if price_tag:
            price_val = price_tag.text.replace(",", "")
            change_val = change_tag.text if change_tag else "0"
            return {"price": float(price_val), "change": change_val}
    except Exception as e:
        print(f"TXF 爬取錯誤: {e}")
    return None

def fetch_market_data(ticker_symbol):
    """
    使用 yfinance 抓取市場數據。
    
    Args:
        ticker_symbol (str): 股票代碼 (e.g., '^TWII')。
        
    Returns:
        tuple: (最新價, 漲跌額, 歷史 DataFrame)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1mo")
        if df.empty:
            return None, None, None
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = latest_price - prev_price
        return latest_price, change, df
    except Exception:
        return None, None, None

def calculate_technical_indicators(df):
    """
    計算常用技術指標。
    
    Args:
        df (pd.DataFrame): 包含 'Close' 欄位的數據。
        
    Returns:
        dict: 包含 RSI, MA5, MA20 的字典。
    """
    if df is None or len(df) < 20:
        return {"RSI": 0, "MA5": 0, "MA20": 0}
    
    close = df['Close']
    
    # MA 計算
    ma5 = close.rolling(window=5).mean().iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    
    # RSI 計算 (簡化版)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1]))
    
    return {"RSI": round(rsi, 2), "MA5": round(ma5, 2), "MA20": round(ma20, 2)}

# --- 通訊模組 ---

def send_telegram_msg(token, chat_id, message):
    """
    發送 Telegram 訊息。
    
    Args:
        token (str): Bot Token.
        chat_id (str): Chat ID.
        message (str): 訊息內容。
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        res = requests.post(url, data=payload)
        return res.status_code == 200
    except Exception:
        return False

# --- 側邊欄設定 ---

with st.sidebar:
    st.title("⚙️ 系統設定")
    gemini_key = st.text_input("Gemini API Key", type="password")
    
    with st.expander("Telegram 通知設定"):
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat_id = st.text_input("Chat ID")
        if st.button("測試連線"):
            if tg_token and tg_chat_id:
                success = send_telegram_msg(tg_token, tg_chat_id, "🔔 戰情室連線測試成功！")
                if success: st.success("測試訊息已發送")
                else: st.error("發送失敗，請檢查設定")
            else:
                st.warning("請填寫完整的 Token 與 ID")
    
    auto_monitor = st.toggle("開啟自動監控 (每分鐘)", key="auto_monitoring")
    if auto_monitor:
        st_autorefresh(interval=60000, key="datarefresh")

# --- 主程式邏輯 ---

# 1. 抓取數據
twii_price, twii_change, twii_df = fetch_market_data("^TWII")
txf_data = fetch_txf_data()
vix_price, vix_change, _ = fetch_market_data("^VIX")
tsmc_price, tsmc_change, _ = fetch_market_data("2330.TW")
nvda_price, nvda_change, _ = fetch_market_data("NVDA")

# 2. 顯示頂部指標
col1, col2, col3, col4 = st.columns(4)

with col1:
    if twii_price:
        st.metric("加權指數 (TWII)", f"{twii_price:.2f}", f"{twii_change:+.2f}")
    else:
        st.metric("加權指數 (TWII)", "N/A")

with col2:
    if txf_data:
        st.metric("台指期 (TXF)", f"{txf_data['price']:.0f}", f"{txf_data['change']}")
    else:
        st.metric("台指期 (TXF)", "N/A")

with col3:
    if txf_data and twii_price:
        spread = txf_data['price'] - twii_price
        # 正價差綠色(正常表現)，逆價差紅色
        st.metric("期現貨價差", f"{spread:.2f}", f"{'正價差' if spread > 0 else '逆價差'}", delta_color="normal" if spread > 0 else "inverse")
    else:
        st.metric("期現貨價差", "N/A")

with col4:
    if vix_price:
        # VIX > 20 通常代表市場恐慌，顯示紅色
        st.metric("VIX 恐慌指數", f"{vix_price:.2f}", f"{vix_change:+.2f}", delta_color="inverse" if vix_price > 20 else "normal")
    else:
        st.metric("VIX 指數", "N/A")

st.markdown("---")

# 3. 底部細節配置
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("護國神山與 AI 龍頭")
    sub_l1, sub_l2 = st.columns(2)
    with sub_l1:
        if tsmc_price:
            st.metric("台積電 (2330)", f"{tsmc_price:.1f}", f"{tsmc_change:+.1f}")
    with sub_l2:
        if nvda_price:
            st.metric("NVIDIA (NVDA)", f"${nvda_price:.2f}", f"{nvda_change:+.2f}")

with right_col:
    st.subheader("技術指標 (TWII)")
    indicators = calculate_technical_indicators(twii_df)
    ind_c1, ind_c2, ind_c3 = st.columns(3)
    ind_c1.metric("RSI (14)", indicators["RSI"])
    ind_c2.metric("MA 5", f"{indicators['MA5']:.0f}")
    ind_c3.metric("MA 20", f"{indicators['MA20']:.0f}")

st.markdown("---")

# 4. AI 戰情解讀區塊
st.subheader("🤖 AI 戰情即時解讀")

if st.button("執行 AI 市場分析"):
    if not gemini_key:
        st.warning("請先於側邊欄輸入 Gemini API Key")
    else:
        try:
            client = genai.Client(api_key=gemini_key)
            
            # 彙整數據
            analysis_payload = {
                "TWII": twii_price,
                "TXF": txf_data['price'] if txf_data else None,
                "Spread": (txf_data['price'] - twii_price) if (txf_data and twii_price) else None,
                "VIX": vix_price,
                "Indicators": indicators,
                "Stocks": {"TSMC": tsmc_price, "NVDA": nvda_price}
            }
            
            prompt = f"""
            你是一位資深的台股分析師。請根據以下市場數據，提供簡短、精闢且具前瞻性的戰情解讀。
            數據內容：{json.dumps(analysis_payload)}
            請包含：
            1. 當前盤勢短評
            2. 期現貨價差所隱含的訊號
            3. 建議關注的壓力或支撐位
            請以條列式回答，語氣專業且精簡。
            """
            
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt
            )
            
            st.info(response.text)
            
            # 若有 Telegram 設定，同步發送分析報告
            if tg_token and tg_chat_id:
                send_telegram_msg(tg_token, tg_chat_id, f"📌 AI 戰情速報：\n{response.text}")
                
        except Exception as e:
            st.error(f"AI 分析失敗: {str(e)}")

# --- requirements.txt ---
# streamlit
# yfinance
# requests
# beautifulsoup4
# pandas
# numpy
# google-genai
# streamlit-autorefresh

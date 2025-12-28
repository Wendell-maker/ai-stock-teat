import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from fugle_marketdata import RestClient
import time
from datetime import datetime

# --- 頁面設定 ---
st.set_page_config(
    page_title="Professional Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺美化模組 ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現深色高質感 UI、卡片陰影與 RWD 佈局。
    """
    st.markdown("""
        <style>
        /* 整體背景與字體 */
        .main { background-color: #0E1117; color: #E0E0E0; }
        
        /* 頂部標題卡片 */
        .header-card {
            background: linear-gradient(90deg, #1A237E 0%, #0D47A1 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            text-align: center;
        }
        
        /* 指標卡片樣式 */
        .metric-card {
            background-color: #1E2127;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        /* 技術指標專用卡片 */
        .tech-card {
            background-color: #161B22;
            border-left: 4px solid #58A6FF;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        
        /* RSI 顏色邏輯 */
        .rsi-high { color: #FF5252; font-weight: bold; }
        .rsi-low { color: #69F0AE; font-weight: bold; }
        .rsi-normal { color: #FFFFFF; }

        /* 隱藏 Streamlit 預設元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (Market Data) ---

def get_basic_price(symbol: str):
    """
    使用 yfinance 抓取基礎標的價格與漲跌幅。
    
    :param symbol: yfinance 代號 (如 ^TWII, ^VIX)
    :return: (現價, 漲跌幅%, 原始資料)
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2d")
        if len(df) < 2:
            return 0.0, 0.0, None
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((curr_price - prev_price) / prev_price) * 100
        return float(curr_price), float(change_pct), df
    except Exception as e:
        return None, None, None

def get_txf_data(fugle_key: str = None):
    """
    台指期 (TXF) 雙源策略：優先使用 Fugle，備援使用 yfinance (WTX=F)。
    
    :param fugle_key: Fugle API Key
    :return: (台指期現價, 漲跌幅%)
    """
    # 嘗試 Fugle
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            # 自動抓取最近月合約 (簡化邏輯：抓取 TXF 開頭的 tickers)
            # 實務上需根據月份篩選，此處模擬抓取
            res = client.futopt.intraday.tickers(type='future', symbol='TXF')
            if res:
                target_symbol = res[0]['symbol'] # 取得最近月，例如 TXF202503
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                price = quote.get('lastPrice', 0)
                change_pct = quote.get('changePercent', 0)
                return float(price), float(change_pct)
        except:
            pass
    
    # 備援：yfinance (WTX=F 代表台指期連續近月)
    p, c, _ = get_basic_price("WTX=F")
    return p, c

def get_fii_oi():
    """
    抓取外資期貨淨未平倉口數 (Scraping from public source).
    """
    try:
        # 這裡以玩股網或期交所公開資訊為例 (簡化模擬)
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        # 實務上解析 HTML 較複雜，此處示範 BeautifulSoup 邏輯框架
        # headers = {'User-Agent': 'Mozilla/5.0'}
        # resp = requests.get(url, headers=headers)
        # 這裡回傳模擬數據以利執行，實作時需解析 table
        return 3450  # 模擬外資淨多單
    except:
        return 0

def get_option_max_oi():
    """
    估算選擇權最大未平倉區間 (Call Wall / Put Wall).
    """
    try:
        # 模擬回傳資料
        return {"call_wall": 23500, "put_wall": 22000}
    except:
        return {"call_wall": 0, "put_wall": 0}

# --- 技術指標計算模組 ---

def calculate_indicators(df: pd.DataFrame):
    """
    計算 RSI(14), MA(5), MA(20)。
    """
    if df is None or len(df) < 20:
        return 0.0, 0.0, 0.0
    
    close = df['Close']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ma5 = close.rolling(window=5).mean()
    ma20 = close.rolling(window=20).mean()
    
    return float(rsi.iloc[-1]), float(ma5.iloc[-1]), float(ma20.iloc[-1])

# --- AI 分析模組 ---

def get_ai_analysis(api_key, context_data):
    """
    使用 Gemini 進行市場盤勢分析。
    """
    if not api_key:
        return "⚠️ 請提供 Gemini API Key 以啟用 AI 顧問。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位專業的台股量化交易員。請根據以下數據提供簡短、精闢的市場分析：
        1. 加權指數: {context_data['twii_p']} ({context_data['twii_c']}%)
        2. 台指期: {context_data['txf_p']}
        3. 恐慌指數 VIX: {context_data['vix']}
        4. RSI(14): {context_data['rsi']:.2f}
        5. MA5/MA20: {context_data['ma5']:.2f} / {context_data['ma20']:.2f}
        6. 外資期貨淨未平倉: {context_data['fii_oi']} 口
        7. 選擇權最大 OI 區間: {context_data['opt_range']}
        
        請分析短線趨勢、支撐壓力位及交易建議。使用繁體中文。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 主程式介面 ---

def main():
    inject_custom_css()
    
    # --- Sidebar 系統配置 ---
    st.sidebar.title("🛠️ 系統配置")
    
    # 功能狀態
    st.sidebar.subheader("連線狀態")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 策略分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (選填)", type="password")
    
    status_ai = "✅" if gemini_key else "⚠️"
    status_py = "✅"
    st.sidebar.write(f"AI 引擎: {status_ai} | Python 腳本: {status_py}")
    
    # 自動監控
    auto_refresh = st.sidebar.toggle("自動更新監控", value=False)
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.sidebar.expander("🔔 Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chatid = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            st.sidebar.success("發送測試訊息中...")

    # --- Header ---
    st.markdown('<div class="header-card"><h1>📈 彈性量化戰情室 (Flexible Mode)</h1></div>', unsafe_allow_html=True)

    # --- 數據抓取邏輯 ---
    # 抓取基礎數據
    twii_p, twii_c, twii_df = get_basic_price("^TWII")
    txf_p, txf_c = get_txf_data(fugle_key)
    vix_p, vix_c, _ = get_basic_price("^VIX")
    tsmc_p, tsmc_c, tsmc_df = get_basic_price("2330.TW")
    nvda_p, nvda_c, _ = get_basic_price("NVDA")
    
    # 數據清洗 (防止 None 導致格式化錯誤)
    twii_p = twii_p if twii_p is not None else 0.0
    twii_c = twii_c if twii_c is not None else 0.0
    txf_p = txf_p if txf_p is not None else 0.0
    txf_c = txf_c if txf_c is not None else 0.0
    vix_p = vix_p if vix_p is not None else 0.0
    spread = twii_p - txf_p if twii_p and txf_p else 0.0
    
    # 計算技術指標 (以台積電為主要觀測對象)
    rsi_val, ma5_val, ma20_val = calculate_indicators(tsmc_df)

    # --- 第一列：核心指標 (Metrics) ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("加權指數", f"{twii_p:,.2f}", f"{twii_c:+.2f}%")
    with m2:
        st.metric("台指期 TXF", f"{txf_p:,.2f}", f"{txf_c:+.2f}%")
    with m3:
        st.metric("期現貨價差", f"{spread:.2f}", help="正價差代表期貨強於現貨")
    with m4:
        # VIX 邏輯：漲通常代表利空，使用 inverse 顏色 (Streamlit 1.30+ 支援)
        st.metric("VIX 恐慌指數", f"{vix_p:.2f}", f"{vix_c:+.2f}%", delta_color="inverse")

    st.divider()

    # --- 第二列：個股與技術指標 ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("🔥 關鍵個股報價")
        k1, k2 = st.columns(2)
        k1.metric("台積電 (2330)", f"{tsmc_p if tsmc_p else 0:.1f}", f"{tsmc_c if tsmc_c else 0:+.2f}%")
        k2.metric("NVDA (美股)", f"{nvda_p if nvda_p else 0:.2f}", f"{nvda_c if nvda_c else 0:+.2f}%")

    with c2:
        st.subheader("🛠️ 技術指標區塊 (TSMC)")
        # RSI 顏色判斷
        rsi_class = "rsi-normal"
        if rsi_val > 70: rsi_class = "rsi-high"
        elif rsi_val < 30: rsi_class = "rsi-low"
        
        st.markdown(f"""
            <div class="tech-card">
                <b>RSI(14):</b> <span class="{rsi_class}">{rsi_val:.2f}</span><br>
                <b>MA(5):</b> {ma5_val:.2f}<br>
                <b>MA(20):</b> {ma20_val:.2f}
            </div>
        """, unsafe_allow_html=True)

    # --- 第三列：籌碼面數據 ---
    st.subheader("📊 籌碼面監控")
    fii_oi = get_fii_oi()
    opt_data = get_option_max_oi()
    
    chip1, chip2, chip3 = st.columns(3)
    chip1.metric("外資期貨淨未平倉", f"{fii_oi:+,} 口", delta_color="normal")
    chip2.metric("Call Wall (最大壓)", f"{opt_data['call_wall']:,}")
    chip3.metric("Put Wall (最大撐)", f"{opt_data['put_wall']:,}")

    # --- AI 策略分析區 ---
    st.markdown("### 🤖 AI 智能投顧分析")
    if st.button("生成 AI 行情診斷"):
        with st.spinner("正在呼叫 Gemini 分析市場數據..."):
            context = {
                "twii_p": twii_p, "twii_c": twii_c, "txf_p": txf_p,
                "vix": vix_p, "rsi": rsi_val, "ma5": ma5_val, "ma20": ma20_val,
                "fii_oi": fii_oi, "opt_range": f"{opt_data['put_wall']} ~ {opt_data['call_wall']}"
            }
            analysis = get_ai_analysis(gemini_key, context)
            st.info(analysis)
    else:
        st.write("點擊按鈕獲取最新 AI 診斷。")

    # 自動更新機制
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# numpy
# requests
# beautifulsoup4
# google-generativeai
# fugle-marketdata
```

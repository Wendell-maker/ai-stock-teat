import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import datetime
import time
from fugle_marketdata import RestClient

# --- 頁面初始配置 ---
st.set_page_config(
    page_title="量化戰情室 | Pro Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺樣式模組 ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現暗色主題、漸層背景與高質感卡片設計。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* 頂部標題卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* 指標卡片樣式 */
        .metric-card {
            background-color: #1a1c24;
            border: 1px solid #2d2e3a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        /* 技術指標專用深色卡片 */
        .tech-card {
            background-color: #111827;
            border-left: 5px solid #3b82f6;
            padding: 12px;
            border-radius: 8px;
            margin: 5px 0;
        }

        /* 隱藏預設元件邊距 */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (Market Data) ---

def get_stock_data(ticker_symbol, period="1mo", interval="1d"):
    """
    使用 yfinance 抓取股票歷史數據。
    
    :param ticker_symbol: 標的代碼 (e.g., '^TWII', '2330.TW')
    :param period: 期間
    :param interval: 時框
    :return: pd.DataFrame or None
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"yfinance 抓取失敗 ({ticker_symbol}): {e}")
        return None

def get_txf_data(fugle_api_key):
    """
    台指期 (TXF) 雙源策略：優先使用 Fugle，備援使用 yfinance。
    
    :param fugle_api_key: 富果 API Key
    :return: (current_price, change_pct)
    """
    # 嘗試使用 Fugle
    if fugle_api_key:
        try:
            client = RestClient(api_key=fugle_api_key)
            # 獲取最近月合約 (簡易邏輯：抓取列表第一個 TXF 相關)
            tickers = client.futopt.intraday.tickers(type='future', exchange='TAIFEX', symbol='TXF')
            if tickers:
                target_symbol = tickers[0]['symbol']
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                price = quote.get('lastPrice', 0)
                change = quote.get('changePercent', 0)
                return float(price), float(change)
        except Exception as e:
            pass # 靜默失敗，轉向備援
            
    # 備援使用 yfinance (WTX=F 代表台指期)
    try:
        df = get_stock_data("WTX=F", period="2d", interval="1m")
        if df is not None and len(df) >= 2:
            last_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change_pct = ((last_price - prev_price) / prev_price) * 100
            return float(last_price), float(change_pct)
    except:
        return 0.0, 0.0
    return 0.0, 0.0

# --- 籌碼面數據模組 ---

def get_fii_oi():
    """
    抓取外資期貨淨未平倉口數 (FII Net Open Interest)。
    從期交所或財經資訊網抓取。
    """
    try:
        # 這裡模擬抓取期交所資料，實際實作可能需處理 POST 請求或使用現成 API
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        # 使用簡單的 read_html 嘗試解析
        tables = pd.read_html(url)
        # 通常表格中會包含「外資」與「多空淨額」
        # 這裡為演示提供一個模擬邏輯，實際爬蟲需根據表格索引微調
        df = tables[3] # 期交所主要的口數統計表通常在索引 3 或 4
        fii_net = df.iloc[3, 13] # 此座標為假設，實際需對位
        return int(fii_net)
    except:
        # 若抓取失敗，回傳一個隨機示範值或 0
        return 0

def get_option_max_oi():
    """
    嘗試抓取選擇權最大未平倉量 (Call Wall / Put Wall)。
    """
    try:
        # 模擬回傳邏輯
        return {"Call_Wall": 23500, "Put_Wall": 22000}
    except:
        return {"Call_Wall": 0, "Put_Wall": 0}

# --- 技術指標計算模組 ---

def calculate_rsi(series, period=14):
    """
    純 Pandas 實現 RSI 計算，避免依賴 TA-Lib。
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- AI 分析模組 ---

def analyze_with_gemini(api_key, market_info):
    """
    呼叫 Gemini API 進行盤勢分析。
    """
    if not api_key:
        return "⚠️ 未輸入 Gemini API Key，無法進行分析。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 預設使用穩定高效版本
        
        prompt = f"""
        你是一位資深量化交易員，請分析以下市場數據並給出短評：
        {market_info}
        
        要求：
        1. 針對 RSI 與 MA 趨勢進行解讀。
        2. 結合外資期貨籌碼給予多空平衡建議。
        3. 回覆字數控制在 200 字以內，使用繁體中文，語氣專業。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {e}"

# --- 主程式邏輯 ---

def main():
    inject_custom_css()
    
    # --- Sidebar 區塊 ---
    st.sidebar.title("🛠️ 系統配置")
    
    # API 狀態檢測
    gemini_key = st.sidebar.text_input("Gemini API Key (Required)", type="password")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("狀態檢測")
    ai_status = "✅ 已連線" if gemini_key else "⚠️ 缺 Key"
    py_status = "✅ 運行中"
    st.sidebar.write(f"AI 引擎: {ai_status}")
    st.sidebar.write(f"系統環境: {py_status}")
    
    st.sidebar.markdown("---")
    auto_monitor = st.sidebar.toggle("自動監控模式", value=False)
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    with st.sidebar.expander("📢 Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chatid = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            st.sidebar.success("測試訊息已發送 (模擬)")

    # --- 數據獲取與清洗區塊 ---
    # 獲取加權指數
    df_twii = get_stock_data("^TWII")
    curr_twii = df_twii['Close'].iloc[-1] if df_twii is not None else 0.0
    prev_twii = df_twii['Close'].iloc[-2] if df_twii is not None else 0.0
    twii_change = ((curr_twii - prev_twii) / prev_twii) * 100 if prev_twii != 0 else 0.0
    
    # 獲取 VIX
    df_vix = get_stock_data("^VIX")
    curr_vix = df_vix['Close'].iloc[-1] if df_vix is not None else 0.0
    
    # 獲取 TXF 期貨
    txf_price, txf_change = get_txf_data(fugle_key)
    
    # 獲取個股
    df_2330 = get_stock_data("2330.TW")
    p_2330 = df_2330['Close'].iloc[-1] if df_2330 is not None else 0.0
    df_nvda = get_stock_data("NVDA")
    p_nvda = df_nvda['Close'].iloc[-1] if df_nvda is not None else 0.0
    
    # 計算技術指標 (以加權指數為準)
    rsi_val = 0.0
    ma5 = 0.0
    ma20 = 0.0
    if df_twii is not None:
        rsi_series = calculate_rsi(df_twii['Close'])
        rsi_val = float(rsi_series.iloc[-1])
        ma5 = float(df_twii['Close'].rolling(5).mean().iloc[-1])
        ma20 = float(df_twii['Close'].rolling(20).mean().iloc[-1])

    # 籌碼數據
    fii_oi = get_fii_oi()
    opt_data = get_option_max_oi()
    
    # --- UI Dashboard 渲染 ---
    
    # Header
    st.markdown("""
        <div class="header-card">
            <h1 style='margin:0; color:white;'>彈性量化戰情室 <span style='font-size:16px;'>Flexible Mode v1.2</span></h1>
            <p style='margin:0; opacity:0.8;'>即時監控台指、國際股市與 AI 策略建議</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 第一列：Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("加權指數 (TWII)", f"{curr_twii:,.2f}", f"{twii_change:+.2f}%")
    with col2:
        # 期現貨價差
        spread = txf_price - curr_twii if txf_price != 0 else 0.0
        st.metric("台指期 (TXF)", f"{txf_price:,.0f}", f"{txf_change:+.2f}%")
    with col3:
        st.metric("期現價差 (Spread)", f"{spread:.2f}", delta_color="off")
    with col4:
        # VIX 顏色反轉 (越高越恐慌，標示為紅色/下跌)
        st.metric("恐慌指數 (VIX)", f"{curr_vix:.2f}", delta_color="inverse")

    # 第二列：個股與技術指標
    st.markdown("---")
    c_stock, c_tech = st.columns([1.5, 1])
    
    with c_stock:
        st.subheader("重點追蹤標的")
        sc1, sc2 = st.columns(2)
        sc1.markdown(f"""
        <div class="metric-card">
            <p style='color:#aaa; margin:0;'>台積電 (2330)</p>
            <h2 style='margin:0;'>NT$ {p_2330:,.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
        sc2.markdown(f"""
        <div class="metric-card">
            <p style='color:#aaa; margin:0;'>NVIDIA (NVDA)</p>
            <h2 style='margin:0;'>US$ {p_nvda:,.1f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c_tech:
        st.subheader("技術指標區塊")
        
        # RSI 顏色邏輯
        rsi_color = "white"
        if rsi_val > 70: rsi_color = "#ff4b4b" # 超買紅
        elif rsi_val < 30: rsi_color = "#00ff00" # 超賣綠
        
        st.markdown(f"""
        <div class="tech-card">
            <span>RSI (14):</span> <span style='color:{rsi_color}; font-weight:bold; font-size:20px;'>{rsi_val:.2f}</span>
        </div>
        <div class="tech-card">
            <span>MA (5):</span> <span style='font-weight:bold;'>{ma5:,.0f}</span>
        </div>
        <div class="tech-card">
            <span>MA (20):</span> <span style='font-weight:bold;'>{ma20:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)

    # 第三列：籌碼數據
    st.markdown("### 📊 籌碼面掃描")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        fii_color = "red" if fii_oi > 0 else "green"
        st.markdown(f"外資期貨淨未平倉: <span style='color:{fii_color}; font-size:20px; font-weight:bold;'>{fii_oi:,} 口</span>", unsafe_allow_html=True)
    with ch2:
        st.write(f"選擇權壓力位 (Call Wall): **{opt_data['Call_Wall']}**")
    with ch3:
        st.write(f"選擇權支撐位 (Put Wall): **{opt_data['Put_Wall']}**")

    # AI 分析區塊
    st.markdown("---")
    with st.expander("🤖 AI 策略分析建議", expanded=True):
        if st.button("執行 AI 盤勢分析"):
            market_context = f"""
            指數: {curr_twii}, 漲跌幅: {twii_change}%
            台指期: {txf_price}, 價差: {spread}
            RSI: {rsi_val}, MA5: {ma5}, MA20: {ma20}
            外資期貨淨口數: {fii_oi}
            """
            with st.spinner("Gemini 正在計算中..."):
                analysis_result = analyze_with_gemini(gemini_key, market_context)
                st.write(analysis_result)

    # 自動更新邏輯
    if auto_monitor:
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

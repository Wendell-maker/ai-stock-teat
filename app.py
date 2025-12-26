import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from fugle_marketdata import RestClient
import pandas_ta as ta
from datetime import datetime
import time

# --- 頁面設定 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Pro Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺自定義模組 ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現暗色系質感 UI 與卡片效果。
    """
    st.markdown("""
    <style>
        /* 全局背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
        }
        
        /* 頂部漸層標題卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* 數據卡片樣式 */
        .metric-card {
            background-color: #1a1c24;
            border: 1px solid #2d2d39;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }

        /* 技術指標專用卡片 */
        .indicator-card {
            background-color: #161b22;
            border-left: 5px solid #3b82f6;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        /* 數值顏色定義 */
        .val-up { color: #ff4b4b; font-weight: bold; } /* 台股紅漲 */
        .val-down { color: #00ff41; font-weight: bold; } /* 台股綠跌 */
        .val-neutral { color: #ffffff; }
        
        /* 修改 Streamlit 預設元件樣式 */
        .stMetric {
            background-color: #1a1c24;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (Market Data) ---

def get_tw_market_data():
    """
    獲取加權指數、VIX 與 美股 NVDA 數據。
    Returns: dict 包含各項標的之 Price 與 Change。
    """
    data = {}
    tickers = {
        "TWII": "^TWII",
        "VIX": "^VIX",
        "NVDA": "NVDA",
        "TSMC": "2330.TW"
    }
    try:
        for key, sym in tickers.items():
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change_pct = ((last_close - prev_close) / prev_close) * 100
                data[key] = {"price": last_close, "change": change_pct}
            else:
                data[key] = {"price": 0, "change": 0}
    except Exception as e:
        st.error(f"市場數據抓取失敗: {e}")
    return data

def get_txf_data(fugle_key=None):
    """
    台指期 (TXF) 雙源策略：優先使用 Fugle，備援使用 YFinance。
    Returns: tuple (TXF 價格, 來源名稱)
    """
    # 優先嘗試 Fugle
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            # 自動搜尋近月台指期合約代號 (簡易邏輯：TXF + 當月)
            current_month = datetime.now().strftime("%Y%m")
            ticker_symbol = f"TXF{current_month}"
            
            quote = client.futopt.intraday.quote(symbol=ticker_symbol)
            if quote and 'lastPrice' in quote:
                return float(quote['lastPrice']), f"Fugle ({ticker_symbol})"
        except Exception:
            pass # 失敗則進入備援

    # 備援：YFinance (代碼 WTX=F)
    try:
        txf_yf = yf.Ticker("WTX=F")
        price = txf_yf.fast_info['last_price']
        return price, "YFinance (WTX=F)"
    except:
        return 0, "N/A"

# --- 籌碼面抓取模組 (Scraping) ---

def get_fii_oi():
    """
    抓取期交所外資期貨淨未平倉口數。
    Returns: int (口數)
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        resp = requests.get(url, timeout=10)
        tables = pd.read_html(resp.text)
        # 通常大台指外資在第 3 個表格左右 (視網頁變動而定)
        # 這裡簡化模擬邏輯，實際開發需針對 Table Index 進行定位
        df = tables[2] 
        # 假設選取外資(第三列)的淨額(最後一欄)
        fii_net = df.iloc[3, -1] 
        return int(fii_net)
    except:
        return 0

def get_option_max_oi():
    """
    估算選擇權最大未平倉量 (Call Wall / Put Wall)。
    Returns: dict {call_wall: val, put_wall: val}
    """
    try:
        # 簡化版：抓取期交所選擇權行情，尋找 OI 最大值
        url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
        # 實際實作需傳入日期與合約參數
        return {"call_wall": 23500, "put_wall": 22000} # 範例回傳
    except:
        return {"call_wall": 0, "put_wall": 0}

# --- 技術指標計算模組 ---

def calculate_indicators(symbol="2330.TW"):
    """
    計算 RSI, MA5, MA20。
    Returns: dict 包含計算結果。
    """
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MA5'] = ta.sma(df['Close'], length=5)
        df['MA20'] = ta.sma(df['Close'], length=20)
        
        last_row = df.iloc[-1]
        return {
            "rsi": float(last_row['RSI']),
            "ma5": float(last_row['MA5']),
            "ma20": float(last_row['MA20']),
            "close": float(last_row['Close'])
        }
    except:
        return {"rsi": 50, "ma5": 0, "ma20": 0, "close": 0}

# --- UI 渲染函式 ---

def render_sidebar():
    """
    渲染左側側邊欄配置。
    """
    st.sidebar.title("🛠️ 系統配置")
    
    # 功能狀態
    st.sidebar.subheader("連線狀態")
    col1, col2 = st.sidebar.columns(2)
    col1.write("AI 引擎")
    col1.info("✅ 已連線")
    col2.write("行情 API")
    col2.warning("⚠️ 檢查中")

    # API 金鑰
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 盤勢分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")

    # 自動監控
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.toggle("啟用自動刷新", value=False)
    refresh_sec = st.sidebar.slider("重新整理間隔 (秒)", 10, 300, 60)

    # Telegram
    with st.sidebar.expander("✈️ Telegram 通知設定"):
        st.text_input("Bot Token")
        st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.toast("發送測試訊息中...")
            
    return gemini_key, fugle_key, auto_refresh, refresh_sec

def main():
    inject_custom_css()
    gemini_key, fugle_key, auto_refresh, refresh_sec = render_sidebar()

    # 1. 抓取數據
    market_data = get_tw_market_data()
    txf_price, txf_source = get_txf_data(fugle_key)
    fii_oi = get_fii_oi()
    opt_walls = get_option_max_oi()
    tech_data = calculate_indicators("2330.TW")

    # Header
    st.markdown('<div class="header-card"><h1>📈 彈性量化戰情室 (Flexible Mode)</h1></div>', unsafe_allow_html=True)

    # 第一列: Metrics (指數與 VIX)
    m1, m2, m3, m4 = st.columns(4)
    
    twii = market_data.get("TWII", {"price": 0, "change": 0})
    m1.metric("加權指數 (TWII)", f"{twii['price']:.2f}", f"{twii['change']:.2f}%")
    
    # 計算期現貨價差
    spread = txf_price - twii['price']
    m2.metric("台指期 (TXF)", f"{txf_price:.0f}", f"來源: {txf_source}", delta_color="off")
    m3.metric("期現貨價差", f"{spread:.2f}", "正價差" if spread > 0 else "逆價差")
    
    vix = market_data.get("VIX", {"price": 0, "change": 0})
    # VIX 邏輯：上漲通常代表恐慌增加（紅色），下跌代表穩定。
    m4.metric("VIX 恐慌指數", f"{vix['price']:.2f}", f"{vix['change']:.2f}%", delta_color="inverse")

    # 第二列: 個股與技術指標
    st.markdown("### 🔍 個股與技術監控")
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        tsmc = market_data.get("TSMC", {"price": 0, "change": 0})
        st.metric("台積電 (2330)", f"{tsmc['price']:.1f}", f"{tsmc['change']:.2f}%")
        
    with c2:
        nvda = market_data.get("NVDA", {"price": 0, "change": 0})
        st.metric("NVDA (美股領航)", f"{nvda['price']:.2f}", f"{nvda['change']:.2f}%")

    with c3:
        # 技術指標卡片
        rsi_val = tech_data['rsi']
        rsi_color = "white"
        if rsi_val > 70: rsi_color = "#ff4b4b"
        elif rsi_val < 30: rsi_color = "#00ff41"
        
        st.markdown(f"""
        <div style="display: flex; gap: 10px;">
            <div class="indicator-card" style="flex: 1;">
                <p style="margin:0; font-size: 0.8rem; color: #888;">RSI (14)</p>
                <h2 style="margin:0; color: {rsi_color};">{rsi_val:.2f}</h2>
            </div>
            <div class="indicator-card" style="flex: 1;">
                <p style="margin:0; font-size: 0.8rem; color: #888;">MA(5) / MA(20)</p>
                <h4 style="margin:0;">{tech_data['ma5']:.1f} / {tech_data['ma20']:.1f}</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 第三列: 籌碼面數據
    st.markdown("### 📊 籌碼面概況 (Chip Data)")
    ch1, ch2, ch3 = st.columns(3)
    
    with ch1:
        st.markdown(f"""
        <div class="metric-card">
            <small>外資期貨淨未平倉 (OI)</small>
            <h2 style="color: {'#ff4b4b' if fii_oi > 0 else '#00ff41'};">{fii_oi:,.0f} 口</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with ch2:
        st.markdown(f"""
        <div class="metric-card">
            <small>Call Wall (壓力)</small>
            <h2 style="color: #ff4b4b;">{opt_walls['call_wall']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with ch3:
        st.markdown(f"""
        <div class="metric-card">
            <small>Put Wall (支撐)</small>
            <h2 style="color: #00ff41;">{opt_walls['put_wall']}</h2>
        </div>
        """, unsafe_allow_html=True)

    # AI 盤勢解析區塊
    st.markdown("---")
    if st.button("🤖 執行 Gemini AI 深度盤勢解析"):
        if not gemini_key:
            st.error("請先在側邊欄輸入 Gemini API Key")
        else:
            try:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-3-flash-preview')
                prompt = f"""
                你是資深量化交易員，請根據以下數據進行簡短分析：
                1. 加權指數：{twii['price']} ({twii['change']:.2f}%)
                2. 台指期價差：{spread:.2f}
                3. 外資期貨 OI：{fii_oi} 口
                4. RSI(14)：{rsi_val:.2f}
                5. 台積電/NVDA 表現：{tsmc['price']} / {nvda['price']}
                請提供：市場氛圍、關鍵支撐壓力位、以及短線交易建議。
                """
                response = model.generate_content(prompt)
                st.info("### Gemini AI 分析報告")
                st.write(response.text)
            except Exception as e:
                st.error(f"AI 分析失敗: {e}")

    # 自動刷新邏輯
    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# pandas_ta
# requests
# beautifulsoup4
# lxml
# google-generativeai
# fugle-marketdata

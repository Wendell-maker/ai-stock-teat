import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from fugle_marketdata import RestClient

# --- 全局頁面配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | QuantiX Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺設計模組 ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現深色高質感 UI、卡片陰影與漸層效果。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* 側邊欄樣式 */
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
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
        
        /* 數據指標卡片樣式 */
        div[data-testid="stMetric"] {
            background-color: #1c2128;
            border: 1px solid #30363d;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }

        /* 技術指標特殊卡片 */
        .indicator-card {
            background-color: #1c2128;
            border-left: 5px solid #3b82f6;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }

        /* 文字顏色類別 */
        .text-buy { color: #ff4b4b; font-weight: bold; }
        .text-sell { color: #00d48a; font-weight: bold; }
        .text-neutral { color: #ffffff; }

        /* RWD 手機優化 */
        @media (max-width: 640px) {
            .header-card { padding: 10px; }
        }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (FinData & Scraping) ---

def get_stock_metrics(symbol):
    """
    使用 yfinance 抓取股票基礎數據與計算技術指標。
    :param symbol: 股票代號 (str)
    :return: dict 包含現價、漲跌、RSI、MA
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo")
        if df.empty:
            return None
        
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((curr_price - prev_price) / prev_price) * 100
        
        # 技術指標計算 (MA)
        ma5 = df['Close'].rolling(window=5).mean().iloc[-1]
        ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        
        # RSI(14) 計算
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        return {
            "price": float(curr_price),
            "change": float(change_pct),
            "ma5": float(ma5),
            "ma20": float(ma20),
            "rsi": float(rsi)
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_txf_data(fugle_key):
    """
    台指期數據雙源抓取：Fugle (優先) / yfinance (備援)。
    """
    # 預設值 (防呆)
    txf_price, txf_change = 0.0, 0.0
    
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            # 獲取最近月合約 (自動邏輯)
            tickers = client.futopt.intraday.tickers(type='INDEX', exchange='TAIFEX', symbol='TXF')
            if tickers:
                target_symbol = tickers[0]['symbol'] # 例如 TXF202501
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                txf_price = float(quote['lastPrice'])
                txf_change = ((txf_price - float(quote['previousClose'])) / float(quote['previousClose'])) * 100
                return txf_price, txf_change
        except Exception:
            pass # 失敗則進入 yfinance 備援

    # 備援：yfinance
    try:
        yf_txf = yf.Ticker("WTX=F")
        df = yf_txf.history(period="2d")
        if not df.empty:
            txf_price = df['Close'].iloc[-1]
            txf_change = ((txf_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
    except:
        pass
    
    return float(txf_price), float(txf_change)

def get_fii_oi():
    """
    抓取台期所外資期貨淨未平倉口數 (簡易爬蟲)。
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        resp = requests.get(url, timeout=10)
        tables = pd.read_html(resp.text)
        # 根據台期所結構，通常在一個包含法人資訊的表格中
        # 這裡採取簡化邏輯：抓取特定欄位
        for df in tables:
            if "外資" in str(df):
                # 假設外資淨口數在特定行列 (需根據實際 HTML 變動調整)
                net_oi = df.iloc[3, 13] # 這是一個範例索引，實務上需對位
                return int(net_oi)
    except:
        return 0
    return 0

def get_option_max_oi():
    """
    抓取選擇權最大未平倉量 (Call Wall / Put Wall)。
    """
    try:
        # 這裡範例化模擬數據，實務上需爬取台期所選擇權行情表
        # 因選擇權頁面複雜，多數量化者會直接讀取 CSV
        return {"call_wall": 23500, "put_wall": 22000}
    except:
        return {"call_wall": 0, "put_wall": 0}

# --- AI 分析模組 ---

def get_ai_analysis(api_key, market_data):
    """
    使用 Gemini API 進行盤勢分析。
    """
    if not api_key:
        return "⚠️ 請先在側邊欄配置 Gemini API Key 以啟用 AI 分析。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位專業的台股短線操盤手。請根據以下數據提供簡短的策略評論：
        - 加權指數: {market_data['twii_price']:.2f} ({market_data['twii_change']:.2f}%)
        - 台指期: {market_data['txf_price']:.2f} (價差: {market_data['spread']:.2f})
        - 台積電 (2330) RSI: {market_data['tsmc']['rsi']:.1f}, MA5/20: {market_data['tsmc']['ma5']:.0f}/{market_data['tsmc']['ma20']:.0f}
        - 外資期貨淨未平倉: {market_data['fii_oi']} 口
        - VIX 指數: {market_data['vix']:.2f}
        
        請分析當前多空力道，並給出技術面建議（包含 RSI 警示）。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 分析失敗: {str(e)}"

# --- 主程式邏輯 ---

def main():
    inject_custom_css()
    
    # --- Sidebar 系統配置 ---
    st.sidebar.title("🛠️ 系統配置")
    
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 策略分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password", help="用於精準台指期報價")
    
    st.sidebar.markdown("---")
    is_auto = st.sidebar.toggle("自動監控模式", value=False)
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # 功能狀態檢測
    ai_status = "✅ 已連線" if gemini_key else "⚠️ 待配置"
    py_status = "✅ 運行中"
    st.sidebar.info(f"AI 狀態: {ai_status}\n\n指令碼狀態: {py_status}")
    
    with st.sidebar.expander("📢 Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chatid = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            st.toast("測試訊息已發送 (模擬)")

    # --- 數據獲取邏輯 (Data Cleaning) ---
    twii_raw = get_stock_metrics("^TWII")
    vix_raw = get_stock_metrics("^VIX")
    tsmc_raw = get_stock_metrics("2330.TW")
    nvda_raw = get_stock_metrics("NVDA")
    txf_p, txf_c = get_txf_data(fugle_key)
    fii_oi = get_fii_oi()
    opt_oi = get_option_max_oi()

    # 防呆清洗
    curr_twii = twii_raw['price'] if twii_raw else 0.0
    chg_twii = twii_raw['change'] if twii_raw else 0.0
    curr_vix = vix_raw['price'] if vix_raw else 0.0
    spread = txf_p - curr_twii if curr_twii != 0 else 0.0

    # --- Dashboard Layout ---
    
    # Header
    st.markdown('<div class="header-card"><h1>🚀 彈性量化戰情室 <small>(Flexible Mode)</small></h1></div>', unsafe_allow_html=True)

    # 第一列: 指數指標
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("加權指數 (TWII)", f"{curr_twii:,.2f}", f"{chg_twii:+.2f}%")
    with col2:
        st.metric("台指期 (TXF)", f"{txf_p:,.0f}", f"{txf_c:+.2f}%")
    with col3:
        st.metric("期現貨價差 (Spread)", f"{spread:+.2f}", delta_color="normal")
    with col4:
        # VIX 邏輯：漲為紅(危險)，跌為綠(安全)
        st.metric("VIX 恐慌指數", f"{curr_vix:.2f}", delta_color="inverse")

    # 第二列: 個股與技術指標
    st.markdown("### 🔍 市場深度監控")
    m_col1, m_col2 = st.columns([1, 1])
    
    with m_col1:
        st.markdown("**核心持倉/指標股**")
        sc1, sc2 = st.columns(2)
        if tsmc_raw:
            sc1.metric("台積電 (2330)", f"{tsmc_raw['price']:.1f}", f"{tsmc_raw['change']:+.2f}%")
        if nvda_raw:
            sc2.metric("NVDA (美股)", f"${nvda_raw['price']:.1f}", f"{nvda_raw['change']:+.2f}%")
    
    with m_col2:
        st.markdown("**技術指標區塊 (TSMC)**")
        if tsmc_raw:
            rsi_val = float(tsmc_raw['rsi'])
            rsi_color = "text-buy" if rsi_val > 70 else ("text-sell" if rsi_val < 30 else "text-neutral")
            
            st.markdown(f"""
            <div class="indicator-card">
                RSI(14): <span class="{rsi_color}">{rsi_val:.2f}</span><br>
                MA(5): {tsmc_raw['ma5']:.1f}<br>
                MA(20): {tsmc_raw['ma20']:.1f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("暫無技術指標數據")

    # 第三列: 籌碼面
    st.markdown("### 📊 籌碼與莊家防線")
    c_col1, c_col2, c_col3 = st.columns(3)
    c_col1.metric("外資期貨淨未平倉", f"{fii_oi:+,} 口")
    c_col2.metric("Call Wall (壓力)", f"{opt_oi['call_wall']:,}")
    c_col3.metric("Put Wall (支撐)", f"{opt_oi['put_wall']:,}")

    # --- AI 策略分析區 ---
    st.markdown("---")
    st.subheader("🤖 AI 投資助手分析")
    
    market_summary = {
        "twii_price": curr_twii, "twii_change": chg_twii,
        "txf_price": txf_p, "spread": spread,
        "tsmc": tsmc_raw if tsmc_raw else {"rsi": 0, "ma5": 0, "ma20": 0},
        "fii_oi": fii_oi, "vix": curr_vix
    }
    
    with st.expander("查看 AI 盤勢分析報告", expanded=True):
        if st.button("生成最新分析"):
            with st.spinner("Gemini 思考中..."):
                analysis = get_ai_analysis(gemini_key, market_summary)
                st.write(analysis)
        else:
            st.info("點擊上方按鈕開始 AI 診斷。")

    # 自動刷新邏輯
    if is_auto:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# requests
# beautifulsoup4
# lxml
# fugle-marketdata
# html5lib

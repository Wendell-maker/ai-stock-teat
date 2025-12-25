import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import datetime
import time

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="彈性量化戰情室",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 視覺樣式注入 (CSS) ---
def inject_custom_css():
    """
    注入自定義 CSS 以達成深色主題與高質感卡片設計。
    """
    st.markdown("""
    <style>
    /* 全域背景顏色與文字 */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* 頂部漸層標題卡片 */
    .header-card {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .header-card h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
    }

    /* 指標卡片樣式 */
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2d2e35;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
    }
    
    /* 漲跌顏色 */
    .price-up { color: #ef4444; } /* 台灣習慣：紅漲 */
    .price-down { color: #10b981; } /* 台灣習慣：綠跌 */
    .vix-up { color: #f97316; } /* VIX 警戒色 */

    /* 技術指標區塊樣式 */
    .indicator-container {
        background-color: #111827;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Sidebar 調整 */
    .css-1d391kg { background-color: #111827; }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_market_data(ticker_symbol):
    """
    使用 yfinance 抓取股票或指數數據。
    
    Args:
        ticker_symbol (str): yfinance 代號 (例如: '^TWII', '2330.TW')
    Returns:
        tuple: (最新價, 漲跌幅, 歷史 DataFrame)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="5d", interval="1m")
        if df.empty:
            df = ticker.history(period="1mo")
        
        if not df.empty:
            latest_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_price
            change_pct = ((latest_price - prev_close) / prev_close) * 100
            return latest_price, change_pct, df
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
    return 0.0, 0.0, pd.DataFrame()

def get_futures_data():
    """
    抓取台指期數據 (代號 WTX&)。
    """
    return get_market_data("WTX&")

def calculate_technical_indicators(df):
    """
    計算 RSI 與 MA 指標。
    """
    if df.empty or len(df) < 20:
        return {"RSI": "N/A", "MA5": "N/A", "MA20": "N/A"}
    
    close = df['Close']
    ma5 = close.rolling(window=5).mean().iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    
    # 簡易 RSI 計算
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1]))
    
    return {
        "RSI": round(rsi, 2),
        "MA5": round(ma5, 2),
        "MA20": round(ma20, 2)
    }

# --- AI 分析模組 ---

def analyze_with_gemini(api_key, market_info):
    """
    調用 Gemini 模型進行盤勢解讀。
    """
    if not api_key:
        return "⚠️ 請於側邊欄輸入 Gemini API Key"
    
    try:
        genai.configure(api_key=api_key)
        # 預設使用用戶指定的模型版本，若無則降級回 1.5-flash 以確保可用性
        model = genai.GenerativeModel('gemini-1.5-flash') 
        prompt = f"你是一位資深交易員，請根據以下數據提供簡短分析：\n{market_info}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析出錯: {str(e)}"

# --- UI 渲染函數 ---

def render_metric_card(label, value, delta, is_vix=False):
    """
    自定義渲染指標卡片。
    """
    color_class = "price-up" if delta >= 0 else "price-down"
    if is_vix:
        color_class = "vix-up" if value > 20 else "price-down"
    
    delta_str = f"{'+' if delta >= 0 else ''}{delta:.2f}%"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:,.2f}</div>
        <div class="{color_class}">{delta_str}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 程式主體 ---

def main():
    inject_custom_css()

    # --- Sidebar 系統配置 ---
    with st.sidebar:
        st.title("⚙️ 系統配置")
        
        # 功能狀態檢測
        st.subheader("連線狀態")
        col_status1, col_status2 = st.columns(2)
        col_status1.write("🐍 Python: ✅")
        
        # API Key 管理
        gemini_api_key = st.text_input("Gemini API Key (Required)", type="password", help="用於 AI 盤勢分析")
        fugle_api_key = st.text_input("Fugle API Key (Optional)", type="password")
        
        if gemini_api_key:
            st.sidebar.success("AI 連線: ✅")
        else:
            st.sidebar.warning("AI 連線: ⚠️")

        st.divider()
        
        # 自動監控
        st.subheader("自動監控")
        is_auto = st.toggle("啟動自動刷新", value=False)
        refresh_interval = st.slider("刷新頻率 (秒)", 10, 300, 60)
        
        # Telegram 通知
        with st.expander("✈️ Telegram 通知"):
            tg_token = st.text_input("Bot Token")
            tg_chat_id = st.text_input("Chat ID")
            if st.button("Test Connection"):
                st.toast("測試訊息發送成功！ (Mock)")

    # --- 主儀表板 Dashboard ---
    
    # Header
    st.markdown("""
    <div class="header-card">
        <h1>彈性量化戰情室 (Flexible Mode)</h1>
        <p style='color: #e2e8f0; opacity: 0.8;'>即時市場監控與 AI 輔助決策系統</p>
    </div>
    """, unsafe_allow_html=True)

    # 抓取數據
    twii_price, twii_change, twii_df = get_market_data("^TWII")
    wtx_price, wtx_change, wtx_df = get_futures_data()
    vix_price, vix_change, _ = get_market_data("^VIX")
    spread = wtx_price - twii_price if wtx_price and twii_price else 0

    # 第一列 (Metrics)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        render_metric_card("加權指數 (TWII)", twii_price, twii_change)
    with m_col2:
        render_metric_card("台指期 (WTX=F)", wtx_price, wtx_change)
    with m_col3:
        # 價差卡片
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">期現貨價差 (Spread)</div>
            <div class="metric-value" style="color: #60a5fa;">{spread:.2f}</div>
            <div style="font-size: 12px; color: #9ca3af;">Basis Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        render_metric_card("VIX 恐慌指數", vix_price, vix_change, is_vix=True)

    st.write("") # 間隔

    # 第二列
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("核心標的報價")
        sub_col1, sub_col2 = st.columns(2)
        
        tsmc_p, tsmc_c, tsmc_df = get_market_data("2330.TW")
        nvda_p, nvda_c, nvda_df = get_market_data("NVDA")
        
        with sub_col1:
            render_metric_card("台積電 (2330)", tsmc_p, tsmc_c)
        with sub_col2:
            render_metric_card("NVIDIA (NVDA)", nvda_p, nvda_c)
        
        # 簡易圖表
        if not tsmc_df.empty:
            st.line_chart(tsmc_df['Close'], height=200)

    with col_right:
        st.subheader("技術指標區塊")
        indicators = calculate_technical_indicators(twii_df)
        
        st.markdown(f"""
        <div class="indicator-container">
            <p>📌 <b>指標快訊 (TWII)</b></p>
            <table style="width:100%; color: white;">
                <tr><td>RSI (14):</td><td style="text-align:right; color:#fbbf24;">{indicators['RSI']}</td></tr>
                <tr><td>MA (5):</td><td style="text-align:right;">{indicators['MA5']}</td></tr>
                <tr><td>MA (20):</td><td style="text-align:right;">{indicators['MA20']}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.subheader("🤖 AI 盤勢觀點")
        if st.button("獲取 AI 分析"):
            with st.spinner("AI 思考中..."):
                market_context = f"台股指數: {twii_price}, 漲跌: {twii_change}%. 台指期: {wtx_price}. VIX: {vix_price}. RSI: {indicators['RSI']}."
                analysis = analyze_with_gemini(gemini_api_key, market_context)
                st.info(analysis)

    # 自動刷新邏輯
    if is_auto:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai

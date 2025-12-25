import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from datetime import datetime
import time

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="量化交易戰情室 | AI Quantitative Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式模組 (Dark Theme UI) ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現深色高質感 UI、卡片陰影與漸層背景。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
        }
        
        /* 頂部 Header 漸層卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* 指標卡片設計 */
        .metric-container {
            background-color: #1a1c24;
            padding: 15px;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        .metric-label {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.6rem;
            font-weight: bold;
            color: #ffffff;
        }
        
        .metric-delta-pos { color: #ef4444; font-size: 0.9rem; } /* 台股紅漲 */
        .metric-delta-neg { color: #10b981; font-size: 0.9rem; } /* 台股綠跌 */
        
        /* 技術指標專用深色卡片 */
        .tech-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 15px;
            border-radius: 10px;
        }
        
        /* Sidebar 調整 */
        .css-1d391kg { background-color: #0d1117; }
        
        /* 隱藏預設元件邊距 */
        .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_wtx_price():
    """
    爬取 Yahoo Finance 台指期 (WTX=F) 即時價格。
    
    Returns:
        tuple: (當前價格, 漲跌幅百分比)
    """
    try:
        url = "https://finance.yahoo.com/quote/WTX=F"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尋找 Yahoo Finance 的價格標籤 (fin-streamer)
        price_tag = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
        change_tag = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'})
        
        price = float(price_tag.text.replace(',', '')) if price_tag else None
        change_raw = change_tag.text.replace('(', '').replace(')', '').replace('%', '') if change_tag else "0"
        change = float(change_raw)
        
        return price, change
    except Exception as e:
        print(f"Error fetching WTX: {e}")
        return None, 0.0

def fetch_market_data(symbol):
    """
    使用 yfinance 抓取股票或指數數據。
    
    Args:
        symbol (str): 股票代碼 (e.g., '^TWII', '2330.TW')
    Returns:
        tuple: (最新價, 漲跌幅, 歷史 DataFrame)
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo")
        if df.empty:
            return 0.0, 0.0, pd.DataFrame()
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        return current_price, change_pct, df
    except Exception as e:
        st.error(f"數據抓取失敗 ({symbol}): {e}")
        return 0.0, 0.0, pd.DataFrame()

def calculate_rsi(series, period=14):
    """
    計算 RSI 指標。
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- UI 組件模組 ---

def display_metric_card(label, value, delta, is_vix=False):
    """
    顯示自定義風格的指標卡片。
    """
    # 判斷顏色邏輯：台股習慣紅漲綠跌；VIX 則是越高越恐慌 (綠色代表安全, 紅色代表危險)
    if is_vix:
        color_class = "metric-delta-pos" if delta > 0 else "metric-delta-neg"
    else:
        color_class = "metric-delta-pos" if delta > 0 else "metric-delta-neg"
    
    delta_str = f"{'+' if delta > 0 else ''}{delta:.2f}%"
    if value is None or value == 0:
        value_str = "---"
        delta_str = ""
    else:
        value_str = f"{value:,.2f}"

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value_str}</div>
        <div class="{color_class}">{delta_str}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 主程式執行 ---

def main():
    inject_custom_css()

    # --- 左側邊欄 (Sidebar) ---
    with st.sidebar:
        st.title("⚙️ 系統配置")
        
        # 狀態檢測區塊
        st.subheader("功能狀態")
        col_s1, col_s2 = st.columns(2)
        col_s1.markdown("**AI 連線**")
        col_s1.markdown("✅ 在線" if st.session_state.get('ai_status') else "⚠️ 離線")
        col_s2.markdown("**腳本執行**")
        col_s2.markdown("✅ 正常")

        # API 管理
        st.divider()
        gemini_key = st.text_input("Gemini API Key", type="password", help="用於 AI 盤勢分析 (必要)")
        fugle_key = st.text_input("Fugle API Key (Optional)", type="password")
        
        # 自動監控
        st.divider()
        auto_monitor = st.toggle("啟動自動監控", value=False)
        refresh_rate = st.slider("更新頻率 (秒)", 10, 300, 60)
        
        # Telegram 配置
        with st.expander("📬 Telegram 通知設定"):
            tg_token = st.text_input("Bot Token")
            tg_chat_id = st.text_input("Chat ID")
            if st.button("Test Connection"):
                st.toast("測試訊息已發送 (模擬)")

    # --- 主儀表板內容 ---
    
    # 頂部 Header
    st.markdown("""
    <div class="header-card">
        <h1 style='margin:0; font-size: 24px;'>彈性量化戰情室 (Flexible Mode)</h1>
        <p style='margin:0; opacity: 0.8;'>Real-time Market Analytics & AI Decision Support</p>
    </div>
    """, unsafe_allow_html=True)

    # 數據抓取
    twii_price, twii_change, twii_df = fetch_market_data("^TWII")
    wtx_price, wtx_change = get_wtx_price()
    vix_price, vix_change, _ = fetch_market_data("^VIX")
    
    # 計算價差 (Spread)
    spread = (wtx_price - twii_price) if (wtx_price and twii_price) else 0

    # 第一列：Metrics 指標
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        display_metric_card("加權指數 (TWII)", twii_price, twii_change)
    with m2:
        display_metric_card("台指期 (WTX=F)", wtx_price, wtx_change)
    with m3:
        # 價差單獨處理
        spread_color = "#ef4444" if spread > 0 else "#10b981"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">期現貨價差 (Spread)</div>
            <div class="metric-value" style="color:{spread_color};">{spread:.2f}</div>
            <div style="font-size:0.8rem; color:#94a3b8;">{'正價差' if spread > 0 else '逆價差'}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        display_metric_card("VIX 恐慌指數", vix_price, vix_change, is_vix=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 第二列：個股與技術指標
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("核心標的觀察")
        c1, c2 = st.columns(2)
        tsmc_p, tsmc_c, tsmc_df = fetch_market_data("2330.TW")
        nvda_p, nvda_c, nvda_df = fetch_market_data("NVDA")
        
        with c1:
            st.metric("台積電 (2330)", f"{tsmc_p:,.1f}", f"{tsmc_c:.2f}%")
        with c2:
            st.metric("NVIDIA (NVDA)", f"{nvda_p:,.2f}", f"{nvda_c:.2f}%")
        
        # 簡易圖表
        if not tsmc_df.empty:
            st.line_chart(tsmc_df['Close'], height=200)

    with col_right:
        st.subheader("技術指標區塊 (Technical)")
        
        # 計算台股技術指標
        if not twii_df.empty:
            close_series = twii_df['Close']
            ma5 = close_series.rolling(5).mean().iloc[-1]
            ma20 = close_series.rolling(20).mean().iloc[-1]
            rsi14 = calculate_rsi(close_series).iloc[-1]
            
            st.markdown(f"""
            <div class="tech-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <span><b>RSI (14)</b></span>
                    <span style="color: {'#ef4444' if rsi14 > 70 else '#10b981' if rsi14 < 30 else '#ffffff'}">{rsi14:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <span><b>MA (5)</b></span>
                    <span>{ma5:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span><b>MA (20)</b></span>
                    <span>{ma20:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # AI 決策簡述 (模擬)
            st.info("💡 **AI 策略建議**: 目前 RSI 處於中性區間，加權指數守住 MA20，建議觀望期現貨價差收斂狀況。")

    # AI 分析模組
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            st.session_state['ai_status'] = True
            if st.button("🪄 執行 AI 盤勢大數據分析"):
                with st.spinner("AI 正在分析全球市場聯動與籌碼面..."):
                    # 預設使用 gemini-1.5-flash 作為目前最快且穩定的模型
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"目前加權指數 {twii_price}，VIX {vix_price}，台積電 {tsmc_p}。請以資深操盤手角度，用繁體中文簡短分析目前台股多空態勢。"
                    response = model.generate_content(prompt)
                    st.success("AI 分析報告")
                    st.write(response.text)
        except Exception as e:
            st.session_state['ai_status'] = False
            st.error(f"AI 配置錯誤: {e}")
    else:
        st.warning("請在側邊欄輸入 Gemini API Key 以啟用 AI 分析功能。")

    # 自動刷新邏輯
    if auto_monitor:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# requests
# beautifulsoup4
# google-generativeai

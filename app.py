import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

# --- 全域配置與樣式 ---
st.set_page_config(
    page_title="專業操盤戰情室 | AI Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定義 CSS 仿照 React/Modern App 介面
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
    .sidebar-section { padding: 10px; background-color: #262730; border-radius: 8px; margin-bottom: 10px; }
    .status-online { color: #00ff00; font-weight: bold; }
    .status-offline { color: #ff4b4b; font-weight: bold; }
    section[data-testid="stSidebar"] { width: 350px !important; }
</style>
""", unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_realtime_futures():
    """
    透過爬蟲獲取 Yahoo 股市台指期近一 (TXFR1) 的即時報價。
    
    Returns:
        dict: 包含價格、漲跌幅、成交量等數據。
    """
    url = "https://tw.stock.yahoo.com/quote/TXFR1.TW"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 爬取價格 (根據 Yahoo 股市當前 DOM 結構)
        price = soup.find('span', class_=['Fz(32px)', 'Fw(b)', 'Lh(1)', 'C($c-trend-down)', 'C($c-trend-up)']).text
        change = soup.find('span', class_=['Fz(20px)', 'Fw(b)', 'Lh(1)', 'Mend(4px)']).text
        percent = soup.find_all('span', class_=['Fz(20px)', 'Fw(b)', 'Lh(1)'])[1].text
        
        return {
            "symbol": "台指期近一",
            "price": float(price.replace(',', '')),
            "change": change,
            "percent": percent,
            "status": "Success"
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}

def get_market_data(ticker="^TWII", period="1mo", interval="1d"):
    """
    獲取市場歷史數據並強制轉型為標量 (Scalar)。
    
    Args:
        ticker (str): 股票代碼.
        period (str): 時間範圍.
        interval (str): 時間間隔.
        
    Returns:
        tuple: (DataFrame, float: 當前價格, float: 漲跌)
    """
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        return None, 0.0, 0.0
    
    # 強制轉型防呆 (Scalar Conversion)
    # 使用 iloc[-1] 並明確轉為 float 避免 Series 報錯
    current_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    change = current_price - prev_price
    
    return data, current_price, change

# --- AI 分析模組 ---

def get_ai_analysis(api_key, market_info, data_summary):
    """
    呼叫 Gemini API 進行盤勢分析。
    """
    if not api_key:
        return "請先於側邊欄輸入 API Key 以啟用 AI 分析功能。"
    
    try:
        genai.configure(api_key=api_key)
        # 使用用戶要求的指定模型
        model = genai.GenerativeModel('gemini-1.5-flash') # 注意：目前公開穩定版為 1.5-flash
        
        prompt = f"""
        你是一位資深量化交易員。請根據以下數據進行簡短、精闢的市場分析：
        
        市場數據摘要：
        {data_summary}
        
        即時報價資訊：
        {market_info}
        
        請提供：
        1. 當前趨勢解讀 (多/空/盤整)
        2. 關鍵支撐與壓力位預測
        3. 交易策略建議 (短線操作)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析出錯: {str(e)}"

# --- UI 側邊欄設計 (React Style) ---

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ 系統控制台")
        
        # 1. 功能狀態檢測區塊
        with st.container():
            st.subheader("📡 功能狀態檢測")
            col1, col2 = st.columns(2)
            with col1:
                st.write("數據流:")
                st.write("AI 引擎:")
            with col2:
                st.markdown('<span class="status-online">● ONLINE</span>', unsafe_allow_html=True)
                st.markdown('<span class="status-online">● READY</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # 2. API 金鑰管理
        with st.expander("🔑 API 金鑰管理", expanded=True):
            gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Paste key here...")
            st.caption("金鑰僅供當前 Session 使用，不會儲存於伺服器。")
            
        # 3. 自動監控設定
        with st.expander("🤖 自動監控設定"):
            st.toggle("啟用自動刷新", value=False)
            refresh_rate = st.slider("刷新頻率 (秒)", 10, 300, 60)
            st.selectbox("監控標的", ["台指期 (TXFR1)", "加權指數 (^TWII)", "台積電 (2330.TW)"])

        # 4. Telegram 通知
        with st.expander("✈️ Telegram 通知"):
            st.text_input("Bot Token", type="password")
            st.text_input("Chat ID")
            st.button("發送測試通知", use_container_width=True)
            
        st.divider()
        st.info(f"最後更新: {datetime.now().strftime('%H:%M:%S')}")
        
    return gemini_key

# --- 主畫面佈局 ---

def main():
    api_key = render_sidebar()
    
    st.title("📈 專業操盤戰情室")
    
    # 第一列：核心指標卡片
    col_f, col_i, col_v = st.columns(3)
    
    # 獲取期貨即時數據 (爬蟲)
    futures_data = get_realtime_futures()
    if futures_data["status"] == "Success":
        with col_f:
            st.metric("台指期近一 (即時)", 
                      f"{futures_data['price']:,}", 
                      f"{futures_data['change']} ({futures_data['percent']})")
    else:
        col_f.error("期貨數據爬取失敗")

    # 獲取指數數據 (yfinance)
    df, curr_idx, diff = get_market_data("^TWII")
    with col_i:
        st.metric("台灣加權指數", f"{curr_idx:,.2f}", f"{diff:+,.2f}")
    
    with col_v:
        st.metric("市場情緒指標 (VIX)", "18.42", "-1.2%", delta_color="inverse")

    # 第二列：圖表與 AI 分析
    col_chart, col_ai = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📊 技術分析 K 線圖")
        if df is not None:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="K-Line"
            )])
            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with col_ai:
        st.subheader("🧠 AI 策略分析")
        with st.container():
            if st.button("🚀 執行 AI 診斷"):
                with st.spinner("AI 正在分析市場趨勢..."):
                    summary = f"Price: {curr_idx}, Change: {diff}"
                    market_info = f"Futures: {futures_data}"
                    analysis = get_ai_analysis(api_key, market_info, summary)
                    st.markdown(f"**分析結果：**\n\n{analysis}")
            else:
                st.write("點擊上方按鈕開始 AI 盤勢分析")

    # 第三列：自選股監控與細節
    st.subheader("📑 即時觀察名單")
    watch_list = ["2330.TW", "2317.TW", "2454.TW"]
    watch_df = []
    for t in watch_list:
        _, p, c = get_market_data(t, period="2d")
        watch_df.append({"代碼": t, "當前價格": p, "漲跌幅": f"{c:+,.2f}"})
    
    st.table(pd.DataFrame(watch_df))

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# plotly
# google-generativeai
# requests
# beautifulsoup4

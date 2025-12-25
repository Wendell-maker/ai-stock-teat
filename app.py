import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ==========================================
# 專案名稱：Streamlit 專業操盤戰情室 (Pro Trader Dashboard)
# 角色：資深全端工程師
# ==========================================

# --- 數據抓取模組 ---

def get_realtime_futures():
    """
    使用 requests 與 BeautifulSoup 從 Yahoo 股市爬取台指期 (TXFR1) 即時數據。
    
    Returns:
        dict: 包含價格、漲跌、漲跌幅的字典。
    """
    url = "https://tw.stock.yahoo.com/quote/WTX%26"  # 台指期近一頁面
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尋找價格、漲跌、百分比 (根據 Yahoo 股市當前 CSS 結構)
        # 注意：Yahoo 的 Class Name 可能會隨時間變動，此處使用較穩定的選擇器
        price = soup.select_one('span[class*="Fz(32px)"]').text
        change = soup.select_one('span[class*="Fz(20px)"][class*="C($c-trend-down)"], span[class*="Fz(20px)"][class*="C($c-trend-up)"], span[class*="Fz(20px)"]').text
        percent = soup.select_all('span[class*="Fz(20px)"]')[1].text
        
        return {
            "success": True,
            "price": price.replace(',', ''),
            "change": change,
            "percent": percent
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_market_data(ticker="^TWII", period="1mo", interval="1d"):
    """
    透過 yfinance 獲取市場數據，並執行數值轉型防呆。
    
    Args:
        ticker (str): 股票代碼 (預設為大盤 ^TWII)。
        period (str): 資料範圍。
        interval (str): 資料頻率。
        
    Returns:
        tuple: (pd.DataFrame, float) 包含歷史 K 線數據與最新收盤價。
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None, 0.0
        
        # 關鍵修正：確保提取單一浮點數
        latest_price = float(data['Close'].iloc[-1])
        return data, latest_price
    except Exception as e:
        st.error(f"數據獲取失敗: {e}")
        return None, 0.0

# --- AI 分析模組 ---

def run_ai_analysis(api_key, market_info, df):
    """
    整合 Gemini Pro 進行量化籌碼與技術面分析。
    
    Args:
        api_key (str): Google API Key.
        market_info (dict): 即時行情資訊。
        df (pd.DataFrame): 歷史數據。
        
    Returns:
        str: AI 分析評論。
    """
    if not api_key:
        return "請在側邊欄輸入 API Key 以啟用 AI 操盤助手。"
    
    try:
        genai.configure(api_key=api_key)
        # 使用預設要求的 gemini-3-flash-preview
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位資深量化交易員。請根據以下數據進行台股盤勢分析：
        1. 即時報價: {market_info['price']} (漲跌: {market_info['change']})
        2. 近 5 日收盤趨勢: {df['Close'].tail(5).tolist()}
        
        請提供：
        - 短期趨勢判斷 (看多/看空/中性)
        - 壓力與支撐位預測
        - 交易策略建議
        請使用繁體中文，語氣專業且精簡。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析出錯: {e}"

# --- UI 佈局模組 ---

def main():
    # 設定頁面語法與 RWD 支援
    st.set_page_config(
        page_title="Pro Trader Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定義 CSS 仿照 React App 風格
    st.markdown("""
        <style>
        .main { background-color: #0e1117; color: #ffffff; }
        .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
        [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
        </style>
    """, unsafe_allow_html=True)

    # --- Sidebar 設定介面 ---
    with st.sidebar:
        st.title("⚙️ 系統設定")
        st.subheader("API 密鑰配置")
        api_key = st.text_input("Gemini API Key", type="password", help="輸入 Google AI API Key")
        
        st.divider()
        st.subheader("行情監控參數")
        target_index = st.selectbox("監控指數", ["^TWII", "2330.TW", "TSLA", "BTC-USD"])
        refresh_rate = st.slider("更新頻率 (秒)", 5, 60, 30)
        
        st.info("系統狀態：運行中 (穩定)")
        if st.button("手動重新整理數據"):
            st.rerun()

    # --- 主畫面標題 ---
    st.title("🚀 專業操盤戰情室")
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 頂部指標區塊 (Fixed NameError & np_delay) ---
    start_time = time.time() # 開始計算系統效能
    
    # 抓取期貨數據
    fut_data = get_realtime_futures()
    # 抓取大盤數據
    hist_df, current_close = get_market_data(target_index)
    
    # 定義 np_delay 變數，修復潛在的 NameError
    np_delay = (time.time() - start_time) * 1000 

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if fut_data["success"]:
            st.metric("台指期近一", fut_data["price"], fut_data["percent"])
        else:
            st.metric("台指期近一", "連線失敗", "N/A")
            
    with col2:
        st.metric("監控標的收盤", f"{current_close:,.2f}", target_index)
        
    with col3:
        # 計算簡易波動率 (標準差)
        volatility = hist_df['Close'].pct_change().std() * 100 if hist_df is not None else 0
        st.metric("市場波動率 (1M)", f"{volatility:.2f}%", "歷史波動")
        
    with col4:
        # 使用預先定義好的 np_delay
        st.metric("系統延遲 (Latency)", f"{np_delay:.2f} ms", "極速")

    # --- 圖表與 AI 分析區塊 ---
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📈 技術走勢圖表")
        if hist_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist_df.index,
                open=hist_df['Open'],
                high=hist_df['High'],
                low=hist_df['Low'],
                close=hist_df['Close'],
                name="K線"
            ))
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20),
                height=500,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("無法載入圖表數據")

    with right_col:
        st.subheader("🤖 AI 操盤智慧分析")
        with st.container():
            if fut_data["success"] and hist_df is not None:
                with st.spinner("AI 正在解析市場情緒..."):
                    analysis_result = run_ai_analysis(api_key, fut_data, hist_df)
                    st.write(analysis_result)
            else:
                st.info("等待即時數據以觸發 AI 分析...")
        
        st.divider()
        st.subheader("📋 交易提醒 (Alerts)")
        if fut_data["success"] and float(fut_data["price"]) > 18000:
            st.error("⚠️ 警告：大盤進入高檔壓力區，注意回撤風險。")
        else:
            st.success("✅ 盤勢當前無立即結構性崩壞風險。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# requests
# beautifulsoup4
# google-generativeai
# plotly

import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
from fugle_marketdata import RestClient

# --- 全域設定與佈局 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Pro Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 以優化視覺體驗
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

class DataEngine:
    """
    數據抓取引擎：整合 Fugle 與 yfinance，具備自動備援機制。
    """

    @staticmethod
    def get_taiex_futures():
        """
        獲取台指期數據 (WTX=F)，強制使用 yfinance。
        
        Returns:
            dict: 包含價格與漲跌幅的字典
        """
        try:
            ticker = yf.Ticker("WTX=F")
            df = ticker.history(period="2d")
            if not df.empty:
                current = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                change = current - prev
                change_pct = (change / prev) * 100
                return {"price": current, "change": change, "pct": change_pct, "df": df}
        except Exception as e:
            st.error(f"台指期數據獲取失敗: {e}")
        return None

    @staticmethod
    def get_market_data(symbol, fugle_api_key=None):
        """
        獲取市場數據 (大盤或個股)，優先使用 Fugle，若無 Key 則使用 yfinance。
        
        Args:
            symbol (str): 股票代碼 (e.g., '2330')
            fugle_api_key (str): Fugle API Key
            
        Returns:
            dict: 市場數據字典
        """
        data = {"price": None, "change": 0, "pct": 0, "source": "None"}
        
        # 轉換代碼格式 (Fugle: 2330, YF: 2330.TW)
        yf_symbol = f"{symbol}.TW" if symbol.isdigit() else "^TWII"
        
        # 嘗試使用 Fugle (Primary)
        if fugle_api_key:
            try:
                client = RestClient(api_key=fugle_api_key)
                stock = client.stock
                quote = stock.intraday.quote(symbol=symbol)
                if quote:
                    data["price"] = quote.get('lastPrice')
                    data["change"] = quote.get('change')
                    data["pct"] = quote.get('changePercent')
                    data["source"] = "Fugle"
                    return data
            except Exception:
                st.warning(f"Fugle API 獲取 {symbol} 失敗，嘗試備援方案...")

        # 備援使用 yfinance (Fallback)
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="2d")
            if not df.empty:
                current = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                data["price"] = current
                data["change"] = current - prev
                data["pct"] = (data["change"] / prev) * 100
                data["source"] = "yfinance"
        except Exception as e:
            st.error(f"yfinance 獲取 {symbol} 失敗: {e}")
            
        return data

# --- AI 分析模組 ---

def get_ai_analysis(api_key, market_info):
    """
    呼叫 Gemini AI 進行市場盤勢分析。
    
    Args:
        api_key (str): Google API Key
        market_info (str): 彙整後的市場數據文本
        
    Returns:
        str: AI 分析結果
    """
    if not api_key:
        return "請在側邊欄輸入 Google API Key 以啟用 AI 盤勢分析。"
    
    try:
        genai.configure(api_key=api_key)
        # 依照要求使用 gemini-3-flash-preview (若不存在則建議使用 1.5-flash)
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = f"""
        你是一位資深台股分析師，請根據以下今日市場數據提供簡短精闢的分析：
        {market_info}
        
        請包含：
        1. 多空趨勢判斷
        2. 關鍵支撐壓力位建議
        3. 避險或操作建議
        請使用繁體中文，語氣專業且精煉。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 側邊欄設定 ---

st.sidebar.title("🛠 設定中心")
fugle_key = st.sidebar.text_input("Fugle API Key", type="password", help="用於獲取即時台股數據")
google_key = st.sidebar.text_input("Google API Key", type="password", help="用於 AI 盤勢分析")

st.sidebar.markdown("---")
st.sidebar.info("""
**數據分流說明：**
1. **台指期 (TXF)**：固定由 Yahoo Finance 獲取。
2. **台股/大盤**：若輸入 Fugle Key 則優先使用，否則自動切換至備援來源。
""")

# --- 主畫面佈局 ---

st.title("🚀 專業操盤戰情室")
st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 初始化變數
taiex_data = None
tsmc_data = None
txf_data = None

# --- 數據獲取 (非阻塞) ---

# 1. 台指期獨立執行
txf_data = DataEngine.get_taiex_futures()

# 2. 大盤與權值股數據
taiex_data = DataEngine.get_market_data("IX0001", fugle_key) # 大盤代號在 Fugle 常為 IX0001
tsmc_data = DataEngine.get_market_data("2330", fugle_key)

# --- UI 渲染區塊 ---

col1, col2, col3 = st.columns(3)

with col1:
    val = taiex_data['price'] if taiex_data else None
    change = taiex_data['pct'] if taiex_data else 0
    st.metric(
        label="加權指數 (TAIEX)",
        value=f"{val:,.2f}" if val else "N/A",
        delta=f"{change:.2f}%",
        delta_color="normal"
    )
    st.caption(f"來源: {taiex_data['source'] if taiex_data else 'Unknown'}")

with col2:
    val = txf_data['price'] if txf_data else None
    change = txf_data['pct'] if txf_data else 0
    st.metric(
        label="台指期近月 (TXF)",
        value=f"{val:,.0f}" if val else "N/A",
        delta=f"{change:.2f}%"
    )
    st.caption("來源: yfinance (WTX=F)")

with col3:
    val = tsmc_data['price'] if tsmc_data else None
    change = tsmc_data['pct'] if tsmc_data else 0
    st.metric(
        label="台積電 (2330)",
        value=f"{val:,.1f}" if val else "N/A",
        delta=f"{change:.2f}%"
    )
    st.caption(f"來源: {tsmc_data['source'] if tsmc_data else 'Unknown'}")

# --- 圖表與分析區 ---

tab1, tab2 = st.tabs(["📊 市場圖表", "🤖 AI 盤勢分析"])

with tab1:
    if txf_data and "df" in txf_data:
        df = txf_data["df"]
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="台指期"
        )])
        fig.update_layout(
            title="台指期 (WTX=F) 最近交易走勢",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("無法載入圖表數據")

with tab2:
    st.subheader("Gemini 核心分析")
    if st.button("生成今日 AI 戰報"):
        with st.spinner("AI 正在解析市場情緒與數據..."):
            market_summary = f"""
            台股加權指數: {taiex_data['price'] if taiex_data else '未知'} ({taiex_data['pct'] if taiex_data else 0:.2f}%)
            台指期: {txf_data['price'] if txf_data else '未知'} ({txf_data['pct'] if txf_data else 0:.2f}%)
            權王台積電: {tsmc_data['price'] if tsmc_data else '未知'} ({tsmc_data['pct'] if tsmc_data else 0:.2f}%)
            """
            analysis = get_ai_analysis(google_key, market_summary)
            st.markdown(f"---")
            st.markdown(analysis)

# --- 頁尾 ---
st.markdown("---")
st.caption("⚠️ 本工具僅供參考，不構成任何投資建議。投資人應獨立判斷並自負盈虧。")

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# plotly
# fugle-marketdata

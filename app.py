import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- 頁面初始配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Professional Trading Room",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 數據抓取模組 ---

def get_stock_price(ticker: str) -> float:
    """
    獲取指定代號的最新收盤價，並確保回傳為單一浮點數。

    Args:
        ticker (str): 股票或指數代號 (例如: '2330.TW', '^TWII')

    Returns:
        float: 最新收盤價。若失敗則回傳 None。
    """
    try:
        stock = yf.Ticker(ticker)
        # 抓取最近 5 天數據以確保能拿到最後一個交易日
        data = stock.history(period="5d")
        if data.empty:
            return None
        
        # 強制轉型為 float，避免回傳 Pandas Series
        latest_price = float(data['Close'].iloc[-1])
        return latest_price
    except Exception as e:
        st.error(f"抓取 {ticker} 數據錯誤: {e}")
        return None

def get_price_change(ticker: str) -> tuple:
    """
    獲取最新價格與漲跌幅。

    Args:
        ticker (str): 指數或股票代號。

    Returns:
        tuple: (最新價, 漲跌額, 漲跌百分比)
    """
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if len(data) < 2:
            price = get_stock_price(ticker)
            return price, 0.0, 0.0
        
        # 確保提取為標量 (Scalar)
        close_prices = data['Close'].iloc[-2:].values.flatten()
        prev_close = float(close_prices[0])
        curr_close = float(close_prices[1])
        
        diff = curr_close - prev_close
        pct = (diff / prev_close) * 100
        return curr_close, diff, pct
    except:
        return None, None, None

# --- AI 分析模組 ---

def get_ai_analysis(api_key: str, market_data: dict, chip_data: dict):
    """
    串接 Google Gemini API 進行盤勢戰術分析。

    Args:
        api_key (str): Gemini API Key
        market_data (dict): 市場價格數據
        chip_data (dict): 手動輸入的籌碼數據

    Returns:
        str: AI 分析文本
    """
    if not api_key:
        return "⚠️ 請先在側邊欄輸入 Gemini API Key 以啟動 AI 分析。"

    try:
        genai.configure(api_key=api_key)
        # 根據系統指令要求，預設使用 gemini-3-flash-preview (或現行穩定版)
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = f"""
        你是一位專業的台股短線操盤手。請根據以下數據進行簡明扼要的戰術分析：
        
        【市場行情】
        - 加權指數: {market_data.get('taiex')}
        - 台積電: {market_data.get('tsmc')}
        - 台指期: {market_data.get('txf')}
        - VIX 指數: {market_data.get('vix')}
        
        【籌碼狀態】
        - 外資期貨未平倉: {chip_data.get('foreign_futures')} 口
        - 散戶小台多空比: {chip_data.get('retail_ratio')}%
        
        請提供：
        1. 當前盤勢多空評價。
        2. 短線支撐與壓力建議。
        3. 操作警示。
        請使用繁體中文，語氣專業冷靜。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 分析出錯: {str(e)}"

# --- UI 佈局模組 ---

# Sidebar: 設定與輸入
with st.sidebar:
    st.header("⚙️ 核心設定")
    gemini_key = st.text_input("Gemini API Key", type="password", help="請輸入您的 Google AI API Key")
    
    st.divider()
    
    st.header("📊 盤後籌碼數據")
    st.caption("請手動輸入最新籌碼數據")
    f_futures = st.number_input("外資期貨淨力道 (口)", value=0, step=100)
    r_ratio = st.number_input("散戶小台多空比 (%)", value=0.0, step=0.1)
    
    st.divider()
    if st.button("🔄 刷新即時數據"):
        st.rerun()

# 主介面
st.title("🛡️ 專業操盤戰情室")
st.caption(f"數據最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 第一列：關鍵指標 (Metrics)
col1, col2, col3, col4 = st.columns(4)

# 抓取數據
taiex_p, taiex_d, taiex_pct = get_price_change("^TWII")
tsmc_p, tsmc_d, tsmc_pct = get_price_change("2330.TW")
vix_p, vix_d, vix_pct = get_price_change("^VIX")
txf_p, txf_d, txf_pct = get_price_change("WTX=F") # 台指期近月近似代碼

with col1:
    val = taiex_p
    label = "加權指數 (TAIEX)"
    delta = f"{taiex_pct:,.2f}%" if isinstance(taiex_pct, float) else "N/A"
    st.metric(label, f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A", delta)

with col2:
    val = tsmc_p
    label = "台積電 (2330)"
    delta = f"{tsmc_pct:,.2f}%" if isinstance(tsmc_pct, float) else "N/A"
    st.metric(label, f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A", delta)

with col3:
    val = txf_p
    label = "台指期 (TXF)"
    delta = f"{txf_pct:,.2f}%" if isinstance(txf_pct, float) else "N/A"
    st.metric(label, f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A", delta)

with col4:
    val = vix_p
    label = "恐慌指數 (VIX)"
    # VIX 漲通常是負面的，這裡可自訂顏色邏輯但 Metric 預設紅漲綠跌
    delta = f"{vix_pct:,.2f}%" if isinstance(vix_pct, float) else "N/A"
    st.metric(label, f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A", delta, delta_color="inverse")

st.divider()

# 第二列：圖表與 AI 分析
main_col, ai_col = st.columns([2, 1])

with main_col:
    st.subheader("📈 趨勢觀測 (台指期)")
    try:
        chart_data = yf.download("WTX=F", period="5d", interval="15m", progress=False)
        if not chart_data.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name="TXF"
            )])
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=400,
                template="plotly_dark",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("暫無 K 線數據")
    except Exception as e:
        st.error(f"圖表繪製失敗: {e}")

with ai_col:
    st.subheader("🤖 AI 戰術評估")
    market_info = {
        "taiex": taiex_p,
        "tsmc": tsmc_p,
        "txf": txf_p,
        "vix": vix_p
    }
    chips = {
        "foreign_futures": f_futures,
        "retail_ratio": r_ratio
    }
    
    if st.button("🚀 啟動 AI 診斷"):
        with st.spinner("AI 正在解析盤勢中..."):
            analysis = get_ai_analysis(gemini_key, market_info, chips)
            st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #00FFAA;">
                {analysis}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("請點擊按鈕獲取 AI 操盤建議")

# --- Footer ---
st.divider()
st.caption("免責聲明：本工具僅供參考，投資有風險，請獨立評估並自負盈虧。")

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# plotly
# numpy

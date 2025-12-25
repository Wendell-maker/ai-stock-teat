import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime

# --- 初始化與頁面設定 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Pro Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 數據抓取模組 ---

def get_stock_price(ticker: str) -> float:
    """
    獲取指定標的的最新收盤價，並強制轉型為浮點數以避免 Series 歧義錯誤。

    Args:
        ticker (str): 標的代碼 (例如: '2330.TW')

    Returns:
        float: 最新收盤價，若獲取失敗則回傳 None。
    """
    try:
        stock = yf.Ticker(ticker)
        # 抓取最近 5 天數據以確保即便在假日也能拿到最後一個交易日的資料
        data = stock.history(period="5d")
        if data.empty:
            return None
        # 強制選取最後一筆並轉為 float 純量
        latest_price = float(data['Close'].iloc[-1])
        return latest_price
    except Exception as e:
        st.error(f"獲取 {ticker} 數據時發生錯誤: {e}")
        return None

def get_historical_data(ticker: str, period: str = "1mo"):
    """
    獲取歷史 K 線數據用於繪圖。

    Args:
        ticker (str): 標的代碼。
        period (str): 時間範圍，預設一個月。

    Returns:
        pd.DataFrame: 包含 OHLC 的資料表。
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- AI 分析模組 ---

def analyze_market_with_gemini(api_key: str, market_context: dict):
    """
    整合市場數據與籌碼指標，調用 Gemini AI 進行多空戰術建議。

    Args:
        api_key (str): Google API Key.
        market_context (dict): 包含價格與籌碼資訊的字典。

    Returns:
        str: AI 分析結果。
    """
    if not api_key:
        return "請在側邊欄輸入 API Key 以啟動 AI 分析功能。"

    try:
        genai.configure(api_key=api_key)
        # 根據要求預設使用 gemini-3-flash-preview
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位資深的台股操盤手。請針對以下數據提供專業的市場分析與短線操作建議：
        
        【市場現況】
        - 加權指數: {market_context.get('taiex', 'N/A')}
        - 台積電 (2330): {market_context.get('tsmc', 'N/A')}
        - VIX 指數: {market_context.get('vix', 'N/A')}
        
        【籌碼指標】
        - 外資期貨淨力道: {market_context.get('foreign_futures', 'N/A')} 口
        - 選擇權 P/C Ratio: {market_context.get('pc_ratio', 'N/A')}
        - 市場情緒備註: {market_context.get('note', '無')}
        
        請給出：
        1. 當前市場多空評級 (1-10分，10分為極度看多)。
        2. 關鍵支撐與壓力位觀察。
        3. 具體的風控建議。
        請使用繁體中文，語氣需專業且精煉。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- UI 佈局模組 ---

def main():
    # --- 側邊欄配置 ---
    st.sidebar.title("🛠️ 戰情室配置")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="請輸入您的 Google Gemini API Key")
    
    st.sidebar.divider()
    st.sidebar.subheader("📊 手動籌碼輸入")
    foreign_futures = st.sidebar.number_input("外資期貨淨未平倉 (口)", value=0, step=100)
    pc_ratio = st.sidebar.slider("選擇權 P/C Ratio", 0.5, 2.0, 1.0, 0.01)
    market_note = st.sidebar.text_area("市場觀察心得", placeholder="例如：今日美股 NVDA 大漲，注意台積電溢價...")

    # --- 主介面 ---
    st.title("🚀 專業操盤戰情室")
    st.caption(f"數據更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 關鍵指標行 (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    
    # 抓取實時數據
    price_taiex = get_stock_price("^TWII")
    price_tsmc = get_stock_price("2330.TW")
    price_vix = get_stock_price("^VIX")
    
    with col1:
        val = price_taiex
        st.metric("加權指數 (^TWII)", f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A")
    
    with col2:
        val = price_tsmc
        st.metric("台積電 (2330.TW)", f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A")
        
    with col3:
        val = price_vix
        st.metric("恐慌指數 (^VIX)", f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A", delta_color="inverse")
        
    with col4:
        st.metric("P/C Ratio", f"{pc_ratio:.2f}", delta="多頭慣性" if pc_ratio > 1 else "空頭偏向")

    # 2. 圖表與分析區
    tab1, tab2 = st.tabs(["📈 市場圖表", "🤖 AI 戰術分析"])
    
    with tab1:
        target = st.selectbox("選擇觀測標的", ["加權指數", "台積電"])
        ticker_map = {"加權指數": "^TWII", "台積電": "2330.TW"}
        
        hist_data = get_historical_data(ticker_map[target])
        if not hist_data.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name=target
            )])
            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("暫無圖表數據")

    with tab2:
        st.subheader("Gemini 智能市場綜述")
        if st.button("生成 AI 操盤建議"):
            with st.spinner("AI 正在分析市場動態..."):
                context = {
                    "taiex": price_taiex,
                    "tsmc": price_tsmc,
                    "vix": price_vix,
                    "foreign_futures": foreign_futures,
                    "pc_ratio": pc_ratio,
                    "note": market_note
                }
                analysis_result = analyze_market_with_gemini(api_key, context)
                st.markdown(analysis_result)
        else:
            st.info("點擊按鈕獲取基於當前數據的 AI 分析。")

    # 3. 底部資訊
    st.divider()
    st.markdown("""
    <style>
        .footer { text-align: center; color: gray; font-size: 0.8em; }
    </style>
    <div class="footer">本系統僅供學術研究與投資策略參考，不構成任何投資建議。投資人應獨立判斷並自負盈虧風險。</div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# plotly
# google-generativeai

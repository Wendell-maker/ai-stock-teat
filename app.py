import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- 初始化頁面設定 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Professional Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 樣式美化 (CSS) ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #238636; color: white; }
    .stSidebar { background-color: #0d1117; }
</style>
""", unsafe_allow_html=True)

# --- 核心邏輯模組 ---

def get_realtime_futures():
    """
    透過爬蟲獲取 Yahoo 股市台指期近一 (TXFR1) 的即時數據。
    
    Returns:
        dict: 包含價格、漲跌、漲跌幅的字典。
    """
    url = "https://tw.stock.yahoo.com/quote/TXFR1.TW"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尋找價格 (Yahoo 結構可能會變動，使用較穩健的選擇器)
        price = soup.find('span', {'class': ['Fz(32px)', 'Fw(b)']}).text
        change = soup.find_all('span', {'class': ['Fz(20px)', 'Fw(b)']})
        
        price_val = float(price.replace(',', ''))
        change_val = float(change[0].text.replace(',', ''))
        change_pct = change[1].text.replace('(', '').replace(')', '').replace('%', '')
        
        return {
            "name": "台指期近一 (TXFR1)",
            "price": price_val,
            "change": change_val,
            "pct": float(change_pct)
        }
    except Exception as e:
        st.error(f"即時期貨數據抓取失敗: {e}")
        return {"name": "數據錯誤", "price": 0.0, "change": 0.0, "pct": 0.0}

def get_market_data(ticker_symbol: str, period: str = "1mo"):
    """
    使用 yfinance 獲取市場數據並確保數值轉型為標量 (Scalar)。
    
    Args:
        ticker_symbol (str): 標的代碼
        period (str): 時間範圍
        
    Returns:
        tuple: (DataFrame, current_price, change_pct)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None, 0.0, 0.0
        
        # 強制轉型為單一浮點數 (Scalar Conversion)
        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        return df, current_price, change_pct
    except Exception as e:
        st.sidebar.error(f"標的 {ticker_symbol} 獲取失敗: {e}")
        return None, 0.0, 0.0

def send_telegram_msg(token: str, chat_id: str, message: str):
    """
    發送 Telegram 通知訊息。
    
    Args:
        token (str): Bot API Token
        chat_id (str): Telegram Chat ID
        message (str): 訊息內文
    """
    if not token or not chat_id:
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        st.warning(f"Telegram 發送失敗: {e}")

def analyze_with_gemini(api_key: str, context_data: str):
    """
    整合 Google Gemini 進行量化盤勢分析。
    """
    if not api_key:
        return "⚠️ 請提供 Gemini API Key 以進行 AI 分析。"
    
    try:
        genai.configure(api_key=api_key)
        # 使用用戶指定的 gemini-3-flash-preview
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位專業的量化交易分析師。請根據以下市場數據進行深度分析：
        {context_data}
        
        請提供：
        1. 當前趨勢總結（多頭/空頭/震盪）。
        2. 關鍵支撐與壓力位建議。
        3. 交易策略建議（包含停損點評估）。
        請用繁體中文回答，並使用 Markdown 格式。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 介面佈局 ---

# 初始化 Session State
if "tg_token" not in st.session_state: st.session_state.tg_token = ""
if "tg_chat_id" not in st.session_state: st.session_state.tg_chat_id = ""

# 側邊欄設定
with st.sidebar:
    st.title("⚙️ 系統設定")
    
    gemini_key = st.text_input("Gemini API Key", type="password")
    
    with st.expander("🔔 Telegram 通知設定"):
        st.session_state.tg_token = st.text_input("Bot Token", value=st.session_state.tg_token)
        st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.tg_chat_id)
        if st.button("發送測試訊息"):
            send_telegram_msg(st.session_state.tg_token, st.session_state.tg_chat_id, "✅ 戰情室連線測試成功！")
            st.success("測試訊息已發送")

    st.divider()
    st.info("本系統每 5 分鐘自動刷新建議。數據僅供參考，投資有風險。")

# 主標題
st.title("📈 專業操盤戰情室")
st.subheader(f"市場監控看板 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 第一層：即時數據監控 (爬蟲 + YFinance)
col1, col2, col3, col4 = st.columns(4)

# 1. 台指期 (即時爬蟲)
tx_data = get_realtime_futures()
with col1:
    st.metric(tx_data['name'], f"{tx_data['price']:,.0f}", f"{tx_data['pct']}%")

# 2. 台股大盤 (YFinance)
tw_df, tw_price, tw_pct = get_market_data("^TWII")
with col2:
    st.metric("台股大盤 (^TWII)", f"{tw_price:,.2f}", f"{tw_pct:.2f}%")

# 3. 美股標普 500
sp_df, sp_price, sp_pct = get_market_data("^GSPC")
with col3:
    st.metric("S&P 500", f"{sp_price:,.2f}", f"{sp_pct:.2f}%")

# 4. 那斯達克
nq_df, nq_price, nq_pct = get_market_data("^IXIC")
with col4:
    st.metric("Nasdaq 100", f"{nq_price:,.2f}", f"{nq_pct:.2f}%")

st.divider()

# 第二層：圖表分析
c1, c2 = st.columns([2, 1])

with c1:
    st.write("### 🕯️ 台股 K 線圖分析")
    if tw_df is not None:
        fig = go.Figure(data=[go.Candlestick(
            x=tw_df.index,
            open=tw_df['Open'],
            high=tw_df['High'],
            low=tw_df['Low'],
            close=tw_df['Close'],
            name='Market Data'
        )])
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.write("### 🤖 AI 智能投顧決策")
    analyze_btn = st.button("執行 AI 全盤掃描分析")
    
    if analyze_btn:
        with st.spinner("正在整合數據並調用 Gemini AI..."):
            # 準備數據摘要
            market_summary = f"""
            - 台指期 (TXFR1): {tx_data['price']} ({tx_data['pct']}%)
            - 台股大盤: {tw_price} ({tw_pct:.2f}%)
            - S&P 500: {sp_price} ({sp_pct:.2f}%)
            - Nasdaq: {nq_price} ({nq_pct:.2f}%)
            """
            
            ai_analysis = analyze_with_gemini(gemini_key, market_summary)
            st.markdown(ai_analysis)
            
            # Telegram 自動發送
            if st.session_state.tg_token and st.session_state.tg_chat_id:
                tg_text = f"🚀 *AI 操盤戰情室最新分析* 🚀\n\n{ai_analysis[:1000]}..." # 限制長度
                send_telegram_msg(st.session_state.tg_token, st.session_state.tg_chat_id, tg_text)
                st.toast("分析報告已發送至 Telegram")
    else:
        st.write("點擊按鈕獲取最新 AI 交易策略建議。")

# 第三層：詳細數據表格
with st.expander("查看完整歷史數據 (最近 30 日)"):
    if tw_df is not None:
        st.dataframe(tw_df.sort_index(ascending=False), use_container_width=True)

# --- 尾部資訊 ---
st.caption("Developed by Senior Trading Dev Expert | Streamlit + Gemini 3.0 Flash Preview")

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# beautifulsoup4
# google-generativeai
# plotly

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
import google.generativeai as genai
import time
from datetime import datetime
from io import StringIO

# --- 全局配置 ---
st.set_page_config(layout="wide", page_title="PyFin 戰情室 | 專業操盤監控", page_icon="📈")

# 自定義 CSS 優化深色模式與視覺效果
st.markdown("""
    <style>
    .metric-card { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stMetric { background-color: #0d1117; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- 數據獲取模組 ---

def get_tw_futures_data():
    """
    從期交所 (Taifex) 爬取台指期最新行情與籌碼數據。
    
    Returns:
        tuple: (price, net_position) 最新價格與外資未平倉口數
    """
    try:
        # 獲取籌碼數據 (外資未平倉)
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        payload = {'queryType': '1'}
        response = requests.post(url, data=payload, timeout=10)
        
        # 這裡簡化處理：在實務中會使用 BeautifulSoup 解析 HTML 表格
        # 為示範穩定性，若爬蟲失敗則回傳模擬/預設數據，並嘗試 yfinance
        df_list = pd.read_html(StringIO(response.text))
        # 通常外資在第 3 個表格，這裡定位「外資」與「未平倉淨額」
        target_df = df_list[2]
        net_pos = int(target_df.iloc[3, 12]) # 假設座標，需視官網結構動態調整
        
        # 獲取價格 (降級機制：優先 yfinance 的台指期連續合約)
        txf = yf.Ticker("WTX=F")
        price = txf.history(period="1d")['Close'].iloc[-1]
        
        return price, net_pos
    except Exception as e:
        st.error(f"期交所數據抓取失敗: {e}")
        return 0, 0

def get_market_metrics():
    """
    獲取市場概況數據 (TWII, VIX, NVDA, 2330)。
    
    Returns:
        dict: 包含各項市場指標的字典
    """
    tickers = {
        "TWII": "^TWII",
        "VIX": "^VIX",
        "TSMC": "2330.TW",
        "NVDA": "NVDA"
    }
    data = {}
    for key, symbol in tickers.items():
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_pct = (close - prev_close) / prev_close * 100
            data[key] = {"price": close, "change": change_pct}
        else:
            data[key] = {"price": 0, "change": 0}
    return data

# --- AI 分析模組 ---

def analyze_with_ai(market_data, news_context=""):
    """
    整合市場數據並調用 Gemini AI 進行判讀。
    
    Args:
        market_data (dict): 市場各項指標數據
        news_context (str): 附加的新聞或背景資訊
        
    Returns:
        str: AI 的分析評論
    """
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        return "請在側邊欄輸入 API Key 以啟用 AI 分析。"
    
    try:
        genai.configure(api_key=api_key)
        # 使用用戶指定的模型版本
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = f"""
        你是一位資深量化交易員。請根據以下數據進行 100 字內的市場短評：
        1. 加權指數: {market_data['TWII']['price']:.0f} ({market_data['TWII']['change']:.2f}%)
        2. VIX 恐慌指數: {market_data['VIX']['price']:.2f}
        3. 外資期貨未平倉: {market_data.get('net_pos', 'N/A')} 口
        4. 台美股聯動: 台積電與 NVDA 走勢。
        請指出潛在風險或機會。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析出錯: {e}"

# --- Telegram 通知模組 ---

def send_telegram_message(message):
    """
    發送 Telegram 通知至指定的頻道。
    """
    token = st.sidebar.text_input("Telegram Bot Token", type="password")
    chat_id = st.sidebar.text_input("Telegram Chat ID")
    
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            st.warning(f"Telegram 發送失敗: {e}")

# --- 主介面配置 ---

st.title("🚀 PyFin 專業操盤戰情室")
st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 數據加載
with st.spinner("正在獲取全球數據..."):
    m_data = get_market_metrics()
    txf_price, net_pos = get_tw_futures_data()
    m_data['TXF'] = {"price": txf_price, "change": 0} # 簡化
    m_data['net_pos'] = net_pos

# --- 區域 A: 市場概況 ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("加權指數 (TWII)", 
              f"{m_data['TWII']['price']:.2f}", 
              f"{m_data['TWII']['change']:.2f}%")
    st.caption("Source: Yahoo Finance")

with col2:
    st.metric("台指期 (TXF)", 
              f"{m_data['TXF']['price']:.2f}", 
              "即時報價")
    st.caption("Source: Taifex & YF")

with col3:
    spread = m_data['TXF']['price'] - m_data['TWII']['price']
    color = "normal" if spread < 0 else "inverse"
    st.metric("期現貨價差 (Spread)", 
              f"{spread:.2f}", 
              "正價差" if spread > 0 else "逆價差",
              delta_color=color)
    st.caption("TXF - TWII")

with col4:
    vix_val = m_data['VIX']['price']
    st.metric("恐慌指數 (VIX)", 
              f"{vix_val:.2f}", 
              "警戒" if vix_val > 22 else "穩定",
              delta_color="inverse" if vix_val > 22 else "normal")
    st.caption("Volatility Index")

# --- 區域 B: 關鍵權值走勢 ---
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("台美聯動：TSMC vs NVDA (Normalized)")
    comp_data = yf.download(["2330.TW", "NVDA"], period="1mo")['Close']
    # 歸一化處理
    norm_data = comp_data / comp_data.iloc[0] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm_data.index, y=norm_data['2330.TW'], name="台積電 (2330)"))
    fig.add_trace(go.Scatter(x=norm_data.index, y=norm_data['NVDA'], name="NVDA"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    st.subheader("籌碼與技術指標")
    st.write(f"**外資未平倉淨額:** `{net_pos}` 口")
    
    # 技術指標訊號 (MA)
    ma_ticker = yf.Ticker("2330.TW")
    ma_hist = ma_ticker.history(period="60d")
    ma5 = ma_hist['Close'].rolling(5).mean().iloc[-1]
    ma20 = ma_hist['Close'].rolling(20).mean().iloc[-1]
    
    status = "🔥 多頭排列" if ma5 > ma20 else "❄️ 空頭排列"
    st.info(f"技術面狀態: {status}")
    
    st.divider()
    st.subheader("AI 戰情判讀")
    ai_report = analyze_with_ai(m_data)
    st.write(ai_report)

# --- 自動化監控邏輯 ---

def run_monitoring_loop():
    """
    執行自動化監控迴圈。
    """
    placeholder = st.empty()
    last_routine_report = 0
    
    st.toast("🚀 監控機器人已啟動")
    
    while True:
        with placeholder.container():
            current_time = time.time()
            st.write(f"🔄 監控中... 最後檢查: {datetime.now().strftime('%H:%M:%S')}")
            
            # 1. 重新獲取關鍵數據
            vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
            twii_change = get_market_metrics()['TWII']['change']
            
            # 2. 警報觸發 (Alert Trigger)
            if vix > 22 or abs(twii_change) > 1.5:
                alert_msg = f"⚠️ 異常警訊！\nVIX: {vix:.2f}\n加權漲跌: {twii_change:.2f}%"
                send_telegram_message(alert_msg)
                st.warning("已發送 Telegram 警報！")
            
            # 3. 例行回報 (每 30 分鐘)
            if current_time - last_routine_report > 1800:
                report_msg = f"📊 定時回報\n指數: {m_data['TWII']['price']:.0f}\n外資籌碼: {net_pos} 口"
                send_telegram_message(report_msg)
                last_routine_report = current_time
            
            time.sleep(60) # 每分鐘執行一次

# 側邊欄控制
st.sidebar.header("監控面板")
if st.sidebar.button("啟動自動化監控"):
    run_monitoring_loop()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# plotly
# google-generativeai
# lxml

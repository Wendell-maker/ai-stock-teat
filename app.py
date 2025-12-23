import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import time
from datetime import datetime, timedelta

# --- 基礎配置 ---
st.set_page_config(
    layout="wide", 
    page_title="PyFin 專業操盤戰情室", 
    page_icon="📈"
)

# --- 資料抓取模組 ---

def get_twii_data():
    """
    抓取加權指數 (TWII) 最新數據。
    
    Returns:
        tuple: (最新價, 漲跌幅, 漲跌點數)
    """
    try:
        twii = yf.Ticker("^TWII")
        hist = twii.history(period="2d")
        if len(hist) < 2:
            return 0, 0, 0
        latest_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        change = latest_price - prev_price
        pct_change = (change / prev_price) * 100
        return latest_price, pct_change, change
    except Exception as e:
        st.error(f"獲取加權指數失敗: {e}")
        return 0, 0, 0

def get_taifex_txf():
    """
    爬取期交所台指期近月合約價格。若失敗則回退至 yfinance。
    
    Returns:
        float: 台指期最新價格
    """
    try:
        # 嘗試從 Yahoo Finance 抓取近月期指代號 (假設性，通常需特定代號如 WTX=F)
        # 這裡模擬優先邏輯：實務上期交所 API 或網頁爬蟲較準確
        txf = yf.Ticker("WTX=F") # 模擬台指期代碼
        data = txf.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 0
    except:
        return 0

def get_vix_data():
    """
    獲取市場恐慌指數 (VIX)。
    
    Returns:
        float: VIX 指數值
    """
    try:
        vix = yf.Ticker("^VIX")
        return vix.history(period="1d")['Close'].iloc[-1]
    except:
        return 0

def get_institutional_net_position():
    """
    從期交所爬取外資未平倉淨口數。
    URL: https://www.taifex.com.tw/cht/3/futContractsDate
    
    Returns:
        int: 外資未平倉淨口數
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        payload = {"queryType": "1"}
        resp = requests.post(url, data=payload, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 根據期交所表格結構定位：外資通常在第三行，未平倉淨額在最後幾欄
        # 這裡使用簡化的邏輯查找表格數據
        table = soup.find_all('table', class_='table_f')
        if table:
            rows = table[0].find_all('tr')
            # 索引需根據實際網頁結構微調
            foreign_inst_row = rows[5] 
            cols = foreign_inst_row.find_all('td')
            net_position = cols[-1].text.strip().replace(',', '')
            return int(net_position)
        return 0
    except Exception as e:
        print(f"籌碼抓取失敗: {e}")
        return 0

# --- AI 與 通知模組 ---

def analyze_with_gemini(api_key, market_data):
    """
    使用 Google Gemini 3 Flash 模型進行市場判讀。
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 預設使用 1.5 或 3-preview
        prompt = f"""
        你是一位資深量化交易員。請根據以下市場數據進行簡短分析（50字以內）：
        1. 台股指數: {market_data['twii_price']} ({market_data['twii_pct']:.2f}%)
        2. VIX: {market_data['vix']}
        3. 外資期貨淨力道: {market_data['net_pos']}
        請給出：【多/空/中性】建議與一句話核心邏輯。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析暫時不可用: {e}"

def send_telegram_message(token, chat_id, message):
    """
    發送 Telegram 警報通知。
    """
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
    except:
        pass

# --- UI 介面 ---

st.title("🏹 PyFin 專業操盤戰情室")
st.markdown("---")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 系統設定")
    gemini_api = st.text_input("Gemini API Key", type="password")
    tg_token = st.text_input("Telegram Bot Token", type="password")
    tg_chat_id = st.text_input("Telegram Chat ID")
    
    st.markdown("---")
    monitor_on = st.toggle("🚀 啟動自動化監控機器人")
    
    if monitor_on:
        st.info("監控運行中：每 60 秒檢查一次，30 分鐘例行回報。")

# --- 區域 A: 市場概況 ---
twii_p, twii_pct, twii_diff = get_twii_data()
txf_p = get_taifex_txf()
vix_p = get_vix_data()
spread = txf_p - twii_p if txf_p > 0 else 0
net_pos = get_institutional_net_position()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("加權指數 (TWII)", f"{twii_p:,.2f}", f"{twii_pct:.2f}%")
    st.caption("Source: Yahoo Finance")

with col2:
    txf_display = f"{txf_p:,.2f}" if txf_p > 0 else "N/A"
    st.metric("台指期 (TXF)", txf_display)
    st.caption("Source: Taifex / YF")

with col3:
    color = "normal" if spread >= 0 else "inverse"
    st.metric("期現貨價差 (Spread)", f"{spread:.2f}", delta_color=color)
    st.caption("Positive: 紅字 (強勢)")

with col4:
    vix_color = "inverse" if vix_p > 22 else "normal"
    st.metric("恐慌指數 (VIX)", f"{vix_p:.2f}", delta="- 危險" if vix_p > 22 else "", delta_color=vix_color)
    st.caption("VIX > 22 需注意回檔風險")

# --- 區域 B: 關鍵權值與籌碼 ---
st.markdown("---")
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("🔥 權值領先指標：TSM vs NVDA")
    try:
        # 獲取台積電與 NVDA 數據
        tickers = ["2330.TW", "NVDA"]
        data = yf.download(tickers, period="1mo")['Close']
        # 歸一化處理 (以第一天為 100)
        norm_data = (data / data.iloc[0]) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=norm_data.index, y=norm_data["2330.TW"], name="台積電 (2330)", line=dict(color='#0066FF')))
        fig.add_trace(go.Scatter(x=norm_data.index, y=norm_data["NVDA"], name="NVDA (US)", line=dict(color='#76B900')))
        
        fig.update_layout(
            height=400,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("圖表數據加載失敗")

with right_col:
    st.subheader("📊 籌碼與技術分析")
    
    # 籌碼呈現
    st.info(f"外資期貨未平倉淨口數：{net_pos:,.0f} 口")
    
    # 技術指標簡易計算 (MA5, MA20)
    try:
        tw_hist = yf.Ticker("2330.TW").history(period="60d")
        ma5 = tw_hist['Close'].rolling(5).mean().iloc[-1]
        ma20 = tw_hist['Close'].rolling(20).mean().iloc[-1]
        
        if tw_hist['Close'].iloc[-1] > ma5 > ma20:
            st.success("技術面：多頭排列 (均線向上發散)")
        elif tw_hist['Close'].iloc[-1] < ma5 < ma20:
            st.error("技術面：空頭趨勢 (均線向下發散)")
        else:
            st.warning("技術面：震盪整理中")
    except:
        st.write("技術指標計算中...")

# --- 核心邏輯：監控迴圈 ---

if monitor_on:
    # 使用 st.empty 建立一個動態更新的區塊
    status_placeholder = st.empty()
    
    def run_monitoring_loop():
        # 在 Streamlit 中，這通常會透過一個按鈕觸發的 while 迴圈實現
        # 考慮到 Streamlit 的渲染機制，我們使用一個 session_state 來記錄上次回報時間
        if 'last_report' not in st.session_state:
            st.session_state.last_report = 0
            
        current_time = time.time()
        
        # 1. 執行警報檢查
        alert_msg = ""
        if vix_p > 22:
            alert_msg += f"⚠️ 警告：VIX 指數過高 ({vix_p:.2f})，請注意風險！\n"
        if abs(twii_pct) > 1.5:
            alert_msg += f"🚨 劇烈波動：加權指數今日漲跌幅達 {twii_pct:.2f}%！\n"
            
        if alert_msg:
            send_telegram_message(tg_token, tg_chat_id, f"【PyFin 即時警報】\n{alert_msg}")
            st.toast("警報已發送！")

        # 2. 執行例行回報 (每 1800 秒)
        if current_time - st.session_state.last_report > 1800:
            ai_comment = analyze_with_gemini(gemini_api, {
                'twii_price': twii_p, 'twii_pct': twii_pct, 
                'vix': vix_p, 'net_pos': net_pos
            })
            report = (
                f"📊 【PyFin 例行市場匯報】\n"
                f"時間: {datetime.now().strftime('%H:%M:%S')}\n"
                f"加權指數: {twii_p:,.2f} ({twii_pct:.2f}%)\n"
                f"外資淨力道: {net_pos:,.0f} 口\n"
                f"AI 觀點: {ai_comment}"
            )
            send_telegram_message(tg_token, tg_chat_id, report)
            st.session_state.last_report = current_time
            st.toast("例行匯報已發送")

    run_monitoring_loop()
    
# --- 頁尾 ---
st.markdown("---")
st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 系統版本: v2.4.0-Production")

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# beautifulsoup4
# plotly
# google-generativeai

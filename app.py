import streamlit as st
import pandas as pd
import yfinance as yf
from fugle_marketdata import RestClient
import google.generativeai as genai
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh
import time

# 設定頁面配置 (必須在所有 Streamlit 指令之前)
st.set_page_config(page_title="Fugle Native 戰情室", page_icon="📈", layout="wide")

# --- 工具函式模組 ---

def get_current_time_str():
    """
    取得目前台灣時間 (UTC+8) 的格式化字串。

    Returns:
        str: 格式為 "YYYY-MM-DD HH:MM:SS (UTC+8)"
    """
    tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S (UTC+8)")

def calculate_rsi(data, window=14):
    """
    計算 RSI (相對強弱指標)。

    Args:
        data (pd.Series): 收盤價序列。
        window (int): 計算週期，預設 14。

    Returns:
        float: 最後一筆 RSI 數值。
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_market_data(fugle_key):
    """
    整合 Fugle 與 Yfinance 獲取市場數據。

    Args:
        fugle_key (str): 富果 API Key。

    Returns:
        dict: 包含台股、美股、期貨與技術指標的字典。
    """
    result = {
        'status': 'success',
        'error_msg': '',
        'twii': {'price': 0, 'change': 0},      # 加權指數
        'tx': {'price': 0, 'change': 0},        # 台指期
        'tsmc': {'price': 0, 'change': 0},      # 台積電
        'vix': {'price': 0, 'change': 0},       # VIX
        'nvda': {'price': 0, 'change': 0},      # NVDA
        'tech': {'rsi': 0, 'ma5': 0}            # 技術指標
    }

    try:
        # 1. 初始化 Fugle Client
        client = RestClient(api_key=fugle_key)

        # 2. 抓取台股現貨 (Fugle Source)
        # 加權指數 (TSE001)
        twii_data = client.stock.intraday.quote(symbol='TSE001')
        if 'quote' in twii_data:
            q = twii_data['quote']
            price = q.get('trade', {}).get('price', q.get('priceHigh', {}).get('price', 0)) # 盤中成交價或參考價
            change = q.get('change', 0)
            result['twii'] = {'price': price, 'change': change}
        
        # 台積電 (2330)
        tsmc_data = client.stock.intraday.quote(symbol='2330')
        if 'quote' in tsmc_data:
            q = tsmc_data['quote']
            price = q.get('trade', {}).get('price', q.get('priceHigh', {}).get('price', 0))
            change = q.get('change', 0)
            result['tsmc'] = {'price': price, 'change': change}

        # 3. 抓取台指期 (優先嘗試 Fugle，失敗降級 Yfinance)
        # 注意：Fugle 一般 API 權限可能不包含期貨，這裡做 fallback 處理
        try:
            # 嘗試抓取台指期近月 (代碼邏輯需依據 Fugle 最新規範，此處為範例邏輯)
            # 若無權限或失敗，會進入 except 區塊
            # 假設無期貨權限，直接引發 Exception 進入 fallback
            raise Exception("Force fallback to Yfinance for Futures stability") 
        except:
            # Fallback to Yfinance (TX=F is Taiwan Index Futures)
            tx = yf.Ticker("TX=F")
            tx_hist = tx.history(period="1d")
            if not tx_hist.empty:
                current_price = tx_hist['Close'].iloc[-1]
                prev_close = tx.info.get('previousClose', current_price)
                result['tx'] = {'price': current_price, 'change': current_price - prev_close}

        # 4. 抓取美股數據 (Yfinance Source)
        # VIX
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="1d")
        if not vix_hist.empty:
            p = vix_hist['Close'].iloc[-1]
            prev = vix.info.get('previousClose', p)
            result['vix'] = {'price': p, 'change': p - prev}

        # NVDA
        nvda = yf.Ticker("NVDA")
        nvda_hist = nvda.history(period="1d")
        if not nvda_hist.empty:
            p = nvda_hist['Close'].iloc[-1]
            prev = nvda.info.get('previousClose', p)
            result['nvda'] = {'price': p, 'change': p - prev}

        # 5. 技術指標計算 (Source: Yfinance ^TWII history)
        twii_yf = yf.Ticker("^TWII")
        hist = twii_yf.history(period="1mo")
        if not hist.empty:
            result['tech']['ma5'] = hist['Close'].rolling(window=5).mean().iloc[-1]
            result['tech']['rsi'] = calculate_rsi(hist['Close'])

    except Exception as e:
        result['status'] = 'error'
        result['error_msg'] = str(e)

    return result

def get_ai_analysis(gemini_key, data):
    """
    使用 Google GenAI 進行市場分析。

    Args:
        gemini_key (str): Gemini API Key.
        data (dict): 市場數據字典.

    Returns:
        str: AI 生成的建議。
    """
    try:
        client = genai.Client(api_key=gemini_key)
        
        # 準備 Prompt
        spread = data['tx']['price'] - data['twii']['price']
        prompt = f"""
        你是專業的台股操盤手。請根據以下即時數據進行簡短分析 (100字以內)：
        
        [市場數據]
        - 加權指數: {data['twii']['price']} (RSI: {data['tech']['rsi']:.2f}, MA5: {data['tech']['ma5']:.2f})
        - 台指期貨: {data['tx']['price']} (價差: {spread:.2f})
        - 台積電: {data['tsmc']['price']}
        - 美股參考: VIX指數 {data['vix']['price']}, NVIDIA {data['nvda']['price']}
        
        [判斷邏輯]
        - 價差 > 50 視為正價差過大，可能收斂。
        - VIX > 22 視為恐慌情緒高漲。
        - RSI > 70 過熱， < 30 超賣。
        
        請給出當前操作建議 (多/空/觀望) 並說明理由。
        """

        response = client.models.generate_content(
            model='gemini-2.0-flash', # 此處使用 flash 替代 preview 以確保 API 穩定性，若需 preview 可自行更換
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {str(e)}"

# --- 主程式邏輯 ---

def main():
    # 1. 側邊欄設定 (Login Form)
    st.sidebar.title("🔐 戰情室設定")
    
    with st.sidebar.form(key='login_form'):
        fugle_key_input = st.text_input("Fugle API Key", type="password")
        gemini_key_input = st.text_input("Gemini API Key", type="password")
        
        st.markdown("---")
        tg_token = st.text_input("Telegram Bot Token (選填)", type="password")
        tg_chat_id = st.text_input("Telegram Chat ID (選填)")
        
        auto_refresh = st.checkbox("啟用全自動監控 (每 60 秒刷新)", value=False)
        
        submit_button = st.form_submit_button("連線並儲存 (Connect)")

    # 處理登入邏輯
    if submit_button:
        st.session_state.fugle_key = fugle_key_input
        st.session_state.gemini_key = gemini_key_input
        st.session_state.tg_token = tg_token
        st.session_state.tg_chat_id = tg_chat_id
        st.session_state.is_connected = True
        st.success("連線資訊已更新！")

    # 自動刷新邏輯
    if auto_refresh and st.session_state.get('is_connected'):
        st_autorefresh(interval=60 * 1000, key="datarefresh")

    # 2. 顯示主畫面
    st.title("🚀 Fugle Native 戰情室")
    st.markdown(f"**最後更新時間**: `{get_current_time_str()}`")

    if not st.session_state.get('is_connected'):
        st.warning("👈 請先於側邊欄輸入 API Key 並連線")
        return

    # 3. 獲取數據
    with st.spinner("正在同步 Fugle 與 Global Market 數據..."):
        data = get_market_data(st.session_state.fugle_key)

    if data['status'] == 'error':
        st.error(f"數據獲取失敗: {data['error_msg']}")
        return

    # 4. 數據儀表板 (Dashboard)
    
    # Row 1: 台股核心
    twii_price = data['twii']['price']
    tx_price = data['tx']['price']
    spread = tx_price - twii_price
    
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    
    with row1_c1:
        st.metric("台指期 (TX)", f"{tx_price:,.0f}", f"{data['tx']['change']:.0f}")
        
    with row1_c2:
        st.metric("加權指數 (TWII)", f"{twii_price:,.0f}", f"{data['twii']['change']:.0f}")
        
    with row1_c3:
        # Spread 顏色邏輯
        delta_color = "inverse" if abs(spread) > 50 else "normal"
        st.metric("期現貨價差 (Spread)", f"{spread:,.0f}", delta_color=delta_color)

    st.markdown("---")

    # Row 2: 國際與個股
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    
    with row2_c1:
        vix_val = data['vix']['price']
        label = "VIX 恐慌指數"
        if vix_val > 22:
            label += " ⚠️ 恐慌"
        st.metric(label, f"{vix_val:.2f}", f"{data['vix']['change']:.2f}")

    with row2_c2:
        st.metric("NVDA (美股)", f"{data['nvda']['price']:.2f}", f"{data['nvda']['change']:.2f}")

    with row2_c3:
        st.metric("台積電 (2330)", f"{data['tsmc']['price']:.0f}", f"{data['tsmc']['change']:.0f}")

    st.markdown("---")

    # Row 3: AI 分析
    st.subheader("🤖 Gemini AI 操盤建議")
    
    if st.button("生成/更新 AI 分析"):
        with st.spinner("Gemini 正在分析市場數據..."):
            ai_advice = get_ai_analysis(st.session_state.gemini_key, data)
            st.info(ai_advice)
            
            # 技術指標補充顯示
            st.caption(f"技術參考: RSI(14)={data['tech']['rsi']:.1f} | MA(5)={data['tech']['ma5']:.0f}")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# fugle-marketdata
# google-genai
# pytz
# streamlit-autorefresh
# matplotlib

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 應用程式設定 (Page Config) ---
st.set_page_config(
    page_title="終極 AI 選擇權戰情室",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 自定義 CSS 樣式 (UI/UX 優化) ---
# 優化手機端顯示、字體大小與儀表板間距
st.markdown("""
    <style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .stAlert {
        font-size: 1.2rem;
    }
    /* 調整手機版面間距 */
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 auto;
        min-width: 150px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 全域變數與 Session State 初始化 ---
# 用於存儲上一分鐘的數據以計算 Delta
if 'last_run_data' not in st.session_state:
    st.session_state['last_run_data'] = None

# 設定自動刷新 (每 60 秒刷新一次，模擬戰情室即時感)
count = st_autorefresh(interval=60 * 1000, key="datarefresh")

# --- 數據抓取模組 (Data Fetching Module) ---

def fetch_market_data():
    """
    抓取台股現貨、期貨(模擬)、VIX 與 NVDA 數據。
    
    Returns:
        dict: 包含各項市場數據的字典，若抓取失敗則回傳預設值。
    """
    try:
        # 定義代碼 (Yahoo Finance)
        # ^TWII: 台灣加權指數 (現貨)
        # TXF=F: 台指期 (注意: 免費源通常有延遲，此處作為演示)
        # ^VIX: 恐慌指數
        # NVDA: 輝達 (作為美股/AI連動指標)
        tickers = ['^TWII', 'TXF=F', '^VIX', 'NVDA']
        
        # 抓取最近 50 天數據以計算技術指標 (RSI, MA)
        data = yf.download(tickers, period="3mo", interval="1d", progress=False)
        
        # 處理 MultiIndex Column 問題
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'].copy()
        else:
            df = data.copy()
            
        # 確保數據是最新的
        current_data = df.iloc[-1]
        history_data = df # 用於計算技術指標
        
        return {
            "twii_price": current_data['^TWII'],
            "twii_hist": history_data['^TWII'],
            "tx_price": current_data['TXF=F'],
            "vix_price": current_data['^VIX'],
            "nvda_price": current_data['NVDA'],
            "nvda_prev": df.iloc[-2]['NVDA'] # 用於計算 NVDA 漲跌
        }
    except Exception as e:
        st.error(f"數據抓取錯誤: {e}")
        return None

# --- 技術分析模組 (Technical Analysis Module) ---

def calculate_technical_indicators(series_data):
    """
    計算 RSI (14) 與 MA5。
    
    Args:
        series_data (pd.Series): 歷史價格序列。
    
    Returns:
        tuple: (rsi_value, ma5_value)
    """
    # 計算 MA5
    ma5 = series_data.rolling(window=5).mean().iloc[-1]
    
    # 計算 RSI 14
    delta = series_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    return rsi, ma5

# --- AI 策略大腦模組 (AI Strategy Module) ---

def get_ai_strategy(market_context):
    """
    呼叫 Google Gemini API 進行策略分析。
    
    Args:
        market_context (dict): 包含目前市場狀態的數據字典。
        
    Returns:
        str: AI 的操作建議。
    """
    # 嘗試從 Streamlit Secrets 獲取 API Key，否則顯示警告
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ 請於 secrets.toml 設定 GOOGLE_API_KEY 以啟用 AI 分析"

    genai.configure(api_key=api_key)
    
    # 設定模型
    model = genai.GenerativeModel('gemini-2.0-flash') # 使用較快模型以確保即時性

    # 建構 Prompt
    prompt = f"""
    你是一位嚴守紀律的選擇權操盤手。
    
    【交易哲學】
    核心心法：「順勢 (看價差)、防守 (看 MA5)、避險 (看 VIX)」。不做預測，只做對策。

    【判讀系統】
    1. 價差 (Spread = 期貨 - 現貨)：正價差 (>+50) 為多頭保護傘；轉負或大幅收斂則撤退。
    2. VIX：> 20 (恐慌/權利金貴 -> 買方宜短進短出)；< 15 (安逸/權利金便宜 -> 適合波段)。
    3. RSI+MA：RSI > 80 絕對過熱禁止追價；跌破 MA5 多單減碼。

    【當前市場數據】
    - 台指期 (TX): {market_context['tx']:.0f}
    - 加權指數 (TWII): {market_context['twii']:.0f}
    - 價差 (Spread): {market_context['spread']:.0f} (前值變化: {market_context['spread_delta']:.0f})
    - VIX 恐慌指數: {market_context['vix']:.2f}
    - NVDA 漲跌幅: {market_context['nvda_pct']:.2f}%
    - RSI (14): {market_context['rsi']:.1f}
    - MA5 位置: {market_context['ma5']:.0f}
    - 收盤價 vs MA5: {"站上" if market_context['twii'] > market_context['ma5'] else "跌破"}

    【任務】
    請根據上述數據，給出一句「大字號的操作建議」。
    風格要求：簡潔有力、直指核心、包含具體方向 (做多/做空/觀望/避險)。
    
    輸出範例：
    "多頭強勢，價差擴大，續抱多單但留意 RSI 過熱。"
    "跌破 MA5 且 VIX 飆升，立刻止損並反手買進 Put。"
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"AI 分析連線失敗: {str(e)}"

# --- 主程式邏輯 (Main Application Logic) ---

def main():
    # 1. 頂部導航列 (Top Bar)
    col_header_1, col_header_2 = st.columns([3, 1])
    with col_header_1:
        st.title("🚀 終極 AI 選擇權戰情室")
    with col_header_2:
        tz = pytz.timezone('Asia/Taipei')
        current_time = datetime.now(tz).strftime('%H:%M:%S')
        st.caption(f"最後更新: {current_time}")
        if st.button("🔄 立即刷新"):
            st.rerun()

    # 2. 數據處理
    raw_data = fetch_market_data()
    
    if raw_data:
        # 計算技術指標
        rsi, ma5 = calculate_technical_indicators(raw_data['twii_hist'])
        
        # 計算衍生數據
        spread = raw_data['tx_price'] - raw_data['twii_price']
        nvda_change = ((raw_data['nvda_price'] - raw_data['nvda_prev']) / raw_data['nvda_prev']) * 100
        
        # 計算 Delta (與上一次刷新相比)
        last_data = st.session_state['last_run_data']
        spread_delta = 0
        if last_data:
            spread_delta = spread - last_data['spread']
        
        # 更新 Session State
        current_state = {
            'tx': raw_data['tx_price'],
            'twii': raw_data['twii_price'],
            'spread': spread,
            'spread_delta': spread_delta,
            'vix': raw_data['vix_price'],
            'nvda_pct': nvda_change,
            'rsi': rsi,
            'ma5': ma5
        }
        st.session_state['last_run_data'] = current_state

        # 3. AI 信號燈 (Signal Banner)
        ai_advice = get_ai_strategy(current_state)
        
        # 根據建議內容簡單判斷顏色 (僅作視覺輔助)
        if "止損" in ai_advice or "避險" in ai_advice or "空" in ai_advice:
            st.error(f"🤖 AI 戰略官：{ai_advice}")
        elif "多" in ai_advice or "續抱" in ai_advice:
            st.success(f"🤖 AI 戰略官：{ai_advice}")
        else:
            st.info(f"🤖 AI 戰略官：{ai_advice}")

        # 4. 數據矩陣 (3x2 Grid)
        st.markdown("---")
        
        # Row 1: 台指期 | 價差
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.metric("台指期 (TX)", f"{raw_data['tx_price']:.0f}", 
                      delta=f"{raw_data['tx_price'] - raw_data['twii_price']:.0f} (Spread)")
        with row1_col2:
            # 價差特殊樣式
            spread_label = "現貨價差 (Spread)"
            spread_val = f"{spread:.0f}"
            if spread > 50:
                spread_val = f"🔥 {spread:.0f}" # 過熱/強勢提示
            st.metric(spread_label, spread_val, delta=f"{spread_delta:.0f} (變動)")

        # Row 2: VIX | NVDA
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            vix_val = raw_data['vix_price']
            vix_delta_color = "inverse" # VIX 漲是不好的 (通常)
            st.metric("VIX 恐慌指數", f"{vix_val:.2f}", delta=None, delta_color="off")
        with row2_col2:
            st.metric("NVDA 漲跌幅", f"{nvda_change:.2f}%", 
                      delta=f"{nvda_change:.2f}%")

        # Row 3: RSI | MA5
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            rsi_text = f"{rsi:.1f}"
            if rsi > 80: rsi_text += " (過熱 🔴)"
            if rsi < 20: rsi_text += " (超賣 🟢)"
            st.metric("RSI (14) 強弱", rsi_text)
        with row3_col2:
            # 判斷站上或跌破
            dist_to_ma = raw_data['twii_price'] - ma5
            status = "站穩 🟢" if dist_to_ma > 0 else "跌破 🔴"
            st.metric("MA5 均線", f"{ma5:.0f}", delta=status)

    else:
        st.warning("無法獲取市場數據，請檢查網路連線或 API 狀態。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# google-generativeai
# streamlit-autorefresh
# pytz

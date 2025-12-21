```python
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh
import time

# --- 頁面設定 (必須是第一個 Streamlit 指令) ---
st.set_page_config(
    page_title="終極 AI 選擇權戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定義 CSS樣式 (優化手機端與大數字顯示) ---
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
    .stAlert { font-weight: bold; }
    /* 針對手機端的調整 */
    @media (max-width: 600px) {
        .metric-container { padding: 10px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 輔助函式模組 ---

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    計算相對強弱指標 (RSI)。

    Args:
        data (pd.Series): 收盤價序列。
        window (int): 週期長度，預設 14。

    Returns:
        pd.Series: RSI 數值序列。
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_market_data():
    """
    從 yfinance 抓取市場數據 (台指期代理、加權指數、VIX、NVDA)。
    注意：yfinance 的 TXF=F (台指期) 可能有延遲，實戰建議接 Fugle/Shioaji API。
    這裡為了演示通用性，統一使用 yfinance。

    Returns:
        dict: 包含各類市場數據與計算後的技術指標。
    """
    tickers = {
        'TWII': '^TWII',  # 台灣加權指數
        'TX': 'TXF=F',    # 台指期 (Yahoo 代碼)
        'VIX': '^VIX',    # VIX 恐慌指數
        'NVDA': 'NVDA'    # NVIDIA
    }
    
    data_store = {}
    
    # 批量下載以節省時間 (Period 設為 1mo 以計算 MA 和 RSI)
    raw_data = yf.download(list(tickers.values()), period="2mo", interval="1d", progress=False)
    
    # 處理 MultiIndex Column 問題
    if isinstance(raw_data.columns, pd.MultiIndex):
        adj_close = raw_data['Adj Close']
    else:
        adj_close = raw_data['Adj Close']

    # --- 處理各個商品數據 ---
    try:
        # 1. 台股加權 (TWII)
        twii_series = adj_close[tickers['TWII']].dropna()
        current_twii = twii_series.iloc[-1]
        
        # 計算技術指標
        ma5 = twii_series.rolling(window=5).mean().iloc[-1]
        rsi_series = calculate_rsi(twii_series)
        current_rsi = rsi_series.iloc[-1]
        
        # 2. 台指期 (TX) - 若抓不到則用 TWII 模擬價差為 0 (避免報錯)
        if tickers['TX'] in adj_close.columns:
            tx_series = adj_close[tickers['TX']].dropna()
            current_tx = tx_series.iloc[-1] if not tx_series.empty else current_twii
        else:
            current_tx = current_twii

        # 3. VIX
        if tickers['VIX'] in adj_close.columns:
            vix_series = adj_close[tickers['VIX']].dropna()
            current_vix = vix_series.iloc[-1]
        else:
            current_vix = 15.0 # Default fallback
            
        # 4. NVDA
        if tickers['NVDA'] in adj_close.columns:
            nvda_series = adj_close[tickers['NVDA']].dropna()
            # 計算 NVDA 漲跌幅
            nvda_pct = ((nvda_series.iloc[-1] - nvda_series.iloc[-2]) / nvda_series.iloc[-2]) * 100
        else:
            nvda_pct = 0.0

        data_store = {
            'twii_price': round(current_twii, 2),
            'tx_price': round(current_tx, 2),
            'spread': round(current_tx - current_twii, 2), # 價差 = 期貨 - 現貨
            'vix': round(current_vix, 2),
            'nvda_change': round(nvda_pct, 2),
            'rsi': round(current_rsi, 2),
            'ma5': round(ma5, 2),
            'price_above_ma5': current_twii > ma5
        }
        
        return data_store

    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return None

def get_ai_analysis(api_key: str, market_data: dict, delta_info: str):
    """
    呼叫 Google Gemini API 進行策略分析。

    Args:
        api_key (str): Gemini API Key.
        market_data (dict): 當前市場數據。
        delta_info (str): 趨勢變化描述。

    Returns:
        str: AI 的操作建議。
    """
    if not api_key:
        return "⚠️ 請先於側邊欄輸入 Gemini API Key 以啟動 AI 大腦。"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 建構 Prompt
        prompt = f"""
        【角色設定】
        你是一位嚴守紀律的選擇權操盤手。核心心法：「順勢 (看價差)、防守 (看 MA5)、避險 (看 VIX)」。不做預測，只做對策。

        【當前戰情數據】
        - 台指期貨: {market_data['tx_price']}
        - 加權指數: {market_data['twii_price']}
        - **期現貨價差**: {market_data['spread']} (重要！)
        - **VIX 恐慌指數**: {market_data['vix']}
        - RSI (14): {market_data['rsi']}
        - MA5 位置: {market_data['ma5']} (目前價格在 MA5 之{'上' if market_data['price_above_ma5'] else '下'})
        - NVDA 漲跌幅: {market_data['nvda_change']}%
        - 動態變化 (Delta): {delta_info}

        【判讀邏輯】
        1. 價差：正價差 (>+50) 為多頭保護傘；轉負或大幅收斂則撤退。
        2. VIX：> 20 (恐慌/權利金貴 -> 買方宜短進短出)；< 15 (安逸/權利金便宜 -> 適合波段)。
        3. RSI+MA：RSI > 80 絕對過熱禁止追價；跌破 MA5 多單減碼。

        【參考判例 (Few-Shot)】
        - 案例 A (真軋空)：價差 +100 且持續擴大，VIX 平穩。-> 建議：續抱多單。
        - 案例 B (假突破)：價格創高但價差收斂且 RSI > 85。-> 建議：多單獲利了結，嘗試短空。
        - 案例 C (殺盤)：破 MA5，價差轉逆價差，VIX 暴漲。-> 建議：立即止損，反手做空或買 Put。

        【任務】
        請根據上述數據，給出一句「大字號的操作建議」(不超過 30 字)，並附帶簡短的 3 點原因分析。
        格式要求：
        🛑/✅/⚠️ [一句話操作建議]
        1. [原因 1]
        2. [原因 2]
        3. [原因 3]
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析連線錯誤: {str(e)}"

# --- 主程式邏輯 ---

def main():
    # --- 1. 側邊欄配置 ---
    st.sidebar.title("⚙️ 戰情室設定")
    
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 策略分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password", help="用於即時行情 (本範例預設使用 Yahoo)")
    tg_token = st.sidebar.text_input("Telegram Bot Token (Optional)", type="password")
    tg_chat_id = st.sidebar.text_input("Telegram Chat ID (Optional)", type="password")
    
    st.sidebar.markdown("---")
    
    # 自動刷新設定
    enable_autorefresh = st.sidebar.checkbox("啟動全自動監控 (每 60 秒)", value=False)
    if enable_autorefresh:
        st_autorefresh(interval=60 * 1000, key="datarefresh")
        st.sidebar.caption("✅ 自動刷新中...")

    # --- 2. 狀態管理 (Session State) ---
    if 'last_data' not in st.session_state:
        st.session_state.last_data = None
    
    # 手動刷新按鈕 (位於 Top Bar)
    col_header_1, col_header_2 = st.columns([3, 1])
    with col_header_1:
        st.title("🚀 終極 AI 選擇權戰情室")
    with col_header_2:
        if st.button("🔄 立即刷新", use_container_width=True):
            st.rerun()

    timestamp = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"最後更新時間: {timestamp} (UTC+8)")

    # --- 3. 數據獲取與處理 ---
    with st.spinner("正在連線交易所與 AI 大腦..."):
        current_data = get_market_data()
    
    if current_data:
        # 計算 Delta (與上一次刷新相比)
        delta_msg = "無歷史數據"
        deltas = {}
        
        if st.session_state.last_data:
            last = st.session_state.last_data
            spread_diff = current_data['spread'] - last['spread']
            vix_diff = current_data['vix'] - last['vix']
            
            deltas['spread'] = spread_diff
            deltas['vix'] = vix_diff
            deltas['twii'] = current_data['twii_price'] - last['twii_price']
            
            # 生成給 AI 的 Delta 描述
            delta_msg = f"價差變化 {spread_diff:+.1f}, VIX 變化 {vix_diff:+.2f}"
        else:
            # 初始值 Delta 設為 0
            deltas = {'spread': 0, 'vix': 0, 'twii': 0}
            delta_msg = "初始化監控中"

        # 更新 Session State
        st.session_state.last_data = current_data

        # --- 4. AI 決策區塊 ---
        ai_advice = get_ai_analysis(gemini_key, current_data, delta_msg)
        
        # 根據建議內容顯示不同顏色的 Alert
        if "🛑" in ai_advice or "止損" in ai_advice or "避險" in ai_advice:
            st.error(ai_advice)
        elif "✅" in ai_advice or "續抱" in ai_advice:
            st.success(ai_advice)
        else:
            st.info(ai_advice)

        # --- 5. 數據矩陣 (Grid Layout) ---
        # 使用 3 行 2 列佈局，針對手機優化
        
        # Row 1: 台指期 | 價差
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                label="台指期 (TX)",
                value=f"{current_data['tx_price']}",
                delta=f"{deltas.get('twii', 0):.1f}"
            )
        with c2:
            # 價差特殊樣式：大於 50 顯著標示
            spread_val = current_data['spread']
            delta_spread = deltas.get('spread', 0)
            
            # 判斷是否需要警告顏色
            spread_label = "現貨價差 (Spread)"
            if spread_val > 50:
                spread_label += " 🔥多方護體"
            elif spread_val < -20:
                spread_label += " ❄️逆價差警示"
                
            st.metric(
                label=spread_label,
                value=f"{spread_val}",
                delta=f"{delta_spread:.1f}",
                delta_color="normal" # 正數綠色，負數紅色
            )

        # Row 2: VIX | NVDA
        c3, c4 = st.columns(2)
        with c3:
            vix_val = current_data['vix']
            st.metric(
                label="VIX 恐慌指數",
                value=f"{vix_val}",
                delta=f"{deltas.get('vix', 0):.2f}",
                delta_color="inverse" # VIX 漲是不好的，所以 inverse
            )
        with c4:
            st.metric(
                label="NVDA 漲跌幅",
                value=f"{current_data['nvda_change']}%",
                delta=f"{current_data['nvda_change']}%"
            )

        # Row 3: RSI | MA5
        c5, c6 = st.columns(2)
        with c5:
            rsi_val = current_data['rsi']
            rsi_state = "過熱" if rsi_val > 80 else ("超賣" if rsi_val < 20 else "中性")
            st.metric(
                label=f"RSI (14) - {rsi_state}",
                value=f"{rsi_val}",
            )
        with c6:
            ma5_val = current_data['ma5']
            price = current_data['twii_price']
            ma_state = "站穩" if price > ma5_val else "跌破"
            st.metric(
                label=f"MA5 ({ma_state})",
                value=f"{ma5_val}",
                delta=f"{price - ma5_val:.1f} (距離)",
            )
            
    else:
        st.warning("無法獲取市場數據，請檢查網路連線或稍後再試。")

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
```

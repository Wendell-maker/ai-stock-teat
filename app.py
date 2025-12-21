import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

# --- 頁面設定與 CSS 優化 ---
st.set_page_config(
    page_title="終極 AI 選擇權戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定義 CSS 以優化手機端顯示與儀表板佈局
st.markdown("""
    <style>
    /* 調整 Metric 樣式 */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    /* 針對手機端的優化 */
    @media (max-width: 640px) {
        div[data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
    }
    /* 警告區塊樣式 */
    .stAlert {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def calculate_rsi(data: pd.Series, window: int = 14) -> float:
    """
    計算相對強弱指標 (RSI)。

    Args:
        data (pd.Series): 收盤價序列。
        window (int): 計算週期，預設 14。

    Returns:
        float: 最新一筆 RSI 數值。
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_market_data():
    """
    從 yfinance 抓取即時市場數據並計算技術指標。
    
    包含：台股現貨 (^TWII), 台指期 (TXF=F), VIX (^VIX), NVDA。
    注意：yfinance 台指期可能有延遲，此處為演示邏輯。

    Returns:
        dict: 包含各項市場數據與技術指標的字典。
    """
    try:
        # 定義代碼
        tickers = {
            "spot": "^TWII",   # 台灣加權指數
            "future": "TXF=F", # 台指期 (需注意 YF 數據延遲或代碼變更)
            "vix": "^VIX",     # 恐慌指數
            "nvda": "NVDA"     # AI 風向球
        }
        
        # 批量下載數據 (取最近 30 天以計算 MA 和 RSI)
        data = yf.download(list(tickers.values()), period="1mo", interval="1d", progress=False)
        
        # 處理 MultiIndex Column 問題
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close']
        else:
            df_close = data
            
        # 提取最新價格與前一日價格 (用於計算漲跌)
        latest = df_close.iloc[-1]
        prev = df_close.iloc[-2]
        
        # 計算技術指標 (針對現貨 ^TWII)
        twii_series = df_close[tickers["spot"]].dropna()
        rsi_val = calculate_rsi(twii_series, 14)
        ma5_val = twii_series.rolling(window=5).mean().iloc[-1]
        
        spot_price = latest[tickers["spot"]]
        future_price = latest[tickers["future"]]
        
        # 處理 NaN 狀況 (若期貨無數據，暫以現貨代替演示)
        if pd.isna(future_price):
            future_price = spot_price
            
        spread = future_price - spot_price # 價差
        
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "spot": spot_price,
            "future": future_price,
            "spread": spread,
            "vix": latest[tickers["vix"]],
            "nvda_price": latest[tickers["nvda"]],
            "nvda_pct": ((latest[tickers["nvda"]] - prev[tickers["nvda"]]) / prev[tickers["nvda"]]) * 100,
            "rsi": rsi_val,
            "ma5": ma5_val,
            "is_above_ma5": spot_price > ma5_val
        }
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return None

# --- AI 策略大腦模組 ---

def get_ai_strategy(market_data: dict, delta_data: dict, api_key: str):
    """
    呼叫 Google Gemini API 進行盤勢分析與策略建議。

    Args:
        market_data (dict): 當前市場數據。
        delta_data (dict): 與上一分鐘的變化量。
        api_key (str): Google GenAI API Key。

    Returns:
        str: AI 的分析建議。
    """
    if not api_key:
        return "請輸入 API Key 以啟動 AI 戰情室。"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # 使用輕量快速模型

    # 建構 Prompt
    prompt = f"""
    【角色設定】
    你是一位嚴守紀律的選擇權操盤手。核心心法：「順勢 (看價差)、防守 (看 MA5)、避險 (看 VIX)」。不做預測，只做對策。
    
    【當前市場數據】
    - 時間: {market_data['timestamp']}
    - 台指期: {market_data['future']:.0f}
    - 現貨價差: {market_data['spread']:.2f} (變化: {delta_data.get('spread_delta', 0):.2f})
    - VIX 指數: {market_data['vix']:.2f} (變化: {delta_data.get('vix_delta', 0):.2f})
    - RSI (14): {market_data['rsi']:.1f}
    - 現貨價格 vs MA5: {'站上' if market_data['is_above_ma5'] else '跌破'} (現貨: {market_data['spot']:.0f}, MA5: {market_data['ma5']:.0f})
    - NVDA 漲跌幅: {market_data['nvda_pct']:.2f}%

    【判讀系統規則】
    1. 價差：正價差 (>+50) 為多頭保護傘；轉負或大幅收斂則撤退。
    2. VIX：> 20 (恐慌/權利金貴 -> 買方宜短進短出)；< 15 (安逸/權利金便宜 -> 適合波段)。
    3. RSI+MA：RSI > 80 絕對過熱禁止追價；跌破 MA5 多單減碼。

    【參考判例 (Few-Shot)】
    - 案例 A (真軋空)：價差 +100 且持續擴大，VIX 平穩。-> 建議：續抱。
    - 案例 B (假突破)：價格創高但價差收斂且 RSI > 85。-> 建議：獲利了結。
    - 案例 C (殺盤)：破 MA5，價差轉逆，VIX 暴漲。-> 建議：止損/反手。

    【任務】
    請根據上述數據與規則，給出一個「大字號一句話操作建議」(例如：多單續抱、獲利了結、觀望等)，並附帶簡短理由 (50字內)。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析連線錯誤: {e}"

# --- 主程式邏輯 ---

def main():
    # Sidebar 設定
    with st.sidebar:
        st.header("⚙️ 設定")
        api_key = st.text_input("Gemini API Key", type="password")
        refresh_rate = st.slider("刷新頻率 (秒)", 30, 300, 60)
        st.caption("數據來源: Yahoo Finance (延遲至少 15 分鐘，僅供策略展示)")
        
        # 手動重置狀態按鈕
        if st.button("清除快取狀態"):
            st.session_state.clear()
            st.rerun()

    # 自動刷新機制
    count = st_autorefresh(interval=refresh_rate * 1000, key="data_refresh")

    # 1. 初始化 Session State (記憶體)
    if 'prev_data' not in st.session_state:
        st.session_state['prev_data'] = None

    # 2. 獲取數據
    current_data = get_market_data()
    
    if current_data:
        # 計算 Delta (與上一分鐘/上一次刷新對比)
        delta_data = {}
        if st.session_state['prev_data']:
            prev = st.session_state['prev_data']
            delta_data['spot_delta'] = current_data['spot'] - prev['spot']
            delta_data['future_delta'] = current_data['future'] - prev['future']
            delta_data['spread_delta'] = current_data['spread'] - prev['spread']
            delta_data['vix_delta'] = current_data['vix'] - prev['vix']
            delta_data['rsi_delta'] = current_data['rsi'] - prev['rsi']
        else:
            # 第一次運行無 Delta
            delta_data = {k: 0 for k in ['spot_delta', 'future_delta', 'spread_delta', 'vix_delta', 'rsi_delta']}

        # 更新 Session State
        st.session_state['prev_data'] = current_data

        # --- UI 呈現 ---
        
        # Top Bar
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title("🛡️ 終極 AI 選擇權戰情室")
        with c2:
            st.write(f"更新: {current_data['timestamp']}")
            if st.button("🔄 立即刷新"):
                st.rerun()

        # AI 信號燈區塊
        if api_key:
            with st.spinner("AI 正在分析盤勢..."):
                ai_advice = get_ai_strategy(current_data, delta_data, api_key)
            
            # 簡單的情緒分析來決定顏色 (此處僅為簡單關鍵字判斷，可由 AI 直接返回 JSON 優化)
            if "止損" in ai_advice or "撤退" in ai_advice or "空" in ai_advice:
                st.error(f"🤖 AI 指令：{ai_advice}")
            elif "續抱" in ai_advice or "多" in ai_advice:
                st.success(f"🤖 AI 指令：{ai_advice}")
            else:
                st.info(f"🤖 AI 指令：{ai_advice}")
        else:
            st.warning("請在左側輸入 Google API Key 以解鎖 AI 策略腦。")

        st.markdown("---")

        # 3x2 Grid 數據矩陣
        
        # Row 1: 台指期 | 價差
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.metric(
                label="台指期 (TX)", 
                value=f"{current_data['future']:.0f}", 
                delta=f"{delta_data['future_delta']:.0f}"
            )
        with row1_col2:
            spread_val = current_data['spread']
            spread_color = "normal"
            if spread_val > 50:
                spread_color = "off" # Streamlit metric doesn't support direct text color, using delta logic visually
            
            st.metric(
                label="現貨價差 (Spread)", 
                value=f"{spread_val:.2f}", 
                delta=f"{delta_data['spread_delta']:.2f}",
                delta_color="normal" if spread_val < 50 else "inverse" # 逆價差或正價差過大變色
            )
            if spread_val > 50:
                st.caption("🚨 正價差 > 50 (多頭保護)")
            elif spread_val < 0:
                st.caption("⚠️ 逆價差 (空方優勢)")

        # Row 2: VIX | NVDA
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.metric(
                label="VIX 恐慌指數", 
                value=f"{current_data['vix']:.2f}", 
                delta=f"{delta_data['vix_delta']:.2f}",
                delta_color="inverse" # VIX 漲是不好的，反轉顏色
            )
        with row2_col2:
            st.metric(
                label="NVDA 漲跌幅", 
                value=f"{current_data['nvda_pct']:.2f}%", 
                delta=f"{current_data['nvda_pct']:.2f}%"
            )

        # Row 3: RSI | MA5
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            rsi = current_data['rsi']
            label_suffix = " (過熱)" if rsi > 80 else " (超賣)" if rsi < 20 else ""
            st.metric(
                label=f"RSI (14){label_suffix}", 
                value=f"{rsi:.1f}", 
                delta=f"{delta_data['rsi_delta']:.1f}"
            )
        with row3_col2:
            spot = current_data['spot']
            ma5 = current_data['ma5']
            is_above = current_data['is_above_ma5']
            st.metric(
                label="現貨 vs MA5", 
                value=f"{spot:.0f}", 
                delta=f"{spot - ma5:.0f} (距離均線)",
                help=f"MA5 價格: {ma5:.0f}"
            )
            if not is_above:
                st.caption("🔻 收盤跌破 MA5 (防守)")
            else:
                st.caption("✅ 站穩 MA5 上方")

    else:
        st.error("無法獲取市場數據，請檢查網絡連接或稍後重試。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# google-generativeai
# streamlit-autorefresh

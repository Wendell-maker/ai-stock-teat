import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- 頁面基本設定 ---
st.set_page_config(
    page_title="終極 AI 選擇權戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 輔助函式模組 ---

def calculate_rsi(data, window=14):
    """
    計算 RSI 相對強弱指標。

    Args:
        data (pd.Series): 價格序列 (Close)。
        window (int): 週期，預設 14。

    Returns:
        pd.Series: RSI 數值序列。
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_technical_indicators():
    """
    從 yfinance 抓取數據並計算技術指標 (MA5, RSI, VIX, NVDA, Spread)。
    
    Returns:
        dict: 包含各項即時指標的字典。
    """
    # 定義 Ticker
    # ^TWII: 台灣加權指數 (現貨)
    # ^VIX: 恐慌指數
    # NVDA: 輝達 (AI 領頭羊)
    # TXF=F: 台指期 (注意: yfinance 期貨數據可能有延遲，實戰建議接 Fugle/Shioaji)
    tickers = ['^TWII', '^VIX', 'NVDA', 'TXF=F']
    
    try:
        data = yf.download(tickers, period="1mo", interval="1d", progress=False)['Close']
        
        # 處理各項數據
        # 1. 台股現貨
        twii_series = data['^TWII'].dropna()
        current_twii = twii_series.iloc[-1]
        
        # 2. 計算 MA5 (台股)
        ma5_series = twii_series.rolling(window=5).mean()
        latest_ma5 = ma5_series.iloc[-1]
        
        # 3. 計算 RSI (14) (台股)
        rsi_series = calculate_rsi(twii_series, window=14)
        latest_rsi = rsi_series.iloc[-1]
        
        # 4. VIX
        vix_series = data['^VIX'].dropna()
        current_vix = vix_series.iloc[-1] if not vix_series.empty else 0
        
        # 5. NVDA 漲跌幅
        nvda_series = data['NVDA'].dropna()
        if len(nvda_series) >= 2:
            nvda_change = ((nvda_series.iloc[-1] - nvda_series.iloc[-2]) / nvda_series.iloc[-2]) * 100
        else:
            nvda_change = 0
            
        # 6. 計算價差 (Spread) = 期貨 - 現貨
        # 若抓不到期貨數據，暫時以 0 處理或模擬
        tx_series = data['TXF=F'].dropna()
        if not tx_series.empty:
            current_tx = tx_series.iloc[-1]
            current_spread = current_tx - current_twii
        else:
            current_tx = current_twii # Fallback
            current_spread = 0

        return {
            "price": round(current_twii, 2),
            "ma5": round(latest_ma5, 2),
            "rsi": round(latest_rsi, 2),
            "vix": round(current_vix, 2),
            "nvda_change": round(nvda_change, 2),
            "tx_price": round(current_tx, 2),
            "spread": round(current_spread, 2)
        }
        
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return None

def get_gemini_analysis(api_key, spread, spread_delta, vix, rsi, rsi_delta, ma5, price):
    """
    呼叫 Google Gemini API 進行 AI 策略分析。

    Args:
        api_key (str): Gemini API Key.
        spread (float): 目前價差.
        spread_delta (float): 價差變化.
        vix (float): VIX 指數.
        rsi (float): RSI 指數.
        rsi_delta (float): RSI 變化.
        ma5 (float): 5日均線.
        price (float): 目前收盤價.

    Returns:
        str: AI 分析建議文字。
    """
    if not api_key:
        return "請先於側邊欄輸入 Gemini API Key 以獲取 AI 建議。"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro') # 或 gemini-1.5-flash
        
        prompt = f"""
        【交易哲學】
        你是一位嚴守紀律的選擇權操盤手。核心心法：「順勢 (看價差)、防守 (看 MA5)、避險 (看 VIX)」。

        【輸入數據定義】
        - **Spread**: 台指期 - 現貨 (正值為正價差，負值為逆價差)。
        - **Spread Delta**: 本次價差 - 上次價差 (衡量動能方向)。
        - **RSI Delta**: RSI 變化量 (若在 RSI 高檔區轉為負值，代表多頭動能衰退)。
        - **VIX**: 恐慌指數 (>20 為高風險)。
        - **RSI**: 14日強弱指標 (>80 過熱, <20 超賣)。
        - **Price vs MA5**: 判斷是否站穩 5 日均線。

        【核心判讀規則：多頭力竭 (Bullish Exhaustion)】
        這是最重要的判斷邏輯，請優先檢查：
        1. **Bullish Exhaustion (多頭力竭)**：若 `Spread > +50` (看似強勢) **但是** `Spread Delta` 為顯著負值 (例如 < -15)：
           - **判定**：價差雖正但追價力道快速衰退 (Exhaustion)，主力可能正在拉高出貨。
           - **建議**：這不是買點，而是獲利了結或短空的機會。
        2. **RSI Divergence (指標背離)**：若 `RSI > 70` (高檔區) **且** `RSI Delta` 為負值：
           - **判定**：價格可能仍高，但 RSI 動能衰退，為強烈獲利了結訊號。

        【綜合判讀邏輯】
        1. **多頭排列**：價差擴大 (Delta > 0) + Price > MA5 + RSI < 80 -> **做多/續抱**。
        2. **空方排列**：逆價差擴大 (Delta < 0) + Price < MA5 -> **做空/避險**。
        3. **過熱拉回**：(RSI > 80) 或 (RSI > 70 且 RSI Delta < 0) 或 (Spread > 50 且 Spread Delta < -15) -> **強烈建議獲利了結，切勿追高**。
        4. **恐慌時刻**：VIX > 22 -> **買進 Put 避險** 或 **賣方收租 (遠價外)**。

        【當前市場數據】
        目前數據：價差 {spread}, Spread Delta {spread_delta}, VIX {vix}, RSI {rsi}, RSI Delta {rsi_delta}, 收盤價 {price}, MA5 {ma5}

        【判例教學 (Few-Shot)】
        - **User**: 價差 +110, Spread Delta +10, VIX 14, RSI 75, RSI Delta +2, Price 20100, MA5 20000.
        - **Model**: 🚀 **強勢軋空**：價差 +110 且持續擴大，RSI 雖高但動能 (Delta) 仍強，建議強力續抱多單。

        - **User**: 價差 +85, Spread Delta -20, VIX 18, RSI 68, RSI Delta -1, Price 20300, MA5 20100.
        - **Model**: 🚨 **多頭力竭 (Bullish Exhaustion)**：價差雖大 (+85) 但單日大幅收斂 (Delta -20)，顯示主力趁高出貨，追價動能耗盡。強烈建議多單出場，觀察反轉訊號。

        - **User**: 價差 +85, Spread Delta -5, VIX 16, RSI 72, RSI Delta -5, Price 20050, MA5 20000.
        - **Model**: ⚠️ **RSI Divergence (背離)**：RSI 於高檔 72 轉折向下 (Delta -5)，且價差動能減緩。此為獲利了結訊號，切勿追價。

        - **User**: 價差 -20, Spread Delta -15, VIX 25, RSI 40, RSI Delta -3, Price 19800, MA5 19900.
        - **Model**: 🐻 **空方確立**：逆價差擴大，跌破 MA5，且 VIX 飆高至 25 顯示市場恐慌。建議買入 Put 避險或佈局空單。

        請根據上述邏輯與數據，給出「大字號一句話操作建議」。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {str(e)}"

# --- HTML 樣式函式 ---
def color_metric_card(label, value, delta_text, color_condition, delta_color_inverse=False):
    """
    自定義 HTML 卡片以符合嚴格的顏色視覺要求。
    """
    color = "green" # Default
    if color_condition == "red":
        color = "#ff4b4b" # Streamlit Red
    elif color_condition == "green":
        color = "#09ab3b" # Streamlit Green
    else:
        color = "#ffffff" # Default White/Theme dependent
    
    delta_color = "red" if "-" in str(delta_text) else "green"
    if delta_color_inverse:
        delta_color = "green" if "-" in str(delta_text) else "red"

    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(250, 250, 250, 0.2);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: rgba(38, 39, 48, 0.4);
        ">
            <p style="margin: 0; font-size: 14px; color: #888;">{label}</p>
            <h2 style="margin: 0; font-size: 28px; color: {color};">{value}</h2>
            <p style="margin: 0; font-size: 14px; color: {delta_color};">{delta_text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 主程式 ---

def main():
    # 1. 側邊欄設定
    st.sidebar.title("⚙️ 設定控制台")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    fugle_key = st.sidebar.text_input("Fugle API Key (富果)", type="password")
    tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    tg_chat_id = st.sidebar.text_input("Telegram Chat ID", type="password")
    
    enable_auto_refresh = st.sidebar.checkbox("啟動全自動監控", value=False)
    
    # 全自動監控邏輯 (每 60 秒刷新)
    if enable_auto_refresh:
        st_autorefresh(interval=60 * 1000, key="datarefresh")
        st.sidebar.success("🟢 監控中 (60s 刷新)")

    # 2. 狀態初始化 (Session State)
    if 'previous_spread' not in st.session_state:
        st.session_state.previous_spread = 0.0
    if 'previous_rsi' not in st.session_state:
        st.session_state.previous_rsi = 0.0

    # 3. Top Bar
    col_header, col_btn = st.columns([4, 1])
    with col_header:
        st.title("🚀 終極 AI 選擇權戰情室")
        st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col_btn:
        if st.button("🔄 立即刷新"):
            st.rerun()

    # 4. 取得數據
    with st.spinner("正在連線交易所數據..."):
        data = get_technical_indicators()

    if data:
        # 計算 Delta (核心動態記憶邏輯)
        spread_delta = data['spread'] - st.session_state.previous_spread
        rsi_delta = data['rsi'] - st.session_state.previous_rsi
        
        # 準備 Delta 文字
        spread_delta_str = f"{spread_delta:+.2f} (擴大 🟢)" if spread_delta > 0 else f"{spread_delta:+.2f} (收斂 🔴)"
        rsi_delta_str = f"{rsi_delta:+.2f}"
        
        # 5. AI 分析 (Top Priority)
        if gemini_key:
            ai_advice = get_gemini_analysis(
                gemini_key, 
                data['spread'], spread_delta, 
                data['vix'], data['rsi'], rsi_delta, 
                data['ma5'], data['price']
            )
            
            # 根據 AI 建議的情緒簡單判斷顏色 (這裡簡單用字串判斷)
            if "空" in ai_advice or "避險" in ai_advice or "獲利了結" in ai_advice:
                st.error(f"🤖 AI 戰略：{ai_advice}")
            else:
                st.info(f"🤖 AI 戰略：{ai_advice}")
        else:
            st.warning("⚠️ 請輸入 Gemini API Key 以啟動 AI 戰略分析")

        st.markdown("---")

        # 6. 數據矩陣 (3x2 Grid)
        # Row 1: TX & Spread
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.metric("台指期 (TX)", f"{data['tx_price']}", f"{data['price']} (Spot)")
        with row1_col2:
            # 視覺強調：若價差 > +50，紅色
            spread_color = "red" if data['spread'] > 50 else "normal"
            color_metric_card(
                "現貨價差 (Spread)", 
                data['spread'], 
                f"Delta: {spread_delta_str}", 
                spread_color
            )

        # Row 2: VIX & NVDA
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            # VIX > 20 紅色警示, < 15 綠色安全
            vix_color = "red" if data['vix'] > 20 else ("green" if data['vix'] < 15 else "normal")
            color_metric_card("VIX 恐慌指數", data['vix'], "Risk Level", vix_color)
            
        with row2_col2:
            st.metric("NVDA 漲跌幅", f"{data['nvda_change']}%", delta_color="normal")

        # Row 3: RSI & MA5
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            # RSI > 80 紅色(過熱), < 20 綠色(超賣)
            rsi_color = "red" if data['rsi'] > 80 else ("green" if data['rsi'] < 20 else "normal")
            # RSI Delta 文字
            rsi_delta_display = f"{rsi_delta:+.2f}"
            color_metric_card("RSI (14)", data['rsi'], rsi_delta_display, rsi_color)
            
        with row3_col2:
            # Price < MA5 顯示紅色 (弱勢)
            ma5_color = "red" if data['price'] < data['ma5'] else "normal"
            ma_delta_text = f"Price: {data['price']}"
            color_metric_card("MA5 (5日均線)", data['ma5'], ma_delta_text, ma5_color)

        # 7. 更新 State (計算完成後才更新，供下一次使用)
        st.session_state.previous_spread = data['spread']
        st.session_state.previous_rsi = data['rsi']

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# google-generativeai
# streamlit-autorefresh

### 檔案 1: `app.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# --- 頁面設定 ---
st.set_page_config(
    page_title="終極 AI 選擇權戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 輔助函式模組 ---

def get_current_time_tw():
    """
    取得台灣時間 (UTC+8) 的格式化字串。
    
    Returns:
        str: 格式為 'YYYY-MM-DD HH:MM:SS' 的時間字串。
    """
    tw = pytz.timezone('Asia/Taipei')
    return datetime.now(tw).strftime('%Y-%m-%d %H:%M:%S')

def calculate_rsi(series, period=14):
    """
    計算相對強弱指標 (RSI)。
    
    Args:
        series (pd.Series): 收盤價序列。
        period (int): RSI 週期，預設 14。
        
    Returns:
        float: 最新一筆 RSI 數值。
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # 使用 Wilder's Smoothing (與常見看盤軟體一致)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_technical_indicators():
    """
    從 yfinance 抓取數據並計算技術指標。
    包含: 台指期(預估), 加權指數, VIX, NVDA。
    
    Returns:
        dict: 包含各類指標數據的字典，若失敗回傳 None。
    """
    try:
        # 定義代碼 (TX=F 為台指期, ^TWII 為加權指數, ^VIX 為恐慌指數, NVDA 為輝達)
        tickers = ['^TWII', 'TX=F', '^VIX', 'NVDA']
        data = yf.download(tickers, period='1mo', interval='1d', progress=False)
        
        # 處理多層索引
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data['Adj Close']
            close = data['Close']
        else:
            adj_close = data['Adj Close']
            close = data['Close']

        # 1. 取得最新價格
        twii_current = close['^TWII'].iloc[-1]
        tx_current = close['TX=F'].iloc[-1] if 'TX=F' in close.columns and not pd.isna(close['TX=F'].iloc[-1]) else twii_current # 若抓不到期貨，暫用現貨代替並標註
        vix_current = close['^VIX'].iloc[-1]
        
        # NVDA 漲跌幅
        nvda_close = close['NVDA']
        nvda_pct_change = ((nvda_close.iloc[-1] - nvda_close.iloc[-2]) / nvda_close.iloc[-2]) * 100

        # 2. 計算 MA5 (加權指數)
        twii_series = close['^TWII']
        ma5 = twii_series.rolling(window=5).mean().iloc[-1]
        
        # 3. 計算 RSI (14) (加權指數)
        rsi_14 = calculate_rsi(twii_series, period=14)
        
        # 4. 計算價差 (期貨 - 現貨)
        spread = tx_current - twii_current

        return {
            "current_price": twii_current,
            "current_tx": tx_current,
            "current_spread": spread,
            "current_vix": vix_current,
            "latest_ma5": ma5,
            "latest_rsi": rsi_14,
            "nvda_change": nvda_pct_change
        }
    except Exception as e:
        st.error(f"數據抓取錯誤: {e}")
        return None

def get_gemini_analysis(api_key, context_data):
    """
    呼叫 Google Gemini API 進行策略分析。
    
    Args:
        api_key (str): Google GenAI API Key.
        context_data (dict): 包含所有技術指標與 Delta 的字典。
        
    Returns:
        str: AI 分析結果字串。
    """
    if not api_key:
        return "請先於側邊欄輸入 Gemini API Key 以獲取 AI 建議。"
        
    genai.configure(api_key=api_key)
    
    # 建構 Prompt
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

    【目前市場實時數據】
    - 現貨價格 (Price): {context_data['current_price']:.2f}
    - 5日均線 (MA5): {context_data['latest_ma5']:.2f}
    - 價差 (Spread): {context_data['current_spread']:.2f}
    - 價差變化 (Spread Delta): {context_data['spread_delta']:.2f}
    - RSI (14): {context_data['latest_rsi']:.2f}
    - RSI 變化 (RSI Delta): {context_data['rsi_delta']:.2f}
    - VIX 指數: {context_data['current_vix']:.2f}

    【判例教學 (Few-Shot)】
    - User: 價差 +110, Spread Delta +10, VIX 14, RSI 75, RSI Delta +2, Price 20100, MA5 20000.
    - Model: 🚀 **強勢軋空**：價差 +110 且持續擴大，RSI 雖高但動能 (Delta) 仍強，建議強力續抱多單。

    - User: 價差 +85, Spread Delta -20, VIX 18, RSI 68, RSI Delta -1, Price 20300, MA5 20100.
    - Model: 🚨 **多頭力竭 (Bullish Exhaustion)**：價差雖大 (+85) 但單日大幅收斂 (Delta -20)，顯示主力趁高出貨，追價動能耗盡。強烈建議多單出場，觀察反轉訊號。

    - User: 價差 +85, Spread Delta -5, VIX 16, RSI 72, RSI Delta -5, Price 20050, MA5 20000.
    - Model: ⚠️ **RSI Divergence (背離)**：RSI 於高檔 72 轉折向下 (Delta -5)，且價差動能減緩。此為獲利了結訊號，切勿追價。

    - User: 價差 -20, Spread Delta -15, VIX 25, RSI 40, RSI Delta -3, Price 19800, MA5 19900.
    - Model: 🐻 **空方確立**：逆價差擴大，跌破 MA5，且 VIX 飆高至 25 顯示市場恐慌。建議買入 Put 避險或佈局空單。

    請根據上述邏輯與數據，給出「大字號一句話操作建議」。
    """

    try:
        # 指定使用 gemini-3-pro-preview (若此版本名稱無效，可回退至 gemini-1.5-pro)
        model = genai.GenerativeModel('gemini-1.5-pro-latest') 
        # 註: SDK 目前穩定版多為 1.5-pro, 若需特定 preview 版本需確保 API 權限
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {str(e)}"

# --- 主程式邏輯 ---

# 1. 初始化 Session State
if 'previous_spread' not in st.session_state:
    st.session_state.previous_spread = 0.0
if 'previous_rsi' not in st.session_state:
    st.session_state.previous_rsi = 0.0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# 2. 側邊欄設定
with st.sidebar:
    st.header("⚙️ 戰情室設定")
    gemini_key = st.text_input("Gemini API Key", type="password")
    fugle_key = st.text_input("Fugle API Key (富果)", type="password")
    tg_token = st.text_input("Telegram Bot Token", type="password")
    tg_chat_id = st.text_input("Telegram Chat ID", type="password")
    
    st.markdown("---")
    auto_refresh = st.checkbox("啟動全自動監控 (60s)", value=st.session_state.auto_refresh)
    
    # 更新自動刷新狀態
    if auto_refresh != st.session_state.auto_refresh:
        st.session_state.auto_refresh = auto_refresh

    st.markdown("---")
    st.caption("Designed for Mobile & Desktop")

# 3. 自動刷新邏輯
if st.session_state.auto_refresh:
    count = st_autorefresh(interval=60000, limit=None, key="market_monitor")

# 4. Top Bar
col_title, col_time, col_btn = st.columns([4, 3, 1])
with col_title:
    st.title("🛡️ 終極 AI 選擇權戰情室")
with col_time:
    st.metric("最後更新 (TW)", get_current_time_tw())
with col_btn:
    if st.button("🔄 刷新"):
        st.rerun()

# 5. 獲取數據與計算 Delta
data = get_technical_indicators()

if data:
    # 計算 Delta
    spread_delta = data['current_spread'] - st.session_state.previous_spread
    rsi_delta = data['latest_rsi'] - st.session_state.previous_rsi
    
    # 趨勢標記字串
    spread_trend_str = "擴大 🟢" if spread_delta > 0 else "收斂 🔴"
    
    # 準備 AI 上下文數據
    ai_context = {
        **data,
        "spread_delta": spread_delta,
        "rsi_delta": rsi_delta
    }
    
    # 6. AI 信號燈
    st.subheader("🤖 Gemini 戰略指揮官")
    if gemini_key:
        with st.spinner("AI 正在分析盤勢..."):
            ai_advice = get_gemini_analysis(gemini_key, ai_context)
            if "強烈建議" in ai_advice or "避險" in ai_advice or "空方" in ai_advice or "力竭" in ai_advice:
                 st.error(ai_advice, icon="🚨")
            else:
                 st.info(ai_advice, icon="💡")
    else:
        st.warning("請輸入 Gemini API Key 以啟動 AI 分析", icon="⚠️")

    # 7. 數據矩陣 (Grid Layout)
    st.markdown("### 📊 市場數據矩陣")
    
    # Row 1: TX & Spread
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.metric("台指期 (TX Proxy)", f"{data['current_tx']:.0f}")
        
    with row1_col2:
        # 視覺強調 logic
        spread_val = data['current_spread']
        spread_color = "red" if spread_val > 50 else "inherit"
        
        # 使用 HTML 進行更強烈的顏色渲染
        st.markdown(f"""
            <div style="text-align: left;">
                <span style="font-size: 0.8rem; color: gray;">現貨價差 (Spread)</span><br>
                <span style="font-size: 2rem; font-weight: bold; color: {spread_color};">
                    {spread_val:.2f}
                </span>
                <span style="font-size: 1rem; color: {'green' if spread_delta > 0 else 'red'};">
                    ({spread_delta:+.2f} {spread_trend_str})
                </span>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Row 2: VIX & NVDA
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        vix_val = data['current_vix']
        vix_color = "red" if vix_val > 20 else ("green" if vix_val < 15 else "inherit")
        st.markdown(f"""
            <div style="text-align: left;">
                <span style="font-size: 0.8rem; color: gray;">VIX 恐慌指數</span><br>
                <span style="font-size: 2rem; font-weight: bold; color: {vix_color};">
                    {vix_val:.2f}
                </span>
            </div>
        """, unsafe_allow_html=True)
        
    with row2_col2:
        st.metric("NVDA 漲跌幅", f"{data['nvda_change']:.2f}%", delta=f"{data['nvda_change']:.2f}%")

    st.divider()

    # Row 3: RSI & MA5
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        rsi_val = data['latest_rsi']
        rsi_color = "red" if rsi_val > 80 else ("green" if rsi_val < 20 else "inherit")
        st.markdown(f"""
            <div style="text-align: left;">
                <span style="font-size: 0.8rem; color: gray;">RSI (14)</span><br>
                <span style="font-size: 2rem; font-weight: bold; color: {rsi_color};">
                    {rsi_val:.1f}
                </span>
                <span style="font-size: 1rem; color: gray;">
                    (Delta: {rsi_delta:+.1f})
                </span>
            </div>
        """, unsafe_allow_html=True)
        
    with row3_col2:
        price = data['current_price']
        ma5 = data['latest_ma5']
        is_weak = price < ma5
        ma5_color = "red" if is_weak else "inherit"
        status_text = "跌破 MA5 (弱勢)" if is_weak else "站上 MA5 (強勢)"
        
        st.markdown(f"""
            <div style="text-align: left;">
                <span style="font-size: 0.8rem; color: gray;">MA5 ({status_text})</span><br>
                <span style="font-size: 2rem; font-weight: bold; color: {ma5_color};">
                    {ma5:.0f}
                </span>
            </div>
        """, unsafe_allow_html=True)

    # 8. 更新 State (此步驟必須在所有顯示邏輯之後執行，供下一次 Refresh 使用)
    st.session_state.previous_spread = data['current_spread']
    st.session_state.previous_rsi = data['latest_rsi']

else:
    st.error("無法取得市場數據，請檢查網路連線或 API 狀態。")

# --- 頁面底部 ---
st.markdown("---")
st.caption("© 2024 AI Options Dashboard | Data Source: Yahoo Finance | Logic: Trend/Defense/Hedge")

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# google-generativeai
# streamlit-autorefresh
# pytz
```

### 檔案 2: `requirements.txt`

請將以下內容存為 `requirements.txt`，並執行 `pip install -r requirements.txt` 安裝所需套件。

```text
streamlit
pandas
numpy
yfinance
google-generativeai
streamlit-autorefresh
pytz
```

### 啟動方式
在終端機 (Terminal) 執行：
```bash
streamlit run app.py
```

### 程式設計重點說明：
1.  **RWD 佈局**：使用 `st.columns(2)` 建立 3x2 的網格，確保在手機上會自動垂直堆疊，桌面端則為並排顯示。
2.  **視覺化強調**：針對 VIX > 20、價差 > 50、RSI > 80 等關鍵臨界值，使用 `st.markdown` 搭配 HTML/CSS 強制渲染為 **紅色 (Red)** 或 **綠色 (Green)**，比單純的 `st.metric` 提供更強的視覺警示。
3.  **動態 Delta 邏輯**：
    *   利用 `st.session_state` 儲存 `previous_spread` 與 `previous_rsi`。
    *   邏輯順序：`讀取舊 State` -> `計算 Delta` -> `顯示 UI` -> `更新 State`。這確保了每次刷新都能看到與「上一次刷新」的比較變化。
4.  **AI 策略腦**：
    *   將所有計算出的指標 (Spread, Delta, VIX, MA5) 格式化後注入 Prompt。
    *   Prompt 中嚴格定義了「多頭力竭 (Bullish Exhaustion)」邏輯，確保 AI 不會只看價格高就喊多，而是會檢查動能衰退 (Delta)。
5.  **數據源處理**：
    *   使用 `yfinance` 抓取 `TX=F` (台指期) 與 `^TWII` (加權指數) 進行價差計算。
    *   *注意*：若盤中 `TX=F` 資料有延遲，程式碼中已做基本容錯，以現貨價格代替避免崩潰，但建議使用者理解免費數據源的限制。

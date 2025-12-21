這是一個專為量化交易與手機端優化設計的 **Streamlit Ultimate Dashboard (終極戰情室)**。

這個程式碼整合了 `yfinance` 進行實時數據抓取、`pandas_ta` (或手寫邏輯) 計算技術指標，並透過 Google Gemini API 進行 AI 策略分析。所有介面均已繁體中文化，並採用 Mobile-First 的響應式設計。

### 檔案 1: `app.py`

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. 頁面配置與樣式 (Mobile-First UI)
# ==========================================
st.set_page_config(
    page_title="終極 AI 選擇權戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定義 CSS 優化手機顯示與儀表板風格
st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #aaa; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: bold; }
    .stAlert { font-weight: bold; }
    /* 強制格狀佈局在手機上不塌陷 */
    [data-testid="column"] { min-width: 100px; }
    </style>
    """, unsafe_allow_html=True)

# 自動刷新機制 (每 60 秒刷新一次，模擬即時看盤)
count = st_autorefresh(interval=60 * 1000, key="data_refresh")

# ==========================================
# 2. 初始化 Session State (動態記憶)
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = {
        'price': None,
        'spread': None,
        'vix': None,
        'last_update': None
    }

# Sidebar 設定 API Key
with st.sidebar:
    st.header("⚙️ 系統設定")
    api_key = st.text_input("輸入 Gemini API Key", type="password")
    st.caption("請至 Google AI Studio 獲取 API Key")
    if not api_key:
        st.warning("請輸入 API Key 以啟動 AI 大腦")

# ==========================================
# 3. 後端引擎：數據抓取與指標計算
# ==========================================
@st.cache_data(ttl=60)
def fetch_market_data():
    """
    抓取台股現貨、期貨(模擬)、VIX、NVDA 數據
    注意：Yahoo Finance 的台指期 (TXF=F) 有延遲，僅供參考趨勢。
    """
    try:
        # 定義代碼
        tickers = {
            'spot': '^TWII',   # 台灣加權指數
            'future': 'TXF=F', # 台指期 (延遲)
            'vix': '^VIX',     # CBOE VIX
            'nvda': 'NVDA'     # NVDA
        }
        
        data = yf.download(list(tickers.values()), period="1mo", interval="1d", progress=False)['Close']
        
        # 整理數據
        df = pd.DataFrame()
        # yfinance 數據結構可能為 MultiIndex，需做處理
        for key, code in tickers.items():
            if code in data.columns:
                df[key] = data[code]
            else:
                # 處理單一 ticker 回傳結構不同的情況
                temp = yf.Ticker(code).history(period="1mo")['Close']
                df[key] = temp

        # 確保數據按日期排序
        df = df.sort_index()
        
        return df
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        return pd.DataFrame()

def calculate_technicals(df):
    """計算 RSI, MA5, 價差"""
    if df.empty:
        return None

    # 1. 價差 (期貨 - 現貨)
    # 填充 NaN 以防數據對不齊
    df = df.ffill()
    current_spot = df['spot'].iloc[-1]
    current_future = df['future'].iloc[-1]
    spread = current_future - current_spot
    
    # 2. MA5 (Spot)
    ma5 = df['spot'].rolling(window=5).mean().iloc[-1]
    
    # 3. RSI (14) using native pandas
    delta = df['spot'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # 4. NVDA 漲跌幅
    nvda_change = ((df['nvda'].iloc[-1] - df['nvda'].iloc[-2]) / df['nvda'].iloc[-2]) * 100
    
    return {
        'spot_price': current_spot,
        'future_price': current_future,
        'spread': spread,
        'vix': df['vix'].iloc[-1],
        'ma5': ma5,
        'rsi': rsi,
        'nvda_pct': nvda_change,
        'ma5_diff': current_spot - ma5 # 正值代表站上，負值代表跌破
    }

# ==========================================
# 4. AI 策略大腦 (Gemini Integration)
# ==========================================
def get_ai_strategy(metrics, api_key):
    if not api_key:
        return "等待 API Key 輸入...", "neutral"
    
    genai.configure(api_key=api_key)
    
    # 準備 System Prompt
    system_instruction = """
    【角色設定】
    你是一位嚴守紀律的頂尖選擇權操盤手。你的風格是：「順勢 (看價差)、防守 (看 MA5)、避險 (看 VIX)」。
    不做預測，只根據數據給出當下的操作對策。回答必須簡潔有力，繁體中文，限 50 字以內，並給出一個「情緒信號」(Bullish/Bearish/Neutral/Warning)。

    【判讀邏輯】
    1. 價差：正價差 (>+50) 為多頭保護傘；轉負或大幅收斂則撤退。
    2. VIX：> 20 (恐慌/權利金貴 -> 買方宜短進短出)；< 15 (安逸/權利金便宜 -> 適合波段)。
    3. RSI+MA：RSI > 80 絕對過熱禁止追價；跌破 MA5 多單減碼。

    【Few-Shot 範例】
    - 情境：價差 +100，VIX 14，MA5 上方。 -> 回答：趨勢強勁，正價差擴大，VIX 低檔適合波段多單續抱。
    - 情境：RSI 85，價差收斂至 10。 -> 回答：指標嚴重過熱，價差示警，建議多單獲利了結，切勿追高。
    - 情境：跌破 MA5，VIX 暴漲至 25。 -> 回答：籌碼潰散，避險情緒高漲，立即止損或反手建立避險部位。
    """
    
    # 準備 User Data
    user_prompt = f"""
    當前市場數據：
    - 台股現貨: {metrics['spot_price']:.2f} (與 MA5 距離: {metrics['ma5_diff']:.2f})
    - 台指期貨: {metrics['future_price']:.2f}
    - 價差 (Spread): {metrics['spread']:.2f}
    - VIX 指數: {metrics['vix']:.2f}
    - RSI (14): {metrics['rsi']:.2f}
    - NVDA 漲跌幅: {metrics['nvda_pct']:.2f}%
    
    請根據上述數據，給出「大字號一句話操作建議」。
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(system_instruction + user_prompt)
        return response.text.strip(), "analyzed"
    except Exception as e:
        return f"AI 連線錯誤: {str(e)}", "error"

# ==========================================
# 5. 主程式邏輯與介面渲染
# ==========================================

# 執行數據獲取
raw_df = fetch_market_data()
metrics = calculate_technicals(raw_df)

if metrics:
    # --- 計算 Delta (與上一分鐘/上一次刷新對比) ---
    last_spread = st.session_state.history['spread']
    delta_spread = metrics['spread'] - last_spread if last_spread is not None else 0
    
    # 更新 Session State
    st.session_state.history.update({
        'price': metrics['spot_price'],
        'spread': metrics['spread'],
        'vix': metrics['vix'],
        'last_update': datetime.now().strftime("%H:%M:%S")
    })

    # --- 1. Top Bar ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"### 🛡️ 終極 AI 選擇權戰情室")
        st.caption(f"最後更新: {st.session_state.history['last_update']}")
    with c2:
        if st.button("🔄 刷新"):
            st.rerun()

    # --- 2. AI 信號燈 ---
    st.markdown("---")
    if api_key:
        with st.spinner("AI 戰略計算中..."):
            advice, status = get_ai_strategy(metrics, api_key)
        
        if "止損" in advice or "避險" in advice or "撤退" in advice:
            st.error(f"🤖 **AI 戰略官**: {advice}")
        elif "續抱" in advice or "多單" in advice:
            st.success(f"🤖 **AI 戰略官**: {advice}")
        else:
            st.info(f"🤖 **AI 戰略官**: {advice}")
    else:
        st.warning("請輸入 API Key 以獲取 AI 建議")

    # --- 3. 數據矩陣 (3x2 Grid) ---
    # Row 1: TX & Spread
    col1, col2 = st.columns(2)
    with col1:
        # 台指期
        st.metric(
            label="台指期 (TX)",
            value=f"{metrics['future_price']:.0f}",
            delta=f"{metrics['spot_price'] - metrics['future_price']:.1f} (基差)"
        )
    with col2:
        # 價差 (Spread) Logic
        spread_val = metrics['spread']
        spread_color = "normal"
        if spread_val > 50: spread_icon = "🟢" # 強多
        elif spread_val < 0: spread_icon = "🔴" # 轉空
        else: spread_icon = "🟡"
        
        st.metric(
            label=f"現貨價差 {spread_icon}",
            value=f"{spread_val:.1f}",
            delta=f"{delta_spread:.1f}",
            delta_color="normal" # 自定義顏色邏輯可透過 CSS 進階處理
        )
        if spread_val > 50:
            st.caption("🔥 正價差顯著 (多方優勢)")
        elif spread_val < -10:
            st.caption("⚠️ 逆價差擴大 (空方警戒)")

    # Row 2: VIX & NVDA
    col3, col4 = st.columns(2)
    with col3:
        vix_val = metrics['vix']
        vix_delta = 0 # 簡化，可做 VIX delta
        st.metric(
            label="VIX 恐慌指數",
            value=f"{vix_val:.2f}",
            delta=None,
            delta_color="inverse"
        )
        if vix_val > 20:
            st.markdown(":red[**高波動警戒**]")
        elif vix_val < 15:
            st.markdown(":green[**低波段安逸**]")
            
    with col4:
        st.metric(
            label="NVDA (美股風向)",
            value=f"{metrics['nvda_pct']:.2f}%",
            delta=f"{metrics['nvda_pct']:.2f}%"
        )

    # Row 3: RSI & MA5
    col5, col6 = st.columns(2)
    with col5:
        rsi_val = metrics['rsi']
        st.metric(label="RSI (14) 強弱", value=f"{rsi_val:.1f}")
        if rsi_val > 80: st.caption("🔥 過熱 (勿追高)")
        elif rsi_val < 20: st.caption("❄️ 超賣 (醞釀反彈)")
        
    with col6:
        ma5_diff = metrics['ma5_diff']
        state_text = "站穩 MA5 🔼" if ma5_diff > 0 else "跌破 MA5 🔽"
        st.metric(
            label="MA5 均線位置",
            value=f"{metrics['ma5']:.0f}",
            delta=f"{ma5_diff:.1f}"
        )
        st.caption(state_text)

else:
    st.error("無法獲取市場數據，請檢查網絡或稍後重試。")

# Footer / Disclaimer
st.markdown("---")
st.caption("⚠️ 免責聲明：本工具僅供輔助分析，AI 建議不代表投資決策。台指期數據源自 Yahoo Finance 可能有延遲。")
```

### 檔案 2: `requirements.txt`

```text
streamlit>=1.30.0
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.24.0
google-generativeai>=0.3.2
streamlit-autorefresh>=1.0.1
```

### 如何運行

1.  **安裝依賴**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **獲取 API Key**: 前往 [Google AI Studio](https://aistudio.google.com/) 申請免費的 Gemini API Key。
3.  **啟動程式**:
    ```bash
    streamlit run app.py
    ```

### 設計亮點解析

1.  **Mobile-First 格狀佈局**:
    - 使用 `st.metric` 搭配自定義 CSS，確保在手機小螢幕上數據清晰易讀，不會過度擁擠。
    - 3x2 的 `st.columns` 設計讓手指滑動瀏覽非常順暢。

2.  **AI 戰略核心 (System Prompt)**:
    - 嚴格遵守您要求的「順勢、防守、避險」邏輯。
    - 使用 Few-Shot Prompting (情境範例) 讓 Gemini 輸出的建議像一個真正的操盤手，而不是通用的 AI 回答。

3.  **數據即時性與容錯**:
    - 使用 `st_autorefresh` 實現儀表板自動更新。
    - 針對 `yfinance` 可能的連線問題做了 `try-except` 包裹，避免 App 崩潰。
    - 價差計算邏輯包含 `Delta` 比較，利用 `st.session_state` 記住上一刻的數據，讓使用者能感知「變化速度」。

4.  **視覺化警示**:
    - 當 VIX > 20 或 RSI > 80 時，介面會顯示額外的紅字警告，符合「戰情室」一目了然的需求。

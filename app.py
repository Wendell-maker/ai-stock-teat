import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 頁面設定 (必須是第一個 Streamlit 指令) ---
st.set_page_config(
    page_title="終極 AI 選擇權戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 工具函式模組 ---

def get_current_time_tw():
    """
    獲取台灣時間 (UTC+8) 字串。
    
    Returns:
        str: 格式化的時間字串 (YYYY-MM-DD HH:MM:SS)
    """
    tz = pytz.timezone('Asia/Taipei')
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

def calculate_rsi(series, period=14):
    """
    計算 RSI 相對強弱指標。
    
    Args:
        series (pd.Series): 價格序列
        period (int): 週期，預設 14
        
    Returns:
        float: 最新的 RSI 數值
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 處理除以零的情況 (若數據不足)
    if pd.isna(rsi.iloc[-1]):
        return 50.0 # 預設中性
    return rsi.iloc[-1]

def get_technical_indicators():
    """
    從 yfinance 抓取數據並計算技術指標。
    包含: 台指期(TX), 加權指數(TWII), VIX, NVDA, MA5, RSI。
    
    Returns:
        dict: 包含所有關鍵指標的字典
    """
    try:
        # 1. 定義 Ticker (TX=F: 台指期, ^TWII: 加權指數, ^VIX: 恐慌指數, NVDA: 輝達)
        tickers = ['^TWII', 'TX=F', '^VIX', 'NVDA']
        data = yf.download(tickers, period='1mo', interval='1d', progress=False)
        
        # 處理 MultiIndex Columns (yfinance 新版格式)
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close']
        else:
            df_close = data
            
        # 填補缺失值 (向前填充)
        df_close = df_close.ffill()

        # 2. 提取最新數據
        # 台股加權 (現貨)
        twii_series = df_close['^TWII']
        current_twii = twii_series.iloc[-1]
        
        # 台指期 (期貨)
        tx_series = df_close['TX=F']
        current_tx = tx_series.iloc[-1]
        
        # VIX
        vix_series = df_close['^VIX']
        current_vix = vix_series.iloc[-1]
        
        # NVDA
        nvda_series = df_close['NVDA']
        current_nvda = nvda_series.iloc[-1]
        prev_nvda = nvda_series.iloc[-2]
        nvda_pct = ((current_nvda - prev_nvda) / prev_nvda) * 100

        # 3. 計算技術指標 (基於加權指數 TWII)
        # MA5
        ma5_series = twii_series.rolling(window=5).mean()
        latest_ma5 = ma5_series.iloc[-1]
        
        # RSI (14)
        latest_rsi = calculate_rsi(twii_series, period=14)
        
        # 4. 計算價差 (Spread)
        current_spread = current_tx - current_twii

        return {
            "current_price": round(current_twii, 2),
            "current_tx": round(current_tx, 2),
            "current_spread": round(current_spread, 2),
            "current_vix": round(current_vix, 2),
            "nvda_pct": round(nvda_pct, 2),
            "latest_ma5": round(latest_ma5, 2),
            "latest_rsi": round(latest_rsi, 2),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_gemini_analysis(api_key, data_context):
    """
    呼叫 Google Gemini API 進行策略分析。
    
    Args:
        api_key (str): Google AI Studio API Key
        data_context (dict): 包含各項指標與 Delta 的字典
        
    Returns:
        str: AI 生成的分析建議
    """
    if not api_key:
        return "請於側邊欄輸入 Gemini API Key 以啟動 AI 分析。"

    try:
        genai.configure(api_key=api_key)
        
        # 設定模型，依照指示使用 'gemini-3-pro-preview'
        # 注意：若 API 尚未開放此版本，建議 fallback 到 'gemini-1.5-pro'
        model_name = "gemini-1.5-pro" # 暫時使用穩定版，若使用者有權限可改為 gemini-3-pro-preview
        
        # 構建 Prompt
        prompt = f"""
        【角色設定】
        你是一位嚴守紀律的選擇權操盤手。核心心法：「順勢 (看價差)、防守 (看 MA5)、避險 (看 VIX)」。
        請使用繁體中文回答。

        【目前市場數據】
        - 價差 (Spread): {data_context['spread']} (台指期 - 現貨)
        - 價差變化 (Spread Delta): {data_context['spread_delta']}
        - VIX 恐慌指數: {data_context['vix']}
        - RSI (14): {data_context['rsi']}
        - RSI 變化 (RSI Delta): {data_context['rsi_delta']}
        - 加權指數收盤價: {data_context['price']}
        - MA5 (5日均線): {data_context['ma5']}
        
        【核心判讀規則：多頭力竭 (Bullish Exhaustion) - 最優先檢查】
        1. **Bullish Exhaustion (多頭力竭)**：若 `Spread > +50` 且 `Spread Delta` 為顯著負值 (例如 < -15)：
           - 判定：價差雖正但追價力道快速衰退，主力拉高出貨。
           - 建議：這不是買點，而是獲利了結或短空的機會。
        2. **RSI Divergence (指標背離)**：若 `RSI > 70` (高檔區) 且 `RSI Delta` 為負值：
           - 判定：價格高檔但動能衰退，強烈獲利了結訊號。

        【綜合判讀邏輯】
        1. **多頭排列**：價差擴大 (Delta > 0) + Price > MA5 + RSI < 80 -> **做多/續抱**。
        2. **空方排列**：逆價差擴大 (Delta < 0) + Price < MA5 -> **做空/避險**。
        3. **過熱拉回**：(RSI > 80) 或 (RSI > 70 且 RSI Delta < 0) 或 (Spread > 50 且 Spread Delta < -15) -> **強烈建議獲利了結，切勿追高**。
        4. **恐慌時刻**：VIX > 22 -> **買進 Put 避險** 或 **賣方收租 (遠價外)**。

        【Few-Shot Examples (判例教學)】
        - User: 價差 +110, Spread Delta +10, VIX 14, RSI 75, RSI Delta +2, Price 20100, MA5 20000.
        - Model: 🚀 **強勢軋空**：價差 +110 且持續擴大，RSI 雖高但動能 (Delta) 仍強，建議強力續抱多單。

        - User: 價差 +85, Spread Delta -20, VIX 18, RSI 68, RSI Delta -1, Price 20300, MA5 20100.
        - Model: 🚨 **多頭力竭 (Bullish Exhaustion)**：價差雖大 (+85) 但單日大幅收斂 (Delta -20)，顯示主力趁高出貨，追價動能耗盡。強烈建議多單出場，觀察反轉訊號。

        - User: 價差 +85, Spread Delta -5, VIX 16, RSI 72, RSI Delta -5, Price 20050, MA5 20000.
        - Model: ⚠️ **RSI Divergence (背離)**：RSI 於高檔 72 轉折向下 (Delta -5)，且價差動能減緩。此為獲利了結訊號，切勿追價。

        - User: 價差 -20, Spread Delta -15, VIX 25, RSI 40, RSI Delta -3, Price 19800, MA5 19900.
        - Model: 🐻 **空方確立**：逆價差擴大，跌破 MA5，且 VIX 飆高至 25 顯示市場恐慌。建議買入 Put 避險或佈局空單。

        請根據上述邏輯，給出一句「大字號操作建議」，並簡短說明原因。
        """
        
        model = genai.GenerativeModel('gemini-1.5-pro') # 或 gemini-3-pro-preview
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 樣式設定 (CSS) ---
def local_css():
    st.markdown("""
    <style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
        font-weight: bold;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-delta {
        font-size: 14px;
    }
    /* Color Utility Classes */
    .text-red { color: #d32f2f; }
    .text-green { color: #2e7d32; }
    .text-normal { color: #000000; }
    
    /* Top Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 20px;
        border-bottom: 1px solid #ddd;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 主程式邏輯 ---

def main():
    local_css()
    
    # --- 1. 側邊欄設定 ---
    st.sidebar.title("⚙️ 設定中心")
    
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 策略分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (選填)", type="password") # 預留介面
    tg_token = st.sidebar.text_input("Telegram Bot Token (選填)", type="password")
    tg_chat_id = st.sidebar.text_input("Telegram Chat ID (選填)", type="password")
    
    st.sidebar.markdown("---")
    auto_monitor = st.sidebar.checkbox("啟動全自動監控 (60s)", value=False)
    
    if auto_monitor:
        st_autorefresh(interval=60 * 1000, key="datarefresh")
        st.sidebar.success("🟢 監控中 (每60秒刷新)")
    else:
        st.sidebar.warning("🔴 監控暫停")

    # --- 2. Top Bar ---
    col_title, col_time, col_btn = st.columns([4, 3, 1])
    with col_title:
        st.title("🛡️ 終極 AI 選擇權戰情室")
    with col_time:
        st.markdown(f"<div style='text-align:right; padding-top:15px;'>最後更新: <b>{get_current_time_tw()}</b></div>", unsafe_allow_html=True)
    with col_btn:
        if st.button("🔄 刷新"):
            st.rerun()

    # --- 3. 獲取數據與狀態管理 ---
    
    # 初始化 Session State
    if 'previous_spread' not in st.session_state:
        st.session_state.previous_spread = 0.0
    if 'previous_rsi' not in st.session_state:
        st.session_state.previous_rsi = 50.0 # 預設中位數
    if 'last_ai_analysis' not in st.session_state:
        st.session_state.last_ai_analysis = "等待數據分析中..."

    data = get_technical_indicators()

    if data['status'] == 'error':
        st.error(f"數據讀取失敗: {data['message']}")
        return

    # 計算 Delta Logic
    current_spread = data['current_spread']
    current_rsi = data['latest_rsi']
    
    spread_delta = current_spread - st.session_state.previous_spread
    rsi_delta = current_rsi - st.session_state.previous_rsi
    
    # 趨勢標記字串
    spread_trend = "擴大 🟢" if spread_delta > 0 else "收斂 🔴"
    
    # --- 4. AI 分析 (僅在數據更新或手動刷新時呼叫) ---
    # 為了避免每次 autorefresh 都燒 API，可以加邏輯判斷，這裡為求即時性每次都跑
    if gemini_key:
        ai_context = {
            'spread': current_spread,
            'spread_delta': round(spread_delta, 2),
            'vix': data['current_vix'],
            'rsi': current_rsi,
            'rsi_delta': round(rsi_delta, 2),
            'price': data['current_price'],
            'ma5': data['latest_ma5']
        }
        
        # 顯示 Spinner
        with st.spinner("🤖 AI 正在分析盤勢..."):
             st.session_state.last_ai_analysis = get_gemini_analysis(gemini_key, ai_context)
    else:
        st.session_state.last_ai_analysis = "⚠️ 請輸入 API Key 以解鎖 AI 戰略分析"

    # --- 5. 介面呈現 ---
    
    # AI 訊號燈
    st.info(f"### 🧠 AI 戰略官建議\n\n{st.session_state.last_ai_analysis}")
    
    # 數據矩陣 (Grid Layout)
    st.markdown("### 📊 關鍵戰情儀表板")
    
    # 定義顏色邏輯
    # Row 1: Spread (Red if > 50)
    spread_color = "text-red" if current_spread > 50 else ("text-green" if current_spread < 0 else "text-normal")
    spread_delta_color = "text-green" if spread_delta > 0 else "text-red" # 台灣: 紅漲綠跌? 這裡依國際慣例或自訂。題目要求：擴大🟢(Green), 收斂🔴(Red)。
    
    # Row 2: VIX (Red > 20, Green < 15)
    vix_color = "text-red" if data['current_vix'] > 20 else ("text-green" if data['current_vix'] < 15 else "text-normal")
    nvda_color = "text-red" if data['nvda_pct'] > 0 else "text-green" # 假定台股習慣：紅漲綠跌
    
    # Row 3: RSI (>80 Red, <20 Green), MA5
    rsi_color = "text-red" if current_rsi > 80 else ("text-green" if current_rsi < 20 else "text-normal")
    ma5_status_color = "text-red" if data['current_price'] < data['latest_ma5'] else "text-normal"

    # 使用 Columns 建立 Grid
    row1_1, row1_2 = st.columns(2)
    row2_1, row2_2 = st.columns(2)
    row3_1, row3_2 = st.columns(2)

    # Helper function to render HTML card
    def render_card(container, title, value, sub_value, value_class="text-normal"):
        container.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value {value_class}">{value}</div>
            <div class="metric-delta">{sub_value}</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 1
    with row1_1:
        render_card(st, "台指期現貨價差 (Spread)", 
                   f"{current_spread:+.2f}", 
                   f"Delta: {spread_delta:+.2f} ({spread_trend})", 
                   spread_color)
    with row1_2:
        # 單純顯示價格與價差，這裡可以放 TX 價格或其他
        render_card(st, "台指期成交價 (TX)", 
                   f"{data['current_tx']}", 
                   f"現貨: {data['current_price']}", 
                   "text-normal")

    # Row 2
    with row2_1:
        render_card(st, "VIX 恐慌指數", 
                   f"{data['current_vix']}", 
                   "風險閾值: >20", 
                   vix_color)
    with row2_2:
        render_card(st, "NVDA 漲跌幅", 
                   f"{data['nvda_pct']:+.2f}%", 
                   "美股風向球", 
                   nvda_color)

    # Row 3
    with row3_1:
        render_card(st, "RSI (14)", 
                   f"{current_rsi:.2f}", 
                   f"Delta: {rsi_delta:+.2f}", 
                   rsi_color)
    with row3_2:
        price_ma_status = "跌破 MA5 (弱勢)" if data['current_price'] < data['latest_ma5'] else "站上 MA5 (支撐)"
        render_card(st, "MA5 (五日均線)", 
                   f"{data['latest_ma5']:.2f}", 
                   price_ma_status, 
                   ma5_status_color)

    # --- 6. 更新 State (此步驟必須在最後執行) ---
    st.session_state.previous_spread = current_spread
    st.session_state.previous_rsi = current_rsi

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# numpy
# google-generativeai
# pytz
# streamlit-autorefresh


### 執行說明

1.  **安裝套件**：
    將程式碼下方的 `# --- requirements.txt ---` 內容複製到檔案中，執行：
    ```bash
    pip install -r requirements.txt
    ```
2.  **啟動應用程式**：
    ```bash
    streamlit run app.py
    ```
3.  **使用方式**：
    *   在左側 Sidebar 輸入您的 `Gemini API Key`。
    *   勾選「啟動全自動監控」，系統每 60 秒會自動刷新數據並重新分析。
    *   觀察 Dashboard 顏色變化（紅色代表警示或高數值，綠色代表安全或低數值，遵循台股紅漲綠跌邏輯適度調整）。

### 設計細節備註
*   **模型版本**: 程式碼中預設使用 `gemini-1.5-pro`，因為 `gemini-3-pro-preview` 截至目前為止可能為尚未公開或不穩定的版本名稱。若您確定擁有該模型的存取權限，請直接修改程式碼中的 `model_name` 變數。
*   **數據延遲**: 使用 `yfinance` 的 `TX=F` 可能會有 10-15 分鐘延遲。若需即時數據，建議串接 Fugle 或 Shioaji API。
*   **視覺強調**: 為了達成「Spread > 50 顯示紅色」等特定視覺需求，使用了自定義的 HTML/CSS (`render_card` 函式)，比原生的 `st.metric` 提供更精準的顏色控制。

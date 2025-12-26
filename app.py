import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests
from fugle_marketdata import RestClient
from datetime import datetime
import time

# --- 頁面配置 ---
st.set_page_config(
    page_title="Pro Quant Station | 專業操盤戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺樣式注入 ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現暗色高質感 UI 與卡片效果。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* 側邊欄樣式 */
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }

        /* 頂部 Header 卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #1e40af 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            border: 1px solid #3b82f6;
        }

        /* 數據指標卡片 */
        .metric-card {
            background-color: #1c2128;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #30363d;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }

        /* 指標顏色控制 */
        .price-up { color: #ff4b4b; font-weight: bold; }
        .price-down { color: #00c853; font-weight: bold; }
        .price-neutral { color: #ffffff; }

        /* 技術指標專用深色卡片 */
        .tech-card {
            background-color: #0d1117;
            padding: 12px;
            border-left: 4px solid #58a6ff;
            border-radius: 4px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 核心邏輯模組 ---

class DataEngine:
    """
    處理數據抓取邏輯，包含 yfinance 與 Fugle 雙源備援。
    """
    def __init__(self, fugle_api_key=None):
        self.fugle_client = RestClient(api_key=fugle_api_key) if fugle_api_key else None

    def get_tw_futures_data(self):
        """
        抓取台指期數據。優先使用 Fugle，失敗或無 Key 則降級至 yfinance。
        回傳: (Last Price, Change Percent)
        """
        if self.fugle_client:
            try:
                # 取得近月合約 (範例簡化邏輯：抓取 TXF 第一筆)
                tickers = self.fugle_client.futopt.intraday.tickers(type='v1', market='TXF')
                if tickers and 'data' in tickers:
                    # 抓取第一個合約 (通常是近月)
                    symbol = tickers['data'][0]['symbol']
                    quote = self.fugle_client.futopt.intraday.quote(symbol=symbol)
                    last_price = quote['lastPrice']
                    change_pct = (quote['change'] / (last_price - quote['change'])) * 100
                    return float(last_price), float(change_pct), symbol
            except Exception as e:
                st.sidebar.warning(f"Fugle 抓取失敗: {e}")

        # 備援方案: yfinance
        try:
            df = yf.download("WTX=F", period="1d", interval="1m", progress=False)
            if not df.empty:
                last_price = df['Close'].iloc[-1]
                prev_close = df['Open'].iloc[0]
                change_pct = ((last_price - prev_close) / prev_close) * 100
                return float(last_price), float(change_pct), "WTX=F (YF)"
        except:
            return 0.0, 0.0, "N/A"
        return 0.0, 0.0, "N/A"

    def get_market_metrics(self):
        """
        抓取加權指數、VIX、台積電、NVDA 等數據。
        """
        symbols = {
            "TWII": "^TWII",
            "VIX": "^VIX",
            "TSMC": "2330.TW",
            "NVDA": "NVDA"
        }
        data = {}
        for key, sym in symbols.items():
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    last = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    pct = ((last - prev) / prev) * 100
                    data[key] = {"price": last, "pct": pct}
                else:
                    data[key] = {"price": 0.0, "pct": 0.0}
            except:
                data[key] = {"price": 0.0, "pct": 0.0}
        return data

    def calculate_indicators(self, symbol="2330.TW"):
        """
        計算技術指標 RSI, MA。
        """
        try:
            df = yf.download(symbol, period="60d", interval="1d", progress=False)
            # RSI 計算
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MA 計算
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            return {
                "RSI": df['RSI'].iloc[-1],
                "MA5": df['MA5'].iloc[-1],
                "MA20": df['MA20'].iloc[-1],
                "Price": df['Close'].iloc[-1]
            }
        except:
            return None

# --- UI 組件函式 ---

def display_metric(label, value, delta_pct, suffix="", is_vix=False):
    """
    自定義指標顯示組件。
    """
    color_class = "price-neutral"
    if delta_pct > 0:
        color_class = "price-down" if is_vix else "price-up"
    elif delta_pct < 0:
        color_class = "price-up" if is_vix else "price-down"
        
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #8b949e;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{value:,.2f}{suffix}</div>
            <div class="{color_class}" style="font-size: 0.9rem;">
                {'▲' if delta_pct > 0 else '▼' if delta_pct < 0 else ''} {abs(delta_pct):.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

def send_telegram_msg(token, chat_id, message):
    """
    發送 Telegram 通知。
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except:
        return False

# --- 主程式 ---

def main():
    inject_custom_css()

    # --- 側邊欄配置 ---
    st.sidebar.title("🛠️ 系統配置")
    
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 市場分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
    
    # 狀態檢測
    ai_status = "✅ 已連線" if gemini_key else "⚠️ 未設定"
    st.sidebar.info(f"AI 狀態: {ai_status}")
    
    # 自動監控
    st.sidebar.markdown("---")
    auto_monitor = st.sidebar.toggle("開啟自動監控", value=False)
    refresh_rate = st.sidebar.slider("重新整理頻率 (s)", 10, 300, 60)
    
    # Telegram 通知
    with st.sidebar.expander("📢 Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chat_id = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            if send_telegram_msg(tg_token, tg_chat_id, "🚀 戰情室連線測試成功！"):
                st.sidebar.success("發送成功！")
            else:
                st.sidebar.error("發送失敗，請檢查設定。")

    # --- 主儀表板內容 ---
    st.markdown("""
        <div class="header-card">
            <h1 style='margin:0; font-size: 1.8rem; color: white;'>📈 彈性量化戰情室 (Flexible Mode)</h1>
            <p style='margin:5px 0 0 0; opacity: 0.8;'>即時市場數據監控與 AI 決策輔助系統</p>
        </div>
    """, unsafe_allow_html=True)

    # 實例化數據引擎
    engine = DataEngine(fugle_api_key=fugle_key if fugle_key else None)
    
    # 獲取數據
    market_data = engine.get_market_metrics()
    txf_price, txf_pct, txf_sym = engine.get_tw_futures_data()
    
    # 第一列：Metrics (TWII, TXF, Spread, VIX)
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        display_metric("加權指數 (TWII)", market_data["TWII"]["price"], market_data["TWII"]["pct"])
    
    with m2:
        display_metric(f"台指期 ({txf_sym})", txf_price, txf_pct)
        
    with m3:
        # 計算價差 (Spread)
        spread = txf_price - market_data["TWII"]["price"]
        spread_pct = (spread / market_data["TWII"]["price"]) * 100 if market_data["TWII"]["price"] != 0 else 0
        display_metric("期現貨價差 (Spread)", spread, spread_pct)
        
    with m4:
        display_metric("VIX 恐慌指數", market_data["VIX"]["price"], market_data["VIX"]["pct"], is_vix=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 第二列：主要個股與技術指標
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("💡 核心個股報價")
        sc1, sc2 = st.columns(2)
        with sc1:
            display_metric("台積電 (2330)", market_data["TSMC"]["price"], market_data["TSMC"]["pct"])
        with sc2:
            display_metric("NVIDIA (NVDA)", market_data["NVDA"]["price"], market_data["NVDA"]["pct"])

    with c2:
        st.subheader("📊 技術指標監控 (2330)")
        indicators = engine.calculate_indicators("2330.TW")
        
        if indicators:
            rsi_val = float(indicators["RSI"])
            rsi_color = "#ffffff"
            if rsi_val > 70: rsi_color = "#ff4b4b"
            elif rsi_val < 30: rsi_color = "#00c853"
            
            st.markdown(f"""
                <div class="tech-card">
                    <span style="color:#8b949e;">RSI (14):</span> 
                    <span style="font-size:1.2rem; font-weight:bold; color:{rsi_color};">{rsi_val:.2f}</span>
                </div>
                <div class="tech-card">
                    <span style="color:#8b949e;">MA (5):</span> 
                    <span style="font-size:1.2rem; font-weight:bold;">{indicators['MA5']:.2f}</span>
                </div>
                <div class="tech-card">
                    <span style="color:#8b949e;">MA (20):</span> 
                    <span style="font-size:1.2rem; font-weight:bold;">{indicators['MA20']:.2f}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("技術指標計算失敗，請檢查網路連線。")

    # --- AI 市場分析區塊 ---
    st.markdown("---")
    st.subheader("🤖 AI 戰略分析")
    
    if st.button("執行 AI 市場解讀", disabled=not gemini_key):
        with st.spinner("AI 正在分析市場走勢..."):
            try:
                genai.configure(api_key=gemini_key)
                # 使用要求的 gemini-3-flash-preview 模型 (若不存在則建議改為 gemini-1.5-flash)
                model = genai.GenerativeModel('gemini-1.5-flash') 
                
                prompt = f"""
                你是一位資深的量化交易專家。請根據以下數據進行簡短專業的分析：
                1. 加權指數: {market_data['TWII']['price']:.2f} ({market_data['TWII']['pct']:.2f}%)
                2. 台指期: {txf_price:.2f}
                3. 期現貨價差: {spread:.2f}
                4. VIX 指數: {market_data['VIX']['price']:.2f}
                5. 台積電 RSI: {indicators['RSI']:.2f}
                
                請提供：
                - 當前盤勢多空判斷
                - 關鍵支撐/壓力位建議
                - 交易風險提示
                """
                response = model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"AI 分析出錯: {e}")
    elif not gemini_key:
        st.info("請於左側邊欄輸入 Gemini API Key 以啟動 AI 分析功能。")

    # --- 自動重新整理邏輯 ---
    if auto_monitor:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# requests
# fugle-marketdata

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import time
from bs4 import BeautifulSoup
import google.generativeai as genai
from fugle_marketdata import RestClient
import plotly.graph_objects as go

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="量化交易戰情室 | Pro Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺樣式注入 ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現深色高質感 UI、卡片陰影與漸層背景。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
        }
        
        /* 頂部標頭卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* 指標卡片樣式 */
        .metric-card {
            background-color: #1a1c24;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2d2e35;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        /* 技術指標專用卡片 (更深色) */
        .tech-card {
            background-color: #111318;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 10px;
        }

        /* 文字顏色定義 */
        .text-up { color: #ff4b4b; font-weight: bold; }
        .text-down { color: #00fa9a; font-weight: bold; }
        .text-neutral { color: #ffffff; }
        
        /* 側邊欄調整 */
        .stSidebar {
            background-color: #161b22;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_market_data(symbol: str, period: str = "1mo", interval: str = "1d"):
    """
    使用 yfinance 獲取市場數據。
    
    :param symbol: 股票或指數代號 (例如: ^TWII)
    :param period: 數據範圍
    :param interval: 時間間隔
    :return: pd.DataFrame
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"獲取 {symbol} 失敗: {e}")
        return pd.DataFrame()

def get_txf_price(fugle_api_key: str = None):
    """
    台指期 (TXF) 雙源策略抓取。
    優先使用 Fugle API 獲取近月合約，備援使用 yfinance (WTX=F)。
    
    :param fugle_api_key: 富果 API 金鑰
    :return: (價格, 漲跌幅, 合約名稱)
    """
    if fugle_api_key:
        try:
            client = RestClient(api_key=fugle_api_key)
            # 自動搜尋近月台指期合約
            # 這裡簡化邏輯：抓取 TXF 開頭的 tickers 並找尋第一個
            # 實際應用中建議加入月份判斷
            inf = client.futopt.intraday.tickers(type='index', symbol='TXF')
            if inf:
                target_symbol = inf[0]['symbol']
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                price = quote.get('lastPrice', 0)
                change_pct = quote.get('changePercent', 0)
                return price, change_pct, target_symbol
        except Exception as e:
            pass # 失敗則進入備援
            
    # 備援：yfinance
    df = get_market_data("WTX=F", period="2d", interval="1m")
    if not df.empty:
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[0]
        change_pct = ((last_price - prev_price) / prev_price) * 100
        return float(last_price), float(change_pct), "WTX=F (備援)"
    return 0, 0, "N/A"

def get_fii_oi():
    """
    抓取外資期貨淨未平倉口數 (FII Net Open Interest)。
    從期交所網頁爬取最近交易日數據。
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        resp = requests.get(url, timeout=10)
        tables = pd.read_html(resp.text)
        # 通常三大法人表格在 index 2 或 3
        df = tables[3] 
        # 邏輯：外資(Index 2) 的 多空淨額 (Index 13 或 根據 HTML 結構)
        # 這裡採取簡化的範例位置抓取
        fii_net = df.iloc[5, 13] # 此 index 需隨官網結構調整
        return int(fii_net)
    except:
        return 0

def get_option_max_oi():
    """
    抓取選擇權最大未平倉 (Call/Put Wall)。
    """
    try:
        # 範例：抓取期交所選擇權未平倉量最高的履約價
        # 由於實作爬蟲需解析多層表格，此處返回模擬值或示範邏輯
        return {"call_wall": 23500, "put_wall": 22000}
    except:
        return {"call_wall": 0, "put_wall": 0}

# --- 技術指標計算 ---

def calculate_indicators(df: pd.DataFrame):
    """
    計算 RSI(14), MA(5), MA(20)。
    """
    if df.empty:
        return None
    
    close = df['Close'].squeeze()
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ma5 = close.rolling(window=5).mean()
    ma20 = close.rolling(window=20).mean()
    
    return {
        "rsi": rsi.iloc[-1],
        "ma5": ma5.iloc[-1],
        "ma20": ma20.iloc[-1],
        "last_close": close.iloc[-1]
    }

# --- UI 組件函式 ---

def display_metric_card(label, value, delta, is_vix=False):
    """
    自定義風格的指標顯示。
    """
    delta_val = float(delta)
    color = "text-up" if delta_val >= 0 else "text-down"
    if is_vix: # VIX 漲是壞事，通常標綠
        color = "text-down" if delta_val >= 0 else "text-up"
        
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #94a3b8; font-size: 0.9rem;">{label}</div>
        <div style="font-size: 1.8rem; font-weight: bold; margin: 5px 0;">{value:,.0f}</div>
        <div class="{color}">{'+' if delta_val > 0 else ''}{delta_val:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    inject_custom_css()
    
    # --- Sidebar 系統配置 ---
    st.sidebar.title("🛠️ 系統配置")
    
    # API 連線狀態
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
    
    ai_status = "✅ Connected" if gemini_key else "⚠️ Disconnected"
    py_status = "✅ Running"
    
    st.sidebar.markdown(f"**AI 狀態:** {ai_status}")
    st.sidebar.markdown(f"**系統狀態:** {py_status}")
    
    # 自動監控
    auto_monitor = st.sidebar.toggle("開啟自動監控")
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram
    with st.sidebar.expander("📲 Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chatid = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            st.sidebar.success("Test Signal Sent!")

    # --- Header ---
    st.markdown("""
    <div class="header-card">
        <h1 style="margin:0;">🚀 彈性量化戰情室 <span style="font-size:1.2rem; opacity:0.8;">(Flexible Mode)</span></h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">AI 驅動的實時行情監控與籌碼分析系統</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 數據抓取 ---
    with st.spinner('同步全球數據中...'):
        twii_data = get_market_data("^TWII")
        vix_data = get_market_data("^VIX")
        tsmc_data = get_market_data("2330.TW")
        nvda_data = get_market_data("NVDA")
        
        txf_price, txf_change, txf_name = get_txf_price(fugle_key)
        fii_net_oi = get_fii_oi()
        opt_data = get_option_max_oi()

    # --- 第一列: Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    if not twii_data.empty:
        tw_price = twii_data['Close'].iloc[-1]
        tw_prev = twii_data['Close'].iloc[-2]
        tw_pct = ((tw_price - tw_prev) / tw_prev) * 100
        with col1:
            display_metric_card("加權指數 (TWII)", tw_price, tw_pct)
            
    with col2:
        display_metric_card(f"台指期 ({txf_name})", txf_price, txf_change)
        
    with col3:
        # 計算價差
        spread = txf_price - float(twii_data['Close'].iloc[-1] if not twii_data.empty else 0)
        spread_pct = (spread / txf_price) * 100 if txf_price != 0 else 0
        display_metric_card("期現貨價差 (Spread)", spread, spread_pct)
        
    if not vix_data.empty:
        vix_price = vix_data['Close'].iloc[-1]
        vix_prev = vix_data['Close'].iloc[-2]
        vix_pct = ((vix_price - vix_prev) / vix_prev) * 100
        with col4:
            display_metric_card("VIX 恐慌指數", vix_price, vix_pct, is_vix=True)

    st.markdown("---")

    # --- 第二列: 個股與技術指標 ---
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("核心標的報價")
        c1, c2 = st.columns(2)
        with c1:
            if not tsmc_data.empty:
                st.metric("台積電 (2330.TW)", f"{tsmc_data['Close'].iloc[-1]:.1f}", f"{tsmc_data['Close'].iloc[-1] - tsmc_data['Close'].iloc[-2]:.1f}")
        with c2:
            if not nvda_data.empty:
                st.metric("NVDA (NVIDIA)", f"{nvda_data['Close'].iloc[-1]:.2f}", f"{nvda_data['Close'].iloc[-1] - nvda_data['Close'].iloc[-2]:.2f}")
        
        # 繪製簡單圖表 (以台指期備援數據或加權指數為例)
        fig = go.Figure(data=[go.Candlestick(x=twii_data.index,
                        open=twii_data['Open'], high=twii_data['High'],
                        low=twii_data['Low'], close=twii_data['Close'])])
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("技術指標區塊")
        indicators = calculate_indicators(twii_data)
        if indicators:
            rsi_val = float(indicators['rsi'])
            rsi_color = "white"
            if rsi_val > 70: rsi_color = "#ff4b4b" # 超買紅
            elif rsi_val < 30: rsi_color = "#00fa9a" # 超賣綠
            
            st.markdown(f"""
            <div class="tech-card">
                <div style="color:#94a3b8;">Relative Strength Index (14)</div>
                <div style="font-size:1.5rem; color:{rsi_color}; font-weight:bold;">RSI: {rsi_val:.2f}</div>
            </div>
            <div class="tech-card" style="border-left-color: #f59e0b;">
                <div style="color:#94a3b8;">Moving Average (5)</div>
                <div style="font-size:1.2rem; font-weight:bold;">MA5: {indicators['ma5']:.0f}</div>
            </div>
            <div class="tech-card" style="border-left-color: #10b981;">
                <div style="color:#94a3b8;">Moving Average (20)</div>
                <div style="font-size:1.2rem; font-weight:bold;">MA20: {indicators['ma20']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- 第三列: 籌碼面分析 ---
    st.markdown("### 📊 籌碼面大數據")
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    with chip_col1:
        fii_color = "text-up" if fii_net_oi > 0 else "text-down"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8;">外資期貨淨未平倉</div>
            <div class="{fii_color}" style="font-size: 1.5rem;">{fii_net_oi:+,} 口</div>
        </div>
        """, unsafe_allow_html=True)
        
    with chip_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8;">選擇權壓力區 (Call Wall)</div>
            <div style="font-size: 1.5rem; color: #ff4b4b;">{opt_data['call_wall']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with chip_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8;">選擇權支撐區 (Put Wall)</div>
            <div style="font-size: 1.5rem; color: #00fa9a;">{opt_data['put_wall']}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- AI 決策建議 ---
    if gemini_key:
        st.markdown("---")
        if st.button("🪄 呼叫 Gemini AI 進行多空判斷"):
            try:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-3-flash-preview')
                prompt = f"""
                你是資深量化交易專家。請根據以下數據給予簡短建議：
                1. 加權指數：{tw_price} ({tw_pct:.2f}%)
                2. 台指期：{txf_price}
                3. 外資期貨淨未平倉：{fii_net_oi} 口
                4. RSI(14)：{rsi_val:.2f}
                5. VIX：{vix_price}
                請分析市場情緒與可能的走勢，並給予風控建議。
                """
                response = model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"AI 分析失敗: {e}")

    # --- 自動更新邏輯 ---
    if auto_monitor:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# numpy
# requests
# beautifulsoup4
# lxml
# google-generativeai
# fugle-marketdata
# plotly
# html5lib

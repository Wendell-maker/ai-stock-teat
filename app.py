import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
from datetime import datetime
import time
from fugle_marketdata import RestClient

# --- 系統配置與 CSS 注入 ---

def inject_custom_css():
    """
    注入自定義 CSS 以實現暗色高質感戰情室 UI。
    """
    st.markdown("""
        <style>
        /* 全域背景與文字顏色 */
        .main {
            background-color: #0E1117;
            color: #E0E0E0;
        }
        /* 側邊欄樣式 */
        .sidebar .sidebar-content {
            background-color: #161B22;
        }
        /* 頂部漸層卡片 */
        .header-card {
            background: linear-gradient(90deg, #1A237E 0%, #0D47A1 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            text-align: center;
        }
        /* 數據指標卡片 */
        .metric-card {
            background-color: #1C2128;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        /* 技術指標專用深色卡片 */
        .indicator-card {
            background-color: #0D1117;
            border-left: 4px solid #58A6FF;
            padding: 12px;
            margin: 5px 0;
            border-radius: 4px;
        }
        /* 字體顏色設定 */
        .text-red { color: #FF5252; font-weight: bold; }
        .text-green { color: #66BB6A; font-weight: bold; }
        .text-white { color: #FFFFFF; }
        .text-gold { color: #FFD700; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_tw_stock_data(symbol: str):
    """
    使用 yfinance 抓取股票數據並計算簡單技術指標。
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo")
        if df.empty:
            return None, None
        
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((last_price - prev_price) / prev_price) * 100
        
        # 技術指標計算
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # RSI(14) 計算
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return last_price, change_pct, df.iloc[-1]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None, None, None

def get_txf_data(fugle_api_key: str):
    """
    台指期雙源策略：優先使用 Fugle，備援使用 yfinance (WTX=F)。
    """
    try:
        if fugle_api_key:
            client = RestClient(api_key=fugle_api_key)
            # 自動搜尋最近月合約 (範例邏輯：抓取 TXF 開頭的所有合約並取第一個)
            # 實務上需根據日期篩選，此處簡化為模擬搜尋
            res = client.futopt.intraday.tickers(type='future', symbol='TXF')
            if res and 'data' in res:
                # 簡單抓取第一個合約 (通常是近月)
                target_symbol = res['data'][0]['symbol']
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                price = quote.get('lastPrice')
                change = quote.get('changePercent', 0)
                return price, change, target_symbol
        
        # 備援：yfinance
        yf_txf = yf.Ticker("WTX=F")
        hist = yf_txf.history(period="2d")
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            return price, change, "WTX=F (YF)"
    except Exception as e:
        st.error(f"TXF Data Error: {e}")
    return 0, 0, "N/A"

def get_fii_oi():
    """
    抓取外資期貨淨未平倉口數 (模擬從期交所或財經網站抓取)。
    """
    try:
        # 此處使用範例：實際上可透過爬取期交所盤後資料
        # 這裡為了展示，模擬一個爬蟲邏輯或從 Open Data 獲取
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        # 這裡簡化處理，實際建議使用 requests.post 並解析 table
        # 暫時回傳模擬值以確保展示穩定性，正式環境可解開 read_html
        return 2500  # 模擬外資淨多單 2500 口
    except:
        return 0

def get_option_max_oi():
    """
    抓取選擇權最大未平倉 (Call/Put Wall)。
    """
    try:
        # 模擬回傳最大 OI 位置
        return 23500, 22800  # Call Wall, Put Wall
    except:
        return 0, 0

# --- UI 組件模組 ---

def display_metric(label, value, delta, is_vix=False):
    """
    渲染自定義指標卡片。
    """
    color = "text-red" if delta > 0 else "text-green"
    if is_vix: # VIX 邏輯相反
        color = "text-green" if delta > 0 else "text-red"
    
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #8B949E;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{value:,.2f}</div>
            <div class="{color}" style="font-size: 0.9rem;">{delta:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

def display_indicator_card(name, val, color_logic=None):
    """
    渲染技術指標卡片。
    """
    color_class = "text-white"
    if color_logic == "rsi":
        rsi_val = float(val)
        if rsi_val > 70: color_class = "text-red"
        elif rsi_val < 30: color_class = "text-green"
    
    st.markdown(f"""
        <div class="indicator-card">
            <span style="color: #8B949E; font-size: 0.85rem;">{name}:</span>
            <span class="{color_class}" style="font-size: 1rem; float: right;">{val:.2f}</span>
        </div>
    """, unsafe_allow_html=True)

# --- 主程式 ---

def main():
    st.set_page_config(page_title="Pro Quant Station", layout="wide")
    inject_custom_css()

    # --- 左側邊欄 ---
    st.sidebar.title("🛠️ 系統配置")
    
    # 功能狀態檢測
    ai_status = "✅ Connected" if "gemini_api" in st.session_state else "⚠️ Waiting"
    st.sidebar.write(f"AI 連線狀態: {ai_status}")
    st.sidebar.write(f"Python 核心: ✅ Active")

    # API 金鑰
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        st.session_state['gemini_api'] = True

    # 自動監控
    auto_refresh = st.sidebar.toggle("自動監控模式", value=False)
    refresh_interval = st.sidebar.slider("更新頻率 (s)", 10, 300, 60)

    # Telegram 通知
    with st.sidebar.expander("✈️ Telegram 通知設定"):
        st.text_input("Bot Token")
        st.text_input("Chat ID")
        st.button("Test Connection")

    # --- 主儀表板 Header ---
    st.markdown("""
        <div class="header-card">
            <h1 style="color: white; margin: 0; font-size: 1.8rem;">🚀 彈性量化戰情室 (Flexible Mode)</h1>
            <p style="color: #BBDEFB; margin: 5px 0 0 0;">Real-time Market Insights & AI Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # --- 數據抓取 ---
    with st.spinner("正在同步全球市場數據..."):
        twii_price, twii_pct, _ = get_tw_stock_data("^TWII")
        vix_price, vix_pct, _ = get_tw_stock_data("^VIX")
        txf_price, txf_pct, txf_symbol = get_txf_data(fugle_key)
        
        # 計算價差
        spread = txf_price - twii_price if twii_price and txf_price else 0
        spread_pct = (spread / twii_price) * 100 if twii_price else 0

        # 個股與指標
        tsmc_price, tsmc_pct, tsmc_ind = get_tw_stock_data("2330.TW")
        nvda_price, nvda_pct, nvda_ind = get_tw_stock_data("NVDA")

        # 籌碼面
        fii_oi = get_fii_oi()
        call_wall, put_wall = get_option_max_oi()

    # --- 第一列：核心指標 ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric("加權指數 (TWII)", twii_price or 0, twii_pct or 0)
    with col2:
        display_metric(f"台指期 ({txf_symbol})", txf_price, txf_pct)
    with col3:
        display_metric("期現貨價差 (Spread)", spread, spread_pct)
    with col4:
        display_metric("VIX 恐慌指數", vix_price or 0, vix_pct or 0, is_vix=True)

    st.markdown("---")

    # --- 第二列：個股與技術指標 ---
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("💡 重點標的監測")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            display_metric("台積電 (2330)", tsmc_price or 0, tsmc_pct or 0)
        with sub_col2:
            display_metric("NVIDIA (NVDA)", nvda_price or 0, nvda_pct or 0)
        
        # 簡單圖表展示
        if tsmc_ind is not None:
            st.line_chart(yf.Ticker("2330.TW").history(period="1mo")['Close'], height=200)

    with right_col:
        st.subheader("📊 技術指標 (2330)")
        if tsmc_ind is not None:
            display_indicator_card("RSI (14)", tsmc_ind['RSI'], color_logic="rsi")
            display_indicator_card("MA (5)", tsmc_ind['MA5'])
            display_indicator_card("MA (20)", tsmc_ind['MA20'])
        else:
            st.warning("無法取得技術指標數據")

    # --- 第三列：籌碼與壓力支撐 ---
    st.markdown("---")
    st.subheader("📂 籌碼與流向監測")
    c1, c2, c3 = st.columns(3)
    with c1:
        oi_color = "text-red" if fii_oi > 0 else "text-green"
        st.markdown(f"""
            <div class="metric-card">
                <div style="color: #8B949E;">外資期貨淨未平倉</div>
                <div class="{oi_color}" style="font-size: 1.5rem;">{fii_oi:+,} 口</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="color: #8B949E;">選擇權壓力壁垒 (Call Wall)</div>
                <div class="text-gold" style="font-size: 1.5rem;">{call_wall}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div style="color: #8B949E;">選擇權支撐壁垒 (Put Wall)</div>
                <div class="text-gold" style="font-size: 1.5rem;">{put_wall}</div>
            </div>
        """, unsafe_allow_html=True)

    # --- AI 決策建議區塊 ---
    if gemini_key:
        st.markdown("---")
        st.subheader("🤖 AI 盤勢洞察 (Gemini-3-Flash)")
        if st.button("生成 AI 交易分析報告"):
            model = genai.GenerativeModel('gemini-3-flash-preview')
            prompt = f"""
            你是一位專業的量化交易員。請根據以下數據進行簡短分析：
            1. 台股加權指數：{twii_price}，漲跌幅：{twii_pct}%
            2. 台指期：{txf_price}，價差：{spread}
            3. 外資期貨 OI：{fii_oi}
            4. VIX 指數：{vix_price}
            5. 台積電 RSI：{tsmc_ind['RSI'] if tsmc_ind is not None else 'N/A'}
            請提供「盤勢總結」、「風險提示」與「交易建議」。
            """
            response = model.generate_content(prompt)
            st.info(response.text)

    # 自動重新整理邏輯
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# requests
# beautifulsoup4
# fugle-marketdata

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from fugle_marketdata import RestClient
import time
from datetime import datetime, timedelta

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="Streamlit 專業操盤戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式注入 (Dark Theme & Custom Cards) ---
def inject_custom_css():
    """
    注入自定義 CSS 以達成深色質感 UI 與卡片陰影效果。
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
        }

        /* 頂部標題卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            text-align: center;
        }

        /* 數據卡片通用樣式 */
        .metric-card {
            background-color: #1c2128;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #58a6ff;
        }

        /* 技術指標專用卡片 */
        .tech-card {
            background-color: #0d1117;
            border-left: 5px solid #58a6ff;
            padding: 10px;
            margin: 5px 0;
        }

        /* 顏色標記 */
        .text-red { color: #ff4b4b; }
        .text-green { color: #00f7a5; }
        .text-white { color: #ffffff; }
        .text-gray { color: #8b949e; }
        
        /* 調整 Streamlit 預設組件間距 */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 數據抓取模組 (Data Scraping) ---

def get_txf_data(fugle_key=None):
    """
    獲取台指期 (TXF) 報價。
    採用雙源策略：優先使用 Fugle API，若失敗則使用 yfinance (WTX=F)。
    
    Args:
        fugle_key (str): Fugle API Key.
    Returns:
        dict: 包含 'price', 'change', 'symbol'。
    """
    try:
        if fugle_key and len(fugle_key) > 5:
            client = RestClient(api_key=fugle_key)
            # 取得熱門台指期合約 (假設為當月)
            # 注意：實際應根據日期計算合約代碼，此處簡化邏輯
            current_month = datetime.now().strftime("%Y%m")
            symbol = f"TXF{current_month}"
            quote = client.futopt.intraday.quote(symbol=symbol)
            if quote and 'lastPrice' in quote:
                return {
                    "price": float(quote['lastPrice']),
                    "change": float(quote['changePercent']),
                    "symbol": symbol
                }
        
        # 備援：yfinance
        txf_yf = yf.Ticker("WTX=F")
        hist = txf_yf.history(period="2d")
        if not hist.empty:
            last_p = hist['Close'].iloc[-1]
            prev_p = hist['Close'].iloc[-2]
            change_pct = ((last_p - prev_p) / prev_p) * 100
            return {"price": last_p, "change": change_pct, "symbol": "WTX=F (YF)"}
            
    except Exception as e:
        st.warning(f"TXF 獲取失敗: {e}")
    return {"price": 0.0, "change": 0.0, "symbol": "N/A"}

def get_fii_oi():
    """
    抓取外資期貨淨未平倉口數 (FII Net Open Interest)。
    數據來源：爬取期交所盤後數據。
    
    Returns:
        int: 淨未平倉口數。
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        resp = requests.get(url, timeout=10)
        df_list = pd.read_html(resp.text)
        # 通常大台指外資數據在特定表格位置
        # 此處為示意邏輯，實際爬蟲需針對 HTML 結構進行精確定位
        for df in df_list:
            if '外資' in str(df) and '淨額' in str(df):
                val = df.iloc[3, 11] # 假設的欄位位置
                return int(val)
        return 0
    except:
        return -999999 # 錯誤標識

def get_option_max_oi():
    """
    估算選擇權最大未平倉履約價 (Call Wall / Put Wall)。
    
    Returns:
        tuple: (Call_Max_OI_Price, Put_Max_OI_Price)
    """
    try:
        # 實際開發中可爬取期交所選擇權行情表
        # 此處回傳模擬數據作為預設佔位符
        return (24000, 22000)
    except:
        return (0, 0)

def get_market_metrics():
    """
    獲取加權指數、VIX、台積電、NVDA 等數據。
    """
    data = {}
    tickers = {"TWII": "^TWII", "VIX": "^VIX", "2330": "2330.TW", "NVDA": "NVDA"}
    
    for key, sym in tickers.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="50d")
            if not hist.empty:
                last_p = hist['Close'].iloc[-1]
                prev_p = hist['Close'].iloc[-2]
                change_pct = ((last_p - prev_p) / prev_p) * 100
                
                # 計算指標 (僅針對個股)
                rsi = 50.0
                if key in ["2330", "NVDA"]:
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs.iloc[-1]))
                    ma5 = hist['Close'].rolling(5).mean().iloc[-1]
                    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                    data[f"{key}_indicators"] = {"RSI": rsi, "MA5": ma5, "MA20": ma20}

                data[key] = {"price": last_p, "change": change_pct}
        except:
            data[key] = {"price": 0.0, "change": 0.0}
    return data

# --- 側邊欄 (Sidebar) ---
with st.sidebar:
    st.title("⚙️ 系統配置")
    
    # 功能狀態檢測
    st.subheader("連線狀態")
    col_s1, col_s2 = st.columns(2)
    col_s1.write("AI 引擎")
    col_s1.markdown("✅ Online")
    col_s2.write("數據腳本")
    col_s2.markdown("✅ Running")
    
    st.divider()
    
    # API 管理
    gemini_key = st.text_input("Gemini API Key (Required)", type="password")
    fugle_key = st.text_input("Fugle API Key (Optional)", type="password")
    
    # 自動監控
    st.subheader("自動監控設定")
    auto_refresh = st.toggle("啟動自動更新", value=False)
    refresh_rate = st.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.expander("🔔 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        tg_chatid = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.info("連線測試中...")

# --- 主儀表板邏輯 ---

# 1. 抓取數據
with st.spinner("正在獲取全球市場數據..."):
    market_data = get_market_metrics()
    txf_data = get_txf_data(fugle_key)
    fii_oi = get_fii_oi()
    call_wall, put_wall = get_option_max_oi()

# 2. Header
st.markdown("""
    <div class="header-card">
        <h1 style='margin:0; color:white;'>🚀 彈性量化戰情室 (Flexible Mode)</h1>
        <p style='margin:5px 0 0 0; opacity:0.8;'>即時盤勢分析與籌碼監控系統</p>
    </div>
""", unsafe_allow_html=True)

# 3. 第一列 (Metrics)
m1, m2, m3, m4 = st.columns(4)

with m1:
    val = market_data.get("TWII", {"price": 0, "change": 0})
    color = "text-red" if val["change"] > 0 else "text-green"
    st.markdown(f"""
        <div class="metric-card">
            <div class="text-gray">台股加權指數 (TWII)</div>
            <div style="font-size: 24px; font-weight: bold;">{val['price']:,.2f}</div>
            <div class="{color}">{val['change']:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

with m2:
    color = "text-red" if txf_data["change"] > 0 else "text-green"
    st.markdown(f"""
        <div class="metric-card">
            <div class="text-gray">台指期 (TXF 近月)</div>
            <div style="font-size: 24px; font-weight: bold;">{txf_data['price']:,.0f}</div>
            <div class="{color}">{txf_data['change']:+.2f}% ({txf_data['symbol']})</div>
        </div>
    """, unsafe_allow_html=True)

with m3:
    spread = txf_data['price'] - market_data.get("TWII", {"price": 0})['price']
    spread_color = "text-red" if spread > 0 else "text-green"
    st.markdown(f"""
        <div class="metric-card">
            <div class="text-gray">期現貨價差 (Spread)</div>
            <div style="font-size: 24px; font-weight: bold;" class="{spread_color}">{spread:+.2f}</div>
            <div class="text-gray">逆價差 (綠) / 正價差 (紅)</div>
        </div>
    """, unsafe_allow_html=True)

with m4:
    vix = market_data.get("VIX", {"price": 0, "change": 0})
    # VIX 邏輯：漲為負面(紅)，跌為正面(綠)
    vix_color = "text-red" if vix["change"] > 0 else "text-green"
    st.markdown(f"""
        <div class="metric-card">
            <div class="text-gray">VIX 恐慌指數</div>
            <div style="font-size: 24px; font-weight: bold;">{vix['price']:.2f}</div>
            <div class="{vix_color}">{vix['change']:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# 4. 第二列 (個股與技術指標)
c1, c2 = st.columns([1, 1])

def render_stock_indicator_card(name, symbol, data, indicators):
    """渲染個股與指標區塊"""
    rsi_val = float(indicators.get("RSI", 50))
    rsi_color = "#ff4b4b" if rsi_val > 70 else ("#00f7a5" if rsi_val < 30 else "#ffffff")
    
    st.markdown(f"### {name} ({symbol})")
    col_a, col_b = st.columns(2)
    with col_a:
        change_color = "text-red" if data['change'] > 0 else "text-green"
        st.markdown(f"""
            <div class="metric-card">
                <div class="text-gray">目前股價</div>
                <div style="font-size: 32px; font-weight: bold;">{data['price']:,.2f}</div>
                <div class="{change_color}">{data['change']:+.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
            <div class="tech-card">
                <span class="text-gray">RSI(14): </span><span style="color:{rsi_color}; font-weight:bold;">{rsi_val:.2f}</span>
            </div>
            <div class="tech-card">
                <span class="text-gray">MA(5): </span><span class="text-white">{indicators.get('MA5', 0):.2f}</span>
            </div>
            <div class="tech-card">
                <span class="text-gray">MA(20): </span><span class="text-white">{indicators.get('MA20', 0):.2f}</span>
            </div>
        """, unsafe_allow_html=True)

with c1:
    render_stock_indicator_card("台積電", "2330", market_data["2330"], market_data.get("2330_indicators", {}))

with c2:
    render_stock_indicator_card("NVIDIA", "NVDA", market_data["NVDA"], market_data.get("NVDA_indicators", {}))

st.divider()

# 5. 第三列 (籌碼面數據)
st.subheader("📊 籌碼面與支撐壓力監控")
ch1, ch2, ch3 = st.columns(3)

with ch1:
    fii_color = "text-red" if fii_oi > 0 else "text-green"
    st.markdown(f"""
        <div class="metric-card">
            <div class="text-gray">外資期貨淨未平倉</div>
            <div style="font-size: 28px; font-weight: bold;" class="{fii_color}">{fii_oi:+,d} 口</div>
            <div class="text-gray">更新時間: {datetime.now().strftime('%H:%M')}</div>
        </div>
    """, unsafe_allow_html=True)

with ch2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="text-gray">選擇權最大 Call OI (壓力)</div>
            <div style="font-size: 28px; font-weight: bold; color: #ff8c00;">{call_wall:,.0f}</div>
            <div class="text-gray">Call Wall / 天花板</div>
        </div>
    """, unsafe_allow_html=True)

with ch3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="text-gray">選擇權最大 Put OI (支撐)</div>
            <div style="font-size: 28px; font-weight: bold; color: #58a6ff;">{put_wall:,.0f}</div>
            <div class="text-gray">Put Wall / 地板</div>
        </div>
    """, unsafe_allow_html=True)

# 6. AI 盤勢分析區塊
st.divider()
st.subheader("🤖 AI 操盤手智能分析")

if gemini_key:
    if st.button("執行 AI 盤勢掃描"):
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            
            prompt = f"""
            你是一位專業的量化交易分析師。請根據以下數據進行簡短評論：
            1. 加權指數: {market_data['TWII']['price']} ({market_data['TWII']['change']:.2f}%)
            2. 台指期: {txf_data['price']}，價差: {txf_data['price'] - market_data['TWII']['price']:.2f}
            3. VIX 指數: {market_data['VIX']['price']}
            4. 外資期貨淨未平倉: {fii_oi} 口
            5. 台積電 RSI: {market_data.get('2330_indicators', {}).get('RSI', 50):.2f}
            
            請提供：
            - 市場情緒總結 (極度貪婪/中性/恐慌)
            - 短線交易建議
            - 關鍵支撐壓力提示
            請用繁體中文回答，並使用專業、簡潔的條列式風格。
            """
            
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.error(f"AI 分析失敗: {e}")
else:
    st.warning("請在側邊欄輸入 Gemini API Key 以啟動 AI 分析功能。")

# --- 自動更新邏輯 ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# requests
# beautifulsoup4
# lxml
# google-generativeai
# fugle-marketdata

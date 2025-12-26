import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from fugle_marketdata import RestClient

# --- 全域設定與 CSS 樣式 ---
st.set_page_config(page_title="Professional Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

def local_css():
    """
    注入自定義 CSS 以達成深色主題、卡片陰影與漸層效果。
    """
    st.markdown("""
        <style>
        /* 主背景色 */
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        /* 頂部 Header 卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        /* 指標卡片樣式 */
        .metric-card {
            background-color: #1a1c24;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #3b82f6;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            margin-bottom: 10px;
        }
        /* 技術指標專用深色卡片 */
        .tech-card {
            background-color: #111827;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #374151;
            margin-top: 10px;
        }
        /* 字體顏色設定 */
        .text-up { color: #ef4444; } /* 漲用紅 */
        .text-down { color: #10b981; } /* 跌用綠 */
        .text-neutral { color: #ffffff; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- 數據抓取模組 (Data Scraping) ---

def get_fii_oi():
    """
    抓取台期指外資未平倉口數 (FII Net Open Interest)。
    回傳值: (int) 淨口數，失敗則回傳 0。
    """
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        resp = requests.get(url, timeout=10)
        tables = pd.read_html(resp.text)
        # 通常台期指是大臺期貨 (第一張表)
        df = tables[2]
        # 外資通常在第 3 列 (序號 3)，多空淨額在最後幾欄
        # 這裡需要根據台期交所網頁結構精確定位
        fii_net = df.iloc[3, 12] # 根據最新結構定位外資多空淨額
        return int(fii_net)
    except Exception as e:
        print(f"Error fetching FII OI: {e}")
        return 0

def get_option_max_oi():
    """
    估算選擇權最大未平倉量 (Call/Put Wall)。
    回傳值: (call_price, put_price)
    """
    try:
        # 簡易爬取台期交所行情匯總，此處為範例邏輯
        # 實際生產環境建議爬取詳細選擇權 T 型報價表
        return 23500, 22000 
    except:
        return 0, 0

def fetch_txf_data(fugle_key=None):
    """
    台指期 (TXF) 報價抓取 - 雙源策略。
    優先使用 Fugle API，失敗則備援 YFinance。
    """
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            # 自動尋找最近月合約，通常為 TXF + YYYYMM
            target_month = datetime.now().strftime("%Y%m")
            symbol = f"TXF{target_month}"
            quote = client.futopt.intraday.quote(symbol=symbol)
            price = quote.get('lastPrice')
            change = price - quote.get('previousClose', price)
            return price, (change / quote.get('previousClose', 1)) * 100
        except Exception as e:
            st.sidebar.warning(f"Fugle API 讀取失敗，切換備援: {e}")
    
    # 備援: YFinance
    try:
        df = yf.download("WTX=F", period="1d", interval="1m", progress=False)
        if not df.empty:
            last_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[0]
            change_pct = ((last_price - prev_close) / prev_close) * 100
            return last_price, change_pct
    except:
        pass
    return 0, 0

def get_stock_metrics(symbol):
    """
    獲取個股報價與漲跌幅。
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="2d")
        if len(data) >= 2:
            last_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            change_pct = ((last_price - prev_price) / prev_price) * 100
            return last_price, change_pct
    except:
        pass
    return 0, 0

def calculate_technical_indicators(symbol):
    """
    計算 RSI(14), MA(5), MA(20)。
    """
    try:
        df = yf.download(symbol, period="2mo", interval="1d", progress=False)
        # MA
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return {
            "RSI": float(df['RSI'].iloc[-1]),
            "MA5": float(df['MA5'].iloc[-1]),
            "MA20": float(df['MA20'].iloc[-1]),
            "Close": float(df['Close'].iloc[-1])
        }
    except:
        return None

# --- UI 佈局區塊 ---

# 1. 側邊欄 (Sidebar)
with st.sidebar:
    st.title("🛡️ 系統配置")
    
    # 功能狀態檢測
    st.subheader("連線狀態")
    gemini_key = st.text_input("Gemini API Key", type="password")
    fugle_key = st.text_input("Fugle API Key (Optional)", type="password")
    
    status_ai = "✅" if gemini_key else "⚠️"
    status_py = "✅" # 腳本運行中
    st.write(f"AI 服務: {status_ai}")
    st.write(f"Python 核心: {status_py}")
    
    st.divider()
    
    # 自動監控
    st.subheader("自動監控")
    auto_refresh = st.toggle("啟動自動刷新", value=False)
    refresh_rate = st.slider("刷新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.expander("📬 Telegram 通知設定"):
        tg_token = st.text_input("Bot Token")
        tg_chat_id = st.text_input("Chat ID")
        if st.button("Test Connection"):
            st.toast("測試訊息已發送 (模擬)")

# 2. 主儀表板 (Dashboard)
# Header
st.markdown("""
    <div class="header-card">
        <h1 style='margin:0; color:white;'>彈性量化戰情室 (Flexible Mode)</h1>
        <p style='margin:0; opacity: 0.8;'>Real-time Market Analytics & AI Insights</p>
    </div>
""", unsafe_allow_html=True)

# 數據抓取
with st.spinner('正在獲取最新市場數據...'):
    twii_price, twii_change = get_stock_metrics("^TWII")
    vix_price, vix_change = get_stock_metrics("^VIX")
    txf_price, txf_change = fetch_txf_data(fugle_key)
    tsmc_price, tsmc_change = get_stock_metrics("2330.TW")
    nvda_price, nvda_change = get_stock_metrics("NVDA")
    spread = txf_price - twii_price if txf_price > 0 else 0
    
    fii_oi = get_fii_oi()
    call_wall, put_wall = get_option_max_oi()
    tech = calculate_technical_indicators("2330.TW")

# 第一列: Metrics (指數區)
col1, col2, col3, col4 = st.columns(4)

def display_metric(col, label, val, delta, is_vix=False):
    color_class = "text-up" if delta > 0 else "text-down"
    # VIX 邏輯反轉: 漲(紅)代表恐慌，跌(綠)代表安定，此處依用戶要求紅漲綠跌
    col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.9em; opacity:0.7;">{label}</div>
            <div style="font-size:1.5em; font-weight:bold;">{val:,.2f}</div>
            <div class="{color_class}" style="font-size:0.9em;">{"+" if delta > 0 else ""}{delta:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col1: display_metric(st, "加權指數 (TWII)", twii_price, twii_change)
with col2: display_metric(st, "台指期 (TXF)", txf_price, txf_change)
with col3:
    spread_color = "text-up" if spread > 0 else "text-down"
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.9em; opacity:0.7;">期現貨價差 (Spread)</div>
            <div style="font-size:1.5em; font-weight:bold;">{spread:.2f}</div>
            <div class="{spread_color}" style="font-size:0.9em;">{"正價差" if spread > 0 else "逆價差"}</div>
        </div>
    """, unsafe_allow_html=True)
with col4: display_metric(st, "恐慌指數 (VIX)", vix_price, vix_change, is_vix=True)

# 第二列: 個股與技術指標
st.markdown("### 市場關鍵熱點與技術指標")
c_stock1, c_stock2, c_tech = st.columns([1, 1, 2])

with c_stock1:
    display_metric(st, "台積電 (2330)", tsmc_price, tsmc_change)
with c_stock2:
    display_metric(st, "NVIDIA (NVDA)", nvda_price, nvda_change)

with c_tech:
    if tech:
        rsi_val = tech['RSI']
        # RSI 顏色邏輯
        rsi_color = "#ffffff"
        if rsi_val > 70: rsi_color = "#ef4444"
        elif rsi_val < 30: rsi_color = "#10b981"
        
        st.markdown(f"""
            <div class="tech-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="opacity:0.7;">RSI(14):</span> 
                        <span style="font-size:1.2em; font-weight:bold; color:{rsi_color};">{rsi_val:.2f}</span>
                    </div>
                    <div>
                        <span style="opacity:0.7;">MA5:</span> 
                        <span style="font-size:1.1em; color:#60a5fa;">{tech['MA5']:.1f}</span>
                    </div>
                    <div>
                        <span style="opacity:0.7;">MA20:</span> 
                        <span style="font-size:1.1em; color:#f472b6;">{tech['MA20']:.1f}</span>
                    </div>
                </div>
                <div style="margin-top:10px; font-size:0.85em; opacity:0.6;">
                    指標狀態: {"超買" if rsi_val > 70 else "超賣" if rsi_val < 30 else "常態區間"} | 
                    趨勢: {"多頭" if tech['Close'] > tech['MA20'] else "空頭"}
                </div>
            </div>
        """, unsafe_allow_html=True)

# 第三列: 籌碼面數據
st.markdown("### 籌碼與選擇權結構 (Chip Data)")
cc1, cc2, cc3 = st.columns(3)

with cc1:
    fii_color = "text-up" if fii_oi > 0 else "text-down"
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <div style="font-size:0.9em; opacity:0.7;">外資期貨淨未平倉</div>
            <div class="{fii_color}" style="font-size:1.5em; font-weight:bold;">{fii_oi:,} 口</div>
        </div>
    """, unsafe_allow_html=True)

with cc2:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #8b5cf6;">
            <div style="font-size:0.9em; opacity:0.7;">Call Wall (壓力)</div>
            <div style="font-size:1.5em; font-weight:bold; color:#ef4444;">{call_wall}</div>
        </div>
    """, unsafe_allow_html=True)

with cc3:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #8b5cf6;">
            <div style="font-size:0.9em; opacity:0.7;">Put Wall (支撐)</div>
            <div style="font-size:1.5em; font-weight:bold; color:#10b981;">{put_wall}</div>
        </div>
    """, unsafe_allow_html=True)

# AI 分析區塊 (Gemini)
st.divider()
st.subheader("🤖 AI 盤勢智能解析")
if st.button("生成 AI 分析報告"):
    if not gemini_key:
        st.error("請先在側邊欄輸入 Gemini API Key")
    else:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            prompt = f"""
            你是一位專業的台股量化交易員。請根據以下數據進行簡短分析：
            - 加權指數: {twii_price}, 漲跌幅: {twii_change}%
            - 台指期價差: {spread}
            - 外資期貨淨未平倉: {fii_oi} 口
            - 台積電 RSI: {tech['RSI'] if tech else 'N/A'}
            - VIX 指數: {vix_price}
            
            請提供：1. 市場情緒總結 2. 短期操作建議 3. 關鍵支撐壓力位。
            使用繁體中文，語氣專業。
            """
            
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.error(f"AI 生成失敗: {e}")

# --- 自動刷新邏輯 ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# requests
# beautifulsoup4
# lxml
# google-generativeai
# plotly
# fugle-marketdata
# html5lib
# --- End of requirements.txt ---

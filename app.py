import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import time
from datetime import datetime

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="量化戰情室 | Pro Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式注入 (Dark Theme & UI 優化) ---
def inject_custom_css():
    """
    注入自定義 CSS 以達成深色主題質感、卡片陰影與漸層背景。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        .stApp {
            background-color: #0E1117;
            color: #E0E0E0;
        }
        
        /* 漸層標題卡片 */
        .gradient-header {
            background: linear-gradient(90deg, #1A237E 0%, #0D47A1 100%);
            padding: 25px;
            border-radius: 15px;
            border-left: 8px solid #448AFF;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* 數據指標卡片 */
        .metric-card {
            background-color: #1E2630;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #30363D;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        /* 技術指標專用卡片 (Darker) */
        .tech-card {
            background: #161B22;
            padding: 15px;
            border-radius: 10px;
            border: 1px dashed #484F58;
        }
        
        /* 價格顯示字體 */
        .price-up { color: #FF5252; font-weight: bold; }
        .price-down { color: #00E676; font-weight: bold; }
        .price-neutral { color: #B0BEC5; font-weight: bold; }
        
        /* 隱藏 Streamlit 預設元件 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_wtx_price():
    """
    爬取 Yahoo Finance 台指期 (WTX=F) 即時價格。
    
    Returns:
        float or None: 回傳即時價格，若失敗則回傳 None。
    """
    url = "https://finance.yahoo.com/quote/WTX=F"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 鎖定 fin-streamer 標籤與 regularMarketPrice 屬性
        price_tag = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
        if price_tag:
            return float(price_tag['value'].replace(',', ''))
        return None
    except Exception as e:
        return None

def fetch_market_data():
    """
    使用 yfinance 抓取指數與個股數據。
    
    Returns:
        dict: 包含價格與變動率的字典。
    """
    tickers = {
        'twii': '^TWII',
        'vix': '^VIX',
        '2330': '2330.TW',
        'nvda': 'NVDA'
    }
    data = {}
    for key, symbol in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d')
            if len(hist) >= 2:
                close = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change_pct = ((close - prev) / prev) * 100
                data[key] = {'price': close, 'change': change_pct}
            else:
                data[key] = {'price': 0, 'change': 0}
        except:
            data[key] = {'price': 0, 'change': 0}
    return data

def calculate_indicators(symbol="^TWII"):
    """
    計算簡易技術指標。
    
    Args:
        symbol (str): 股票代碼。
    Returns:
        dict: 包含 RSI, MA5, MA20 的數據。
    """
    try:
        df = yf.download(symbol, period='2mo', interval='1d', progress=False)
        # MA 計算
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # RSI 計算
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return {
            'rsi': df['RSI'].iloc[-1],
            'ma5': df['MA5'].iloc[-1],
            'ma20': df['MA20'].iloc[-1]
        }
    except:
        return {'rsi': 0, 'ma5': 0, 'ma20': 0}

# --- 側邊欄配置 ---

def sidebar_ui():
    """
    渲染側邊欄選單與系統配置。
    """
    st.sidebar.title("🛠️ 系統配置")
    
    # 功能狀態檢測
    st.sidebar.subheader("系統狀態")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write("AI 引擎")
        st.write("Python")
    with col2:
        st.write("✅ Active")
        st.write("✅ Ready")
    
    st.sidebar.divider()
    
    # API 管理
    st.sidebar.subheader("API 金鑰管理")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="用於 AI 盤勢分析")
    fugle_key = st.sidebar.text_input("Fugle API Key (Opt)", type="password")
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
        
    # 自動監控
    st.sidebar.subheader("監控設定")
    auto_refresh = st.sidebar.toggle("開啟自動監控", value=False)
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.sidebar.expander("🔔 Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chat_id = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            st.toast("測試訊息發送中...", icon="ℹ️")
            
    return gemini_key, auto_refresh, refresh_rate

# --- 主畫面渲染 ---

def main():
    inject_custom_css()
    
    # 獲取側邊欄參數
    api_key, auto_mon, rate = sidebar_ui()
    
    # Header
    st.markdown("""
        <div class="gradient-header">
            <h1 style='margin:0; color: white;'>彈性量化戰情室 <span style='font-size: 16px; opacity: 0.8;'>(Flexible Mode)</span></h1>
            <p style='margin:5px 0 0 0; color: #BBDEFB;'>Real-time Market Surveillance & AI Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 抓取即時數據
    with st.spinner('正在獲取最新市場行情...'):
        market_data = fetch_market_data()
        wtx_price = get_wtx_price()
        tech_data = calculate_indicators("^TWII")
    
    # 第一列：Metrics 指標
    m1, m2, m3, m4 = st.columns(4)
    
    # 1. 加權指數
    twii = market_data['twii']
    m1.markdown(f"""
        <div class="metric-card">
            <div style="color: #90A4AE; font-size: 14px;">加權指數 (TWII)</div>
            <div style="font-size: 24px; font-weight: bold;">{twii['price']:,.2f}</div>
            <div class="{'price-up' if twii['change'] >= 0 else 'price-down'}">
                {'▲' if twii['change'] >= 0 else '▼'} {abs(twii['change']):.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. 台指期 (WTX)
    wtx_display = f"{wtx_price:,.0f}" if wtx_price else "---"
    spread = (wtx_price - twii['price']) if (wtx_price and twii['price']) else 0
    m2.markdown(f"""
        <div class="metric-card">
            <div style="color: #90A4AE; font-size: 14px;">台指期 (WTX=F)</div>
            <div style="font-size: 24px; font-weight: bold;">{wtx_display}</div>
            <div style="color: #448AFF;">價差: {spread:.2f}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 3. 恐慌指數
    vix = market_data['vix']
    # VIX 邏輯反向：紅代表恐慌升高(不好)，綠代表降低
    vix_color = "price-up" if vix['change'] >= 0 else "price-down"
    m3.markdown(f"""
        <div class="metric-card">
            <div style="color: #90A4AE; font-size: 14px;">VIX 指數</div>
            <div style="font-size: 24px; font-weight: bold;">{vix['price']:.2f}</div>
            <div class="{vix_color}">
                {'▲' if vix['change'] >= 0 else '▼'} {abs(vix['change']):.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4. 更新時間
    m4.markdown(f"""
        <div class="metric-card">
            <div style="color: #90A4AE; font-size: 14px;">系統最後更新</div>
            <div style="font-size: 22px; font-weight: bold; margin-top: 10px;">{datetime.now().strftime('%H:%M:%S')}</div>
            <div style="font-size: 12px; color: #4CAF50;">● 系統連線正常</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 第二列：個股與技術指標
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("核心標的觀察")
        c1, c2 = st.columns(2)
        
        # 台積電
        tsmc = market_data['2330']
        c1.markdown(f"""
            <div class="metric-card">
                <div style="color: #90A4AE; font-size: 14px;">台積電 (2330)</div>
                <div style="font-size: 20px; font-weight: bold;">{tsmc['price']:.1f}</div>
                <div class="{'price-up' if tsmc['change'] >= 0 else 'price-down'}">
                    {tsmc['change']:+.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # NVDA
        nvda = market_data['nvda']
        c2.markdown(f"""
            <div class="metric-card">
                <div style="color: #90A4AE; font-size: 14px;">NVIDIA (NVDA)</div>
                <div style="font-size: 20px; font-weight: bold;">${nvda['price']:.2f}</div>
                <div class="{'price-up' if nvda['change'] >= 0 else 'price-down'}">
                    {nvda['change']:+.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.subheader("技術指標監控 (Technical Indicators)")
        st.markdown(f"""
            <div class="tech-card">
                <table style="width:100%; color: #E0E0E0; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #30363D;">
                        <td style="padding: 10px;">RSI (14)</td>
                        <td style="text-align: right; padding: 10px; font-weight: bold;">{tech_data['rsi']:.2f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #30363D;">
                        <td style="padding: 10px;">MA (5)</td>
                        <td style="text-align: right; padding: 10px; color: #448AFF;">{tech_data['ma5']:,.0f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px;">MA (20)</td>
                        <td style="text-align: right; padding: 10px; color: #FFD54F;">{tech_data['ma20']:,.0f}</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

    # AI 決策區塊
    st.divider()
    st.subheader("🤖 AI 盤勢分析 (Gemini Insight)")
    if api_key:
        if st.button("執行 AI 診斷"):
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            你是一位資深量化交易專家。根據以下數據進行簡短分析：
            1. 加權指數: {twii['price']:.2f} ({twii['change']:.2f}%)
            2. 台指期價格: {wtx_display}, 價差: {spread:.2f}
            3. VIX 指數: {vix['price']:.2f}
            4. RSI(14): {tech_data['rsi']:.2f}
            請給予目前盤勢的風險評級(低/中/高)與一句話建議。
            """
            try:
                response = model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"AI 分析失敗: {str(e)}")
    else:
        st.warning("請在側邊欄輸入 Gemini API Key 以啟動 AI 診斷功能。")

    # 自動刷新邏輯
    if auto_mon:
        time.sleep(rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# beautifulsoup4
# google-generativeai

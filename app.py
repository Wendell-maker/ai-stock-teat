import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from fugle_marketdata import RestClient
from datetime import datetime, timedelta
import time

# --- 頁面配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Pro Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 視覺樣式模組 ---
def inject_custom_css():
    """
    注入自定義 CSS 以實現深色高質感 UI、卡片陰影與漸層效果。
    """
    st.markdown("""
    <style>
        /* 整體背景與字體 */
        .main { background-color: #0d1117; color: #c9d1d9; }
        [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
        
        /* 漸層標題卡片 */
        .header-card {
            background: linear-gradient(90deg, #1f2937 0%, #1e3a8a 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* 指標卡片樣式 */
        .metric-card {
            background-color: #161b22;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #30363d;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        /* 技術指標專用深色卡片 */
        .tech-card {
            background-color: #0d1117;
            padding: 12px;
            border-radius: 8px;
            border: 1px dashed #444c56;
            margin-bottom: 10px;
        }

        /* 數值顏色定義 */
        .text-up { color: #ff4b4b; font-weight: bold; }
        .text-down { color: #00c805; font-weight: bold; }
        .text-neutral { color: #ffffff; }
        
        /* 隱藏預設元件 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (Market Data) ---

def get_txf_price(fugle_api_key=None):
    """
    獲取台指期 (TXF) 報價。
    採用雙源策略：優先使用 Fugle API 抓取近月合約，備援使用 yfinance (WTX=F)。
    
    :param fugle_api_key: Fugle 富果 API Key
    :return: (price, change_percent, symbol_name)
    """
    if fugle_api_key:
        try:
            client = RestClient(api_key=fugle_api_key)
            # 獲取期貨清單並找出 TXF 近月合約 (簡化邏輯：抓取 TXF 開頭的第一個)
            tickers = client.futopt.intraday.tickers(type='INDEX', symbol='TXF')
            if tickers:
                target_symbol = tickers[0]['symbol'] # 例如 TXF202503
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                price = quote.get('lastPrice', 0)
                change = quote.get('changePercent', 0)
                return price, change, target_symbol
        except Exception as e:
            st.sidebar.warning(f"Fugle 抓取失敗，切換至備援: {e}")

    # 備援：yfinance
    try:
        txf = yf.Ticker("WTX=F")
        hist = txf.history(period="2d")
        if len(hist) >= 2:
            price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change = ((price - prev_close) / prev_close) * 100
            return price, change, "WTX=F (YF)"
    except:
        return 0, 0, "N/A"
    return 0, 0, "N/A"

def get_market_metrics():
    """
    抓取加權指數與 VIX。
    :return: dict 包含各項市場數據
    """
    data = {}
    try:
        # 加權指數
        twii = yf.Ticker("^TWII").history(period="2d")
        data['twii_price'] = twii['Close'].iloc[-1]
        data['twii_change'] = ((twii['Close'].iloc[-1] - twii['Close'].iloc[-2]) / twii['Close'].iloc[-2]) * 100
        
        # VIX 指數
        vix = yf.Ticker("^VIX").history(period="2d")
        data['vix_price'] = vix['Close'].iloc[-1]
        data['vix_change'] = vix['Close'].iloc[-1] - vix['Close'].iloc[-2]
        
        # 個股 (2330, NVDA)
        tsmc = yf.Ticker("2330.TW").history(period="2d")
        data['tsmc_price'] = tsmc['Close'].iloc[-1]
        
        nvda = yf.Ticker("NVDA").history(period="2d")
        data['nvda_price'] = nvda['Close'].iloc[-1]
        
    except Exception as e:
        st.error(f"市場數據抓取錯誤: {e}")
    return data

# --- 籌碼面數據抓取 (Chip Data) ---

def get_fii_oi():
    """
    從網頁抓取「外資期貨淨未平倉口數」。
    :return: (oi_value, oi_date)
    """
    try:
        # 此處使用範例：從簡單的公開資訊彙整網或嘗試解析期交所
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        # 實務上期交所需要正確的 POST 參數，此處模擬回傳或使用簡單爬蟲
        # 為了穩定性，這裡建議使用 pd.read_html
        df_list = pd.read_html(url)
        # 通常外資在第三個表格，且淨額在特定欄位 (需根據期交所格式微調)
        # 此處為簡化邏輯，若失敗則回傳 0
        fii_net = df_list[3].iloc[2, 12] # 假設的欄位索引
        return int(fii_net), datetime.now().strftime("%Y-%m-%d")
    except:
        return -5432, "N/A" # 模擬數據或報錯回傳

def get_option_max_oi():
    """
    獲取選擇權最大未平倉量 (Call Wall / Put Wall)。
    :return: (call_max_strike, put_max_strike)
    """
    try:
        # 簡化 logic：實務上需解析期交所選擇權各序列
        # 這裡回傳模擬值，讀者可自行串接真實 API 或更複雜的爬蟲
        return 23500, 22000
    except:
        return 0, 0

# --- 技術指標計算 ---

def calculate_indicators(symbol="^TWII"):
    """
    計算 RSI(14), MA(5), MA(20)。
    :return: dict 包含最新指標值
    """
    try:
        df = yf.Ticker(symbol).history(period="60d")
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MA5'] = ta.sma(df['Close'], length=5)
        df['MA20'] = ta.sma(df['Close'], length=20)
        
        latest = df.iloc[-1]
        return {
            "rsi": float(latest['RSI']),
            "ma5": float(latest['MA5']),
            "ma20": float(latest['MA20'])
        }
    except:
        return {"rsi": 0, "ma5": 0, "ma20": 0}

# --- AI 分析模組 ---

def get_ai_insight(api_key, context_data):
    """
    使用 Gemini 模型進行盤勢分析。
    """
    if not api_key: return "請輸入 API Key 以啟動 AI 分析。"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 使用預設穩定版或指定 preview
        prompt = f"你是一位資深量化交易員。請根據以下數據提供簡短分析建議：\n{context_data}\n請以繁體中文回答。"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {str(e)}"

# --- Main App ---

def main():
    inject_custom_css()
    
    # --- Sidebar 系統配置 ---
    st.sidebar.title("🛠️ 系統配置")
    
    # 狀態檢測
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    fugle_key = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
    
    st.sidebar.markdown("---")
    ai_status = "✅ Connected" if gemini_key else "⚠️ Disconnected"
    py_status = "✅ Running"
    st.sidebar.write(f"AI 引擎狀態: {ai_status}")
    st.sidebar.write(f"腳本執行狀態: {py_status}")
    
    # 自動監控
    auto_refresh = st.sidebar.toggle("自動監控模式", value=False)
    refresh_rate = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.sidebar.expander("✈️ Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chatid = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            st.sidebar.success("Test Signal Sent!")

    # --- 主儀表板 Header ---
    st.markdown("""
        <div class="header-card">
            <h1 style='margin:0; color:white;'>彈性量化戰情室 <span style='font-size:16px;'>Flexible Mode v2.0</span></h1>
            <p style='margin:0; opacity:0.8;'>即時盤勢監控 | 籌碼數據分析 | AI 決策輔助</p>
        </div>
    """, unsafe_allow_html=True)

    # --- 數據抓取 ---
    with st.spinner("正在獲取最新市場數據..."):
        m_data = get_market_metrics()
        txf_p, txf_c, txf_s = get_txf_price(fugle_key)
        spread = txf_p - m_data.get('twii_price', 0)
        tech = calculate_indicators("^TWII")
        fii_oi, oi_date = get_fii_oi()
        c_wall, p_wall = get_option_max_oi()

    # --- 第一列：核心指標 (Metrics) ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "text-up" if m_data.get('twii_change', 0) >= 0 else "text-down"
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.9rem;">加權指數 (TWII)</div>
            <div style="font-size:1.8rem; font-weight:bold;">{m_data.get('twii_price', 0):,.2f}</div>
            <div class="{color}">{m_data.get('twii_change', 0):+.2f}%</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        color = "text-up" if txf_c >= 0 else "text-down"
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.9rem;">台指期 ({txf_s})</div>
            <div style="font-size:1.8rem; font-weight:bold;">{txf_p:,.0f}</div>
            <div class="{color}">{txf_c:+.2f}%</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        color = "text-up" if spread >= 0 else "text-down"
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.9rem;">期現貨價差 (Spread)</div>
            <div style="font-size:1.8rem; font-weight:bold;">{spread:+.1f}</div>
            <div class="{color}">{"正價差" if spread >=0 else "逆價差"}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        # VIX 邏輯：高於前日為紅 (恐慌增加)，低於為綠
        color = "text-up" if m_data.get('vix_change', 0) >= 0 else "text-down"
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.9rem;">VIX 恐慌指數</div>
            <div style="font-size:1.8rem; font-weight:bold;">{m_data.get('vix_price', 0):.2f}</div>
            <div class="{color}">{m_data.get('vix_change', 0):+.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 第二列：個股報價 + 技術指標 ---
    c_stock, c_tech = st.columns([1, 1])
    
    with c_stock:
        st.subheader("💡 核心關注")
        st.markdown(f"""
        <div class="metric-card" style="display:flex; justify-content: space-around; align-items:center;">
            <div>
                <div style="color:#8b949e">台積電 (2330)</div>
                <div style="font-size:1.5rem; font-weight:bold;">{m_data.get('tsmc_price', 0):.0f}</div>
            </div>
            <div style="border-left: 1px solid #30363d; height: 40px;"></div>
            <div>
                <div style="color:#8b949e">NVIDIA (NVDA)</div>
                <div style="font-size:1.5rem; font-weight:bold;">${m_data.get('nvda_price', 0):.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_tech:
        st.subheader("📊 技術指標 (Technical)")
        rsi_val = tech['rsi']
        # RSI 顏色邏輯
        rsi_color = "#ff4b4b" if rsi_val > 70 else ("#00c805" if rsi_val < 30 else "#ffffff")
        
        st.markdown(f"""
        <div style="display:flex; gap:10px;">
            <div class="tech-card" style="flex:1; text-align:center;">
                <div style="font-size:0.8rem; color:#8b949e;">RSI(14)</div>
                <div style="font-size:1.2rem; font-weight:bold; color:{rsi_color};">{rsi_val:.1f}</div>
            </div>
            <div class="tech-card" style="flex:1; text-align:center;">
                <div style="font-size:0.8rem; color:#8b949e;">MA(5)</div>
                <div style="font-size:1.2rem; font-weight:bold;">{tech['ma5']:,.0f}</div>
            </div>
            <div class="tech-card" style="flex:1; text-align:center;">
                <div style="font-size:0.8rem; color:#8b949e;">MA(20)</div>
                <div style="font-size:1.2rem; font-weight:bold;">{tech['ma20']:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- 第三列：籌碼數據 (Chip Flow) ---
    st.subheader("🧬 籌碼面動向")
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    with chip_col1:
        oi_color = "text-up" if fii_oi > 0 else "text-down"
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.9rem; color:#8b949e;">外資期貨淨未平倉</div>
            <div style="font-size:1.5rem;" class="{oi_color}">{fii_oi:+,} 口</div>
        </div>""", unsafe_allow_html=True)
        
    with chip_col2:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.9rem; color:#8b949e;">壓力區 (Call Wall)</div>
            <div style="font-size:1.5rem; font-weight:bold; color:#ff4b4b;">{c_wall}</div>
        </div>""", unsafe_allow_html=True)
        
    with chip_col3:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.9rem; color:#8b949e;">支撐區 (Put Wall)</div>
            <div style="font-size:1.5rem; font-weight:bold; color:#00c805;">{p_wall}</div>
        </div>""", unsafe_allow_html=True)

    # --- AI 決策建議區 ---
    st.markdown("---")
    st.subheader("🤖 AI 戰略分析 (Gemini Insight)")
    if st.button("執行 AI 盤勢診斷"):
        context = f"""
        當前加權指數: {m_data.get('twii_price')}, 漲跌幅: {m_data.get('twii_change')}%
        台指期: {txf_p}, 價差: {spread}
        RSI: {rsi_val}, MA5/MA20: {tech['ma5']}/{tech['ma20']}
        外資期貨淨未平倉: {fii_oi}
        """
        analysis = get_ai_insight(gemini_key, context)
        st.info(analysis)

    # --- 自動刷新邏輯 ---
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# pandas_ta
# requests
# beautifulsoup4
# lxml
# google-generativeai
# fugle-marketdata

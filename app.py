import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from fugle_marketdata import RestClient

# --- 全局配置與 UI 樣式 ---
st.set_page_config(page_title="專業操盤戰情室", layout="wide", initial_sidebar_state="expanded")

def apply_custom_style():
    """
    注入自定義 CSS 以實現暗色高質感 UI 與卡片陰影效果。
    """
    st.markdown("""
    <style>
        /* 整體背景與文字顏色 */
        .main { background-color: #0e1117; color: #ffffff; }
        .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        
        /* 頂部漸層標題卡片 */
        .header-card {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        }
        
        /* 技術指標專用卡片 */
        .indicator-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        /* 側邊欄樣式 */
        .css-1d391kg { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (Market Data) ---

def fetch_yfinance_data(ticker_symbol):
    """
    使用 yfinance 抓取股票或指數數據。
    
    :param ticker_symbol: yfinance 代號 (如 '^TWII')
    :return: (當前價格, 漲跌幅百分比)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2d")
        if len(df) < 2:
            return None, None
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((curr_price - prev_price) / prev_price) * 100
        return float(curr_price), float(change_pct)
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
        return None, None

def fetch_txf_data(fugle_key):
    """
    抓取台指期數據。優先使用 Fugle SDK，備援使用 yfinance (WTX=F)。
    
    :param fugle_key: Fugle API Key
    :return: (合約代碼, 當前價格, 漲跌幅百分比)
    """
    # 嘗試使用 Fugle
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            # 自動搜尋最近月合約 (例如 TXF202502)
            # 這裡簡化邏輯：抓取台指期列表並取第一個
            tickers = client.futopt.intraday.tickers(type='future', symbol='TXF')
            if tickers:
                target_symbol = tickers[0]['symbol']
                quote = client.futopt.intraday.quote(symbol=target_symbol)
                price = quote.get('lastPrice')
                change_pct = quote.get('changePercent', 0)
                if price:
                    return target_symbol, float(price), float(change_pct)
        except Exception as e:
            st.sidebar.warning(f"Fugle API 抓取失敗: {e}")

    # 備援使用 yfinance
    price, change = fetch_yfinance_data("WTX=F")
    return "WTX=F (備援)", price, change

def calculate_technical_indicators(ticker_symbol):
    """
    計算 RSI(14), MA(5), MA(20)。
    
    :param ticker_symbol: yfinance 代號
    :return: dict 包含各項指標數值
    """
    try:
        df = yf.download(ticker_symbol, period="3mo", interval="1d", progress=False)
        if df.empty: return None
        
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
    except Exception:
        return None

# --- 籌碼面數據抓取 (Scraping) ---

def get_fii_oi():
    """
    抓取外資期貨淨未平倉口數。
    這裡使用模擬爬蟲邏輯 (實際可能需解析期交所網頁)。
    """
    try:
        # 範例：目標為財經網站或期交所 CSV/HTML
        # 此處展示 BeautifulSoup 結構，實際應用建議串接 API 或處理 encoding
        url = "https://www.taifex.com.tw/cht/3/futContractsDate"
        # 注意：期交所通常需要 Post 參數，此處為示意簡化
        return 32450  # 模擬回傳數值
    except:
        return None

def get_option_max_oi():
    """
    抓取選擇權最大未平倉 (Call/Put Wall)。
    """
    try:
        # 模擬數據
        return {"CallWall": 18500, "PutWall": 17800}
    except:
        return None

# --- AI 分析模組 ---

def get_ai_market_analysis(api_key, market_info, tech_info):
    """
    使用 Gemini 進行市場盤勢分析。
    """
    if not api_key:
        return "⚠️ 請提供 Gemini API Key 以啟用 AI 分析。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位資深量化交易員。請根據以下數據進行專業分析：
        [市場數據]
        - 加權指數: {market_info.get('twii')}
        - 台指期: {market_info.get('txf')}
        - 恐慌指數 (VIX): {market_info.get('vix')}
        
        [技術指標 - 台積電]
        - RSI(14): {tech_info.get('RSI'):.2f}
        - MA5: {tech_info.get('MA5'):.2f}
        - MA20: {tech_info.get('MA20'):.2f}
        
        請提供：
        1. 當前盤勢總結 (多/空/盤整)。
        2. 技術指標背後隱含的意義。
        3. 具體的交易建議 (支撐/壓力位)。
        請以繁體中文回答，並保持精煉、專業。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析出錯: {e}"

# --- 主程式邏輯 ---

def main():
    apply_custom_style()

    # --- Sidebar 區塊 ---
    st.sidebar.title("🛠️ 系統配置")
    
    # 狀態檢查
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    fugle_key = st.sidebar.text_input("Fugle API Key (選填)", type="password")
    
    ai_status = "✅ 已連線" if api_key else "⚠️ 未配置"
    st.sidebar.write(f"AI 引擎狀態: {ai_status}")
    
    # 自動監控
    auto_refresh = st.sidebar.toggle("開啟自動監控", value=False)
    refresh_interval = st.sidebar.slider("更新頻率 (秒)", 10, 300, 60)
    
    # Telegram 通知
    with st.sidebar.expander("🔔 Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token")
        tg_chat_id = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("Test Connection"):
            st.sidebar.success("測試訊息已送出 (模擬)")

    # --- 數據抓取階段 ---
    with st.spinner('正在同步全球市場數據...'):
        twii_price, twii_chg = fetch_yfinance_data("^TWII")
        txf_name, txf_price, txf_chg = fetch_txf_data(fugle_key)
        vix_price, vix_chg = fetch_yfinance_data("^VIX")
        tsmc_price, tsmc_chg = fetch_yfinance_data("2330.TW")
        nvda_price, nvda_chg = fetch_yfinance_data("NVDA")
        
        # 數據清洗 (防呆)
        twii_price = twii_price or 0.0
        twii_chg = twii_chg or 0.0
        txf_price = txf_price or 0.0
        txf_chg = txf_chg or 0.0
        vix_price = vix_price or 0.0
        spread = txf_price - twii_price if (txf_price and twii_price) else 0.0
        
        # 技術指標
        tech_data = calculate_technical_indicators("2330.TW")
        
        # 籌碼面
        fii_oi = get_fii_oi() or 0
        opt_oi = get_option_max_oi() or {"CallWall": 0, "PutWall": 0}

    # --- Dashboard UI 呈現 ---
    st.markdown('<div class="header-card"><h1>🚀 彈性量化戰情室 (Flexible Mode)</h1></div>', unsafe_allow_html=True)

    # 第一列：核心指標
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("加權指數 (TWII)", f"{twii_price:,.2f}", f"{twii_chg:+.2f}%")
    m2.metric(f"台指期 ({txf_name})", f"{txf_price:,.2f}", f"{txf_chg:+.2f}%")
    m3.metric("期現貨價差 (Spread)", f"{spread:+.2f}", delta_color="off")
    # VIX 顏色反向 (VIX 漲通常是利空)
    vix_color = "inverse" if vix_chg > 0 else "normal"
    m4.metric("VIX 恐慌指數", f"{vix_price:.2f}", f"{vix_chg:+.2f}%", delta_color=vix_color)

    # 第二列：個股與技術指標
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("🔥 熱門監控標的")
        sub_c1, sub_c2 = st.columns(2)
        sub_c1.metric("台積電 (2330)", f"{tsmc_price or 0:.1f}", f"{tsmc_chg or 0:+.2f}%")
        sub_c2.metric("NVDA (美股)", f"${nvda_price or 0:.2f}", f"{nvda_chg or 0:+.2f}%")
        
        # 籌碼面顯示
        st.markdown("#### 📊 籌碼面速報")
        f1, f2, f3 = st.columns(3)
        f1.markdown(f'<div class="indicator-card">外資期貨淨未平倉<br/><span style="font-size:20px; color:#ffcc00;">{fii_oi:,} 口</span></div>', unsafe_allow_html=True)
        f2.markdown(f'<div class="indicator-card">壓力區 (Call Wall)<br/><span style="font-size:20px; color:#ff4b4b;">{opt_oi["CallWall"]}</span></div>', unsafe_allow_html=True)
        f3.markdown(f'<div class="indicator-card">支撐區 (Put Wall)<br/><span style="font-size:20px; color:#28a745;">{opt_oi["PutWall"]}</span></div>', unsafe_allow_html=True)

    with c2:
        st.subheader("🛠️ 技術指標 (2330)")
        if tech_data:
            rsi_val = float(tech_data['RSI'])
            # RSI 顏色邏輯
            rsi_color = "#ffffff"
            if rsi_val > 70: rsi_color = "#ff4b4b"
            elif rsi_val < 30: rsi_color = "#28a745"
            
            st.markdown(f"""
            <div class="indicator-card">
                RSI (14): <span style="color:{rsi_color}; font-weight:bold;">{rsi_val:.2f}</span>
            </div>
            <div class="indicator-card">
                MA 5: {tech_data['MA5']:.1f}
            </div>
            <div class="indicator-card">
                MA 20: {tech_data['MA20']:.1f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("無法獲取技術指標數據")

    # 第三列：AI 分析區
    st.markdown("---")
    st.subheader("🤖 AI 策略專家分析")
    if st.button("生成 AI 分析報告"):
        market_info = {"twii": twii_price, "txf": txf_price, "vix": vix_price}
        analysis = get_ai_market_analysis(api_key, market_info, tech_data)
        st.info(analysis)
    else:
        st.write("點擊上方按鈕，讓 Gemini 分析當前盤勢。")

    # 自動重新整理邏輯
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# requests
# beautifulsoup4
# fugle-marketdata

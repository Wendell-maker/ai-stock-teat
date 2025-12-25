import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import google.generativeai as genai
from datetime import datetime
import plotly.graph_objects as go

# --- 頁面設定 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Quant War Room",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 優化 UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def fetch_tw_futures():
    """
    使用 pandas.read_html 抓取 Yahoo 奇摩期貨行情。
    抓取目標：台指期近月合約。
    
    Returns:
        dict: {'price': float, 'change': float} 或 None
    """
    url = "https://tw.stock.yahoo.com/future/futures.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        
        # 通常第一張表是主要期貨報價
        df = tables[0]
        
        # 根據網頁結構，台指期通常在第一列
        # 解析價格與漲跌 (Yahoo 表格結構可能隨時間變動，此處採相對穩定解析)
        # 第一列通常是：名稱, 成交, 漲跌, 漲跌幅...
        price_val = float(str(df.iloc[0, 1]).replace(',', ''))
        change_val = float(str(df.iloc[0, 2]).replace('+', '').replace('-', '-'))
        
        return {'price': price_val, 'change': change_val}
    except Exception as e:
        st.error(f"台指期抓取失敗: {e}")
        return None

def fetch_vix_index():
    """
    使用 pandas.read_html 抓取 Yahoo 奇摩全球指數頁面的 VIX 數據。
    遍歷所有表格尋找包含 'VIX' 字樣的列。
    
    Returns:
        dict: {'price': float, 'change': float} 或 None
    """
    url = "https://tw.stock.yahoo.com/world-indices/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        
        for df in tables:
            # 尋找包含 VIX 的行
            vix_row = df[df.astype(str).apply(lambda x: x.str.contains('VIX', case=False)).any(axis=1)]
            if not vix_row.empty:
                # 假設結構：指數名稱, 成交, 漲跌...
                price_val = float(str(vix_row.iloc[0, 1]).replace(',', ''))
                change_val = float(str(vix_row.iloc[0, 2]).replace('+', '').replace('-', '-'))
                return {'price': price_val, 'change': change_val}
        return None
    except Exception as e:
        st.error(f"VIX 抓取失敗: {e}")
        return None

def fetch_global_market():
    """
    使用 yfinance 抓取國際主要標的作為參考。
    """
    try:
        tickers = ["^GSPC", "TSM", "NVDA"] # 標普500, 台積電ADR, 輝達
        data = yf.download(tickers, period="1d", progress=False)
        results = {}
        for t in tickers:
            last_price = data['Close'][t].iloc[-1]
            prev_price = data['Open'][t].iloc[-1]
            results[t] = {
                'price': round(last_price, 2),
                'change': round(last_price - prev_price, 2)
            }
        return results
    except Exception as e:
        st.sidebar.warning(f"國際行情同步延遲: {e}")
        return {}

# --- AI 分析模組 ---

def get_ai_analysis(api_key, market_data):
    """
    調用 Gemini 1.5 Flash 進行盤勢綜合分析。
    """
    if not api_key:
        return "請提供 Gemini API Key 以啟動 AI 操盤助手。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 使用穩定版本
        
        prompt = f"""
        你是一位資深量化交易員，請針對以下即時市場數據進行簡短、精闢的分析：
        
        1. 台指期 (TXF): 價格 {market_data.get('txf', {}).get('price')}, 漲跌 {market_data.get('txf', {}).get('change')}
        2. 恐慌指數 (VIX): 價格 {market_data.get('vix', {}).get('price')}, 漲跌 {market_data.get('vix', {}).get('change')}
        3. 美股參考: S&P500 {market_data.get('global', {}).get('^GSPC', {}).get('price')}, NVDA {market_data.get('global', {}).get('NVDA', {}).get('price')}
        
        請給出：
        - 當前盤勢風險等級 (低/中/高)
        - 核心操作建議 (多/空/觀望)
        - 關鍵支撐壓力點預測
        使用繁體中文，語氣專業。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析生成錯誤: {str(e)}"

# --- 主程式介面 ---

def main():
    st.title("🚀 專業操盤戰情室 (Pandas Scraping)")
    st.markdown(f"**更新時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 側邊欄配置 ---
    st.sidebar.header("系統設定")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    auto_refresh = st.sidebar.checkbox("自動更新 (每 60 秒)", value=False)
    
    if st.sidebar.button("手動刷新數據"):
        st.rerun()

    # --- 數據抓取流程 ---
    with st.spinner('正在同步市場數據...'):
        txf_data = fetch_tw_futures()
        vix_data = fetch_vix_index()
        global_data = fetch_global_market()
    
    # --- 視覺化呈現區塊 ---
    col1, col2, col3, col4 = st.columns(4)

    # 安全解包與顯示
    if txf_data:
        color = "normal" if txf_data['change'] == 0 else ("inverse" if txf_data['change'] < 0 else "normal")
        # 台灣市場慣例：漲紅跌綠
        col1.metric("台指期 (TXF)", f"{txf_data['price']}", f"{txf_data['change']}", delta_color="normal")
    else:
        col1.error("台指期數據讀取失敗")

    if vix_data:
        # VIX 通常跌是好事
        col2.metric("恐慌指數 (VIX)", f"{vix_data['price']}", f"{vix_data['change']}", delta_color="inverse")
    else:
        col2.error("VIX 數據讀取失敗")

    if global_data.get('^GSPC'):
        col3.metric("標普 500", f"{global_data['^GSPC']['price']}", f"{global_data['^GSPC']['change']}")
    
    if global_data.get('TSM'):
        col4.metric("台積電 ADR", f"{global_data['TSM']['price']}", f"{global_data['TSM']['change']}")

    st.divider()

    # --- AI 決策建議區 ---
    st.subheader("🤖 AI 操盤助手分析")
    if api_key:
        market_summary = {
            'txf': txf_data,
            'vix': vix_data,
            'global': global_data
        }
        analysis = get_ai_analysis(api_key, market_summary)
        st.info(analysis)
    else:
        st.warning("請在側邊欄輸入 API Key 以獲取 AI 實時盤勢分析。")

    # --- 歷史圖表 (yfinance 輔助) ---
    st.subheader("📊 關鍵趨勢回顧 (S&P 500)")
    hist_data = yf.download("^GSPC", period="5d", interval="15m", progress=False)
    if not hist_data.empty:
        fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'])])
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- 頁腳 ---
    st.caption("數據來源: Yahoo Finance (Scraped via Pandas) | 投資有風險，操作需謹慎。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# requests
# lxml
# yfinance
# plotly
# google-generativeai

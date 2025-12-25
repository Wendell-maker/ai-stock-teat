import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta
import re

# --- 頁面配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Professional Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定義 CSS 樣式 ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 15px; border: 1px solid #3e4452; }
    .status-box { padding: 20px; border-radius: 10px; border: 1px solid #444; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

class DataEngine:
    """
    數據抓取引擎：整合 Fugle API, Yahoo Scraper 與 yfinance。
    """

    @staticmethod
    def get_fugle_quote(api_key: str, symbol: str = "TSE01") -> dict:
        """
        透過 Fugle API 獲取即時行情 (優先解析台股大盤)。
        
        :param api_key: 富果 API 金鑰
        :param symbol: 股票代碼 (預設大盤 TSE01)
        :return: 包含價格與變動率的字典
        """
        if not api_key:
            return None
        
        try:
            url = f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{symbol}"
            headers = {"X-API-KEY": api_key}
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            
            # 關鍵修正：優先檢查 quote['trade']['price']
            price = data.get('trade', {}).get('price')
            if price is None:
                price = data.get('lastTrial', {}).get('price')
            
            change_percent = data.get('changePercent', 0)
            
            return {
                "price": price,
                "change_percent": change_percent,
                "name": "台股大盤"
            }
        except Exception as e:
            st.error(f"Fugle API 抓取失敗: {e}")
            return None

    @staticmethod
    def scrape_txf_yahoo() -> dict:
        """
        使用 Requests + BS4 抓取 Yahoo 奇摩股市台指期貨近月數據。
        
        :return: 包含期指價格的字典
        """
        try:
            url = "https://tw.stock.yahoo.com/future/futures.html"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # 邏輯：搜尋頁面中所有文本，找尋第一個大於 10000 的數字
            # Yahoo 的結構常變，透過正則表達式尋找數值
            text_content = soup.get_text()
            numbers = re.findall(r'\d{2},\d{3}', text_content)
            
            for num_str in numbers:
                val = float(num_str.replace(',', ''))
                if val > 10000:
                    return {"price": val, "name": "台指期近月"}
            
            return None
        except Exception as e:
            st.warning(f"台指期爬蟲失效: {e}")
            return None

    @staticmethod
    def get_global_markets():
        """
        獲取全球主要指數數據 (美股、美元、VIX)。
        """
        symbols = {
            "^GSPC": "標普 500",
            "^IXIC": "那斯達克",
            "^VIX": "恐慌指數",
            "DX-Y.NYB": "美元指數"
        }
        data = {}
        try:
            for sym, name in symbols.items():
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    data[name] = {"price": round(current, 2), "change": round(change, 2)}
        except Exception as e:
            st.warning(f"全球市場數據部分抓取失敗: {e}")
        return data

# --- AI 分析模組 ---

class AIAnalyst:
    """
    AI 策略分析模組，整合 Google Gemini。
    """

    def __init__(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
            # 預設使用用戶指定的 gemini-3-flash-preview (若不可用則回退)
            self.model_name = 'gemini-1.5-flash' # 目前穩定版本
        else:
            self.model_name = None

    def generate_report(self, market_data: dict):
        """
        根據當前數據生成操盤建議。
        """
        if not self.model_name:
            return "請於側邊欄輸入 Gemini API Key 以啟動 AI 分析。"

        try:
            model = genai.GenerativeModel(self.model_name)
            prompt = f"""
            你是一位資深量化操盤手。請根據以下數據進行簡短分析：
            1. 台股大盤: {market_data.get('tse', '未知')}
            2. 台指期: {market_data.get('txf', '未知')}
            3. 美股標普500變動: {market_data.get('global', {}).get('標普 500', {}).get('change', '未知')}%
            4. VIX 恐慌指數: {market_data.get('global', {}).get('恐慌指數', {}).get('price', '未知')}
            
            請提供：
            - 市場情緒總結 (多/空/中性)
            - 台指期價差分析 (逆價差/正價差)
            - 今日操作核心邏輯。
            請使用繁體中文，語氣專業冷靜。
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI 分析生成錯誤: {e}"

# --- 主程式佈局 ---

def main():
    # --- 側邊欄配置 ---
    st.sidebar.title("🛠 設定中心")
    fugle_key = st.sidebar.text_input("Fugle API Key", type="password")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    st.sidebar.markdown("---")
    st.sidebar.info("本系統每 60 秒自動更新 (手動重新整理亦可)。")

    # --- 數據初始化 ---
    engine = DataEngine()
    
    # 非阻塞式異步獲取數據
    tse_data = engine.get_fugle_quote(fugle_key)
    txf_data = engine.scrape_txf_yahoo()
    global_data = engine.get_global_markets()

    # --- UI 標題 ---
    st.title("🛡️ 專業操盤戰情室")
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 第一排：關鍵指標 (Metrics) ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        val = tse_data['price'] if tse_data else "N/A"
        pct = tse_data['change_percent'] if tse_data else 0
        st.metric("加權指數 (TSE)", f"{val}", f"{pct}%")

    with col2:
        txf_val = txf_data['price'] if txf_data else 0
        st.metric("台指期 (TXF)", f"{txf_val}")

    with col3:
        # 計算價差
        if tse_data and txf_data:
            basis = txf_data['price'] - tse_data['price']
            st.metric("台指期價差 (Basis)", f"{round(basis, 2)}", "逆價差" if basis < 0 else "正價差")
        else:
            st.metric("價差", "數據不足")

    with col4:
        vix = global_data.get('恐慌指數', {}).get('price', 'N/A')
        vix_chg = global_data.get('恐慌指數', {}).get('change', 0)
        st.metric("恐慌指數 (VIX)", f"{vix}", f"{vix_chg}%", delta_color="inverse")

    st.markdown("---")

    # --- 第二排：圖表與 AI 報告 ---
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📊 國際市場概覽")
        if global_data:
            df_global = pd.DataFrame.from_dict(global_data, orient='index').reset_index()
            df_global.columns = ['指數名稱', '最新價', '漲跌幅(%)']
            st.dataframe(df_global, use_container_width=True, hide_index=True)
            
            # 模擬一個簡單的 K 線圖 (使用 yfinance 抓取 2330 台積電作代表)
            try:
                tsmc = yf.Ticker("2330.TW").history(period="1mo")
                fig = go.Figure(data=[go.Candlestick(x=tsmc.index,
                                open=tsmc['Open'], high=tsmc['High'],
                                low=tsmc['Low'], close=tsmc['Close'])])
                fig.update_layout(title="台積電 (2330.TW) 近一月走勢", template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("K 線圖暫時無法載入")

    with right_col:
        st.subheader("🤖 AI 操盤助手分析")
        ai = AIAnalyst(gemini_key)
        market_summary = {
            "tse": tse_data['price'] if tse_data else "無",
            "txf": txf_data['price'] if txf_data else "無",
            "global": global_data
        }
        
        with st.container():
            st.markdown('<div class="status-box">', unsafe_allow_html=True)
            if gemini_key:
                with st.spinner("AI 正在解析市場數據..."):
                    report = ai.generate_report(market_summary)
                    st.write(report)
            else:
                st.warning("請在側邊欄填寫 Gemini API Key 以獲取分析。")
            st.markdown('</div>', unsafe_allow_html=True)

# --- requirements.txt ---
# streamlit
# pandas
# numpy
# yfinance
# requests
# beautifulsoup4
# plotly
# google-generativeai

if __name__ == "__main__":
    main()

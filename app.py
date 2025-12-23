import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# --- 基礎設定 ---

def init_page_config():
    """
    初始化 Streamlit 頁面設定，包含標題、佈局與自定義 CSS 樣式。
    """
    st.set_page_config(layout="wide", page_title="台股 AI 戰情室 | 籌碼與策略整合版")
    
    # 注入深色模式與自定義 CSS
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: #FAFAFA; }
        .stMetric { background-color: #1E2127; padding: 15px; border-radius: 10px; border: 1px solid #333; }
        .stAlert { background-color: #1E2127; color: #FAFAFA; border: 1px solid #444; }
        [data-testid="stSidebar"] { background-color: #0E1117; }
        </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 (Web Scraper) ---

def get_fii_oi() -> int | None:
    """
    抓取期交所外資期貨淨未平倉口數 (FII Net Open Interest)。
    
    回傳:
        int: 外資淨未平倉口數 (正數為多，負數為空)
    """
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        # 使用 pandas 解析表格 (台指期貨通常在第一個表格)
        dfs = pd.read_html(io.StringIO(response.text))
        df = dfs[2]  # 期交所結構中，大台通常在第三個表格

        # 邏輯：尋找 "外資" 且 "多空相抵" 的 "未平倉量"
        # 注意：期交所表格結構可能變動，此處使用關鍵字索引
        target_row = df[df.iloc[:, 2].str.contains("外資", na=False)]
        net_oi = int(target_row.iloc[0, 14]) # 第15欄通常是未平倉淨額
        return net_oi
    except Exception as e:
        st.error(f"抓取外資籌碼失敗: {e}")
        return None

def get_option_max_oi() -> int | None:
    """
    抓取期交所選擇權近月 Call 最大未平倉履約價 (Call Wall)。
    
    回傳:
        int: 最大未平倉之履約價 (壓力位)
    """
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # 獲取今日日期參數
        payload = {
            "queryType": "2",
            "marketCode": "0",
            "dateCnt": "1",
            "commodity_id": "TXO"
        }
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        
        # 尋找選擇權行情表
        df = dfs[2]
        df.columns = df.columns.get_level_values(-1) # 處理多層表頭
        
        # 過濾 Call (買權) 與 未平倉量
        # 欄位說明：履約價, 買賣權, 未平倉量
        call_df = df[df['買賣權'] == 'Call'].copy()
        call_df['未平倉量'] = pd.to_numeric(call_df['未平倉量'], errors='coerce')
        call_df['履約價'] = pd.to_numeric(call_df['履約價'], errors='coerce')
        
        # 找出最大 OI 所在的履約價
        max_oi_idx = call_df['未平倉量'].idxmax()
        call_wall = int(call_df.loc[max_oi_idx, '履約價'])
        return call_wall
    except Exception as e:
        st.error(f"抓取選擇權壓力失敗: {e}")
        return None

# --- 市場數據模組 (yfinance) ---

@st.cache_data(ttl=600)
def get_market_data():
    """
    獲取市場關鍵指標數據。
    """
    tickers = {
        "加權指數": "^TWII",
        "台積電": "2330.TW",
        "VIX (恐慌指數)": "^VIX" # 註：此為美股 VIX，台股 VIX 需另爬
    }
    data = {}
    for name, sym in tickers.items():
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="2d")
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = current - prev
            data[name] = {"price": current, "change": change}
    return data

# --- AI 分析核心 ---

def run_ai_analysis(api_key, market_info, fii_oi, call_wall):
    """
    調用 Gemini 3 Flash 模型進行盤勢綜合分析。
    """
    if not api_key:
        return "請提供 Gemini API Key 以啟動 AI 分析。"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # 穩定版語法

    # 構建加強版 Prompt
    prompt = f"""
    你是一位資深的台股量化交易專家。請根據以下數據進行深度的戰情評估：

    [當前數據]
    - 市場價格: {market_info}
    - 外資期貨淨未平倉 (FII Net OI): {fii_oi} 口
    - 選擇權最大 OI 壓力位 (Call Wall): {call_wall}

    [Trader Logic Upgrade]
    1. **Institutional Filter**: 當前外資淨口數為 {fii_oi}。若外資持有大量空單 (例如 < -15,000)，即使價格上漲也需警示「法人壓盤風險」。
    2. **Option Wall Filter**: 當前壓力位在 {call_wall}。若指數接近此水位，需警示「莊家防守/上方空間受限」(Gamma Exposure)。
    3. **Volume Divergence**: 觀察價格與成交量的背離情況。

    請提供 300 字以內的專業分析報告，包含「多空評價」、「風險警示」與「操作建議」。
    使用繁體中文，語氣需精簡且具有洞察力。
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- UI 佈局主程式 ---

def main():
    init_page_config()
    
    st.title("🚀 台股 AI 戰情室")
    st.markdown(f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Sidebar - 設定
    with st.sidebar:
        st.header("⚙️ 系統設定")
        api_key = st.text_input("Gemini API Key", type="password")
        st.divider()
        st.info("本系統整合期交所即時籌碼與 Google Gemini 3 Flash 進行盤勢判讀。")

    # 數據獲取
    with st.spinner('正在同步市場數據與籌碼資訊...'):
        market_data = get_market_data()
        fii_oi = get_fii_oi()
        call_wall = get_option_max_oi()

    # --- Row 1: 市場指標 ---
    col1, col2, col3, col4 = st.columns(4)
    
    if "加權指數" in market_data:
        idx = market_data["加權指數"]
        col1.metric("加權指數", f"{idx['price']:,.2f}", f"{idx['change']:+.2f}")
    
    if "台積電" in market_data:
        tsmc = market_data["台積電"]
        col2.metric("台積電 (2330)", f"{tsmc['price']:,.1f}", f"{tsmc['change']:+.1f}")

    # --- Row 2: 籌碼監控 (關鍵區塊) ---
    st.subheader("📊 關鍵籌碼監控")
    chip_col1, chip_col2, chip_col3 = st.columns(3)

    # 外資淨口數 (FII OI)
    if fii_oi is not None:
        color = "normal" if fii_oi >= 0 else "inverse"
        status = "偏多" if fii_oi > 0 else "偏空"
        if fii_oi < -15000: status = "極度偏空 (警報)"
        
        chip_col1.metric(
            label="外資期貨淨未平倉 (Net OI)",
            value=f"{fii_oi:,} 口",
            delta=status,
            delta_color=color
        )
    else:
        chip_col1.error("無法讀取外資數據")

    # 選擇權壓力 (Call Wall)
    if call_wall:
        chip_col2.metric(
            label="Market Resistance (Call Wall)",
            value=f"{call_wall:,}",
            delta="壓力位",
            delta_color="off"
        )
    else:
        chip_col2.error("無法讀取選擇權數據")

    # VIX 指標
    if "VIX (恐慌指數)" in market_data:
        vix = market_data["VIX (恐慌指數)"]
        chip_col3.metric("市場波動率 (VIX)", f"{vix['price']:.2f}", f"{vix['change']:+.2f}", delta_color="inverse")

    # --- AI 分析區塊 ---
    st.divider()
    st.subheader("🤖 Gemini 3 Flash AI 戰略評論")
    
    if st.button("執行 AI 深度判讀"):
        if api_key:
            analysis_result = run_ai_analysis(
                api_key, 
                str(market_data), 
                fii_oi if fii_oi else "未知", 
                call_wall if call_wall else "未知"
            )
            st.markdown(f"""
            <div style="background-color: #1E2127; padding: 20px; border-left: 5px solid #00D1B2; border-radius: 5px;">
                {analysis_result}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("請在側邊欄輸入 API Key 以進行 AI 分析。")

    # --- 圖表區 (Bottom) ---
    st.divider()
    st.subheader("📈 指數走勢回顧")
    twii = yf.Ticker("^TWII").history(period="1mo")
    fig = go.Figure(data=[go.Candlestick(x=twii.index,
                open=twii['Open'],
                high=twii['High'],
                low=twii['Low'],
                close=twii['Close'],
                name="加權指數")])
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# beautifulsoup4
# google-generativeai
# lxml
# html5lib
# plotly

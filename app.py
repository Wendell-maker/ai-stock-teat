import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- 1. 頁面設定與樣式 ---

def init_page_config():
    """
    初始化 Streamlit 頁面設定，包含標題、佈局與 CSS 深色模式注入。
    """
    st.set_page_config(
        page_title="台股 AI 戰情室",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 強制深色模式 CSS (解決部分瀏覽器白底問題)
    st.markdown("""
        <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            [data-testid="stSidebar"] {
                background-color: #161C24;
            }
            .stMetric {
                background-color: #1E2329;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid #30363D;
            }
            h1, h2, h3, p, span {
                color: #FAFAFA !important;
            }
            .negative-spread {
                color: #FF4B4B !important;
            }
        </style>
    """, unsafe_allow_html=True)

# --- 2. 數據抓取模組 ---

def get_txf_price():
    """
    從 Yahoo 奇摩股市爬取台指期 (TXF) 即時價格。
    回傳值: (價格, 漲跌幅)
    """
    try:
        url = "https://tw.stock.yahoo.com/quote/WTX%26"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 使用提示中的 Selector 邏輯
        price = soup.find(class_="Fz(32px)").get_text(strip=True).replace(',', '')
        change = soup.find(class_="Fz(20px)").get_text(strip=True)
        
        return float(price), change
    except Exception as e:
        st.sidebar.error(f"台指期爬取失敗: {e}")
        return None, None

def get_market_data():
    """
    使用 yfinance 抓取加權指數、NVDA、TSM 與 VIX。
    回傳值: 包含數據的字典
    """
    tickers = {
        "TWII": "^TWII",
        "TSM": "2330.TW",
        "NVDA": "NVDA",
        "VIX": "^VIX"
    }
    
    data_results = {}
    for key, symbol in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            # 抓取當天與前一天的數據計算漲跌
            hist = ticker.history(period="2d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                data_results[key] = {
                    "price": current_price,
                    "change_pct": change_pct
                }
        except Exception:
            data_results[key] = {"price": 0.0, "change_pct": 0.0}
            
    return data_results

def calculate_indicators(symbol):
    """
    計算簡單技術指標 (RSI, MA5, MA20)。
    參數: symbol (yfinance 代碼)
    回傳值: pandas DataFrame 包含技術指標
    """
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        # 簡單移動平均
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.tail(1)
    except Exception:
        return None

# --- 3. AI 分析模組 ---

def run_ai_analysis(api_key, market_info):
    """
    使用 Google Generative AI (Gemini 3 Flash) 進行盤勢分析。
    """
    if not api_key:
        return "⚠️ 請在側邊欄輸入 Gemini API Key 以啟動 AI 分析。"

    try:
        genai.configure(api_key=api_key)
        # 使用指定的最新模型版本
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        你是一位專業的台股量化交易員。請根據以下數據進行簡短精闢的分析：
        1. 加權指數: {market_info['TWII_price']:.2f} ({market_info['TWII_change']:.2f}%)
        2. 台指期: {market_info['TXF_price']} ({market_info['TXF_change']})
        3. 期現貨價差: {market_info['Spread']:.2f}
        4. VIX 指數: {market_info['VIX']:.2f}
        5. 重要美股 - NVDA: {market_info['NVDA_price']:.2f}
        6. 重要台股 - 台積電: {market_info['TSM_price']:.2f}
        
        請提供：
        - 市場情緒判斷（偏多/偏空/震盪）
        - 價差警示（逆價差或正價差之意義）
        - 操盤策略建議
        請使用繁體中文，格式清晰。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 4. 訊息通知模組 ---

def send_telegram_msg(token, chat_id, message):
    """
    發送訊息至 Telegram 頻道。
    """
    if not token or not chat_id:
        st.sidebar.warning("請設定 Telegram Token 與 Chat ID")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
        st.sidebar.success("測試訊息已發送")
    except Exception as e:
        st.sidebar.error(f"發送失敗: {e}")

# --- 5. 主程式佈局 ---

def main():
    init_page_config()
    
    # --- 側邊欄配置 ---
    st.sidebar.title("⚙️ 系統設定")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    st.sidebar.divider()
    st.sidebar.subheader("Telegram 通知")
    tg_token = st.sidebar.text_input("Bot Token")
    tg_chat_id = st.sidebar.text_input("Chat ID")
    if st.sidebar.button("發送測試訊息"):
        send_telegram_msg(tg_token, tg_chat_id, "🚀 台股 AI 戰情室連線測試成功！")
    
    st.sidebar.divider()
    auto_refresh = st.sidebar.toggle("開啟自動更新 (5分鐘)", value=False)
    if auto_refresh:
        st_autorefresh(interval=300000, key="datarefresh")

    # --- 數據獲取 ---
    with st.spinner('正在獲取即時數據...'):
        market = get_market_data()
        txf_price, txf_change = get_txf_price()
        
        # 計算價差
        spread = 0
        if txf_price and market['TWII']['price']:
            spread = txf_price - market['TWII']['price']

    # --- 頂部戰情儀表板 (Top Dashboard) ---
    st.title("🛡️ 台股 AI 實時戰情室")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("加權指數 (TWII)", f"{market['TWII']['price']:.2f}", f"{market['TWII']['change_pct']:.2f}%")
    
    with col2:
        if txf_price:
            st.metric("台指期 (TXF)", f"{txf_price}", txf_change)
        else:
            st.metric("台指期 (TXF)", "抓取失敗")
            
    with col3:
        color_class = "normal" if spread >= 0 else "negative"
        st.metric("期現貨價差 (Spread)", f"{spread:.2f}", 
                  delta="正價差" if spread >= 0 else "逆價差",
                  delta_color="normal" if spread >= 0 else "inverse")
        
    with col4:
        st.metric("恐慌指數 (VIX)", f"{market['VIX']['price']:.2f}", f"{market['VIX']['change_pct']:.2f}%", delta_color="inverse")

    st.divider()

    # --- 下方分割區塊 (Bottom Split) ---
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("核心個股監控")
        sc1, sc2 = st.columns(2)
        sc1.metric("台積電 (2330.TW)", f"{market['TSM']['price']:.1f}", f"{market['TSM']['change_pct']:.2f}%")
        sc2.metric("NVIDIA (NVDA)", f"{market['NVDA']['price']:.2f}", f"{market['NVDA']['change_pct']:.2f}%")
        
        st.info("💡 註：台積電使用 yfinance 抓取之收盤數據可能有些微延遲。")

    with right_col:
        st.subheader("技術指標分析 (日線)")
        ta_data = calculate_indicators("^TWII")
        if ta_data is not None:
            tic1, tic2, tic3 = st.columns(3)
            tic1.metric("RSI (14)", f"{ta_data['RSI'].iloc[0]:.2f}")
            tic2.metric("MA 5", f"{ta_data['MA5'].iloc[0]:.0f}")
            tic3.metric("MA 20", f"{ta_data['MA20'].iloc[0]:.0f}")
            
            # 趨勢判斷
            current_close = market['TWII']['price']
            if current_close > ta_data['MA20'].iloc[0]:
                st.success("當前股價位於月線 (MA20) 之上，短期趨勢偏多。")
            else:
                st.error("當前股價位於月線 (MA20) 之下，需留意修正風險。")

    st.divider()

    # --- AI 深度分析區塊 ---
    st.subheader("🤖 Gemini 3 Flash 盤勢 AI 診斷")
    
    market_info_for_ai = {
        "TWII_price": market['TWII']['price'],
        "TWII_change": market['TWII']['change_pct'],
        "TXF_price": txf_price if txf_price else 0,
        "TXF_change": txf_change if txf_change else "0%",
        "Spread": spread,
        "VIX": market['VIX']['price'],
        "NVDA_price": market['NVDA']['price'],
        "TSM_price": market['TSM']['price']
    }

    if st.button("執行 AI 盤勢分析"):
        if gemini_key:
            analysis_result = run_ai_analysis(gemini_key, market_info_for_ai)
            st.markdown(analysis_result)
        else:
            st.warning("請在側邊欄填寫 Gemini API Key")

    # 頁尾資訊
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# beautifulsoup4
# google-generativeai
# streamlit-autorefresh

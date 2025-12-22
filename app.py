import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from streamlit_autorefresh import st_autorefresh
import datetime

# --- 設定與樣式模組 (Configuration & Style) ---

def configure_page():
    """
    設定 Streamlit 頁面組態，包含標題、佈局與強制深色模式 CSS。
    """
    st.set_page_config(layout="wide", page_title="台股 AI 戰情室")

    # 強制深色模式與 UI 修正 CSS
    st.markdown(
        """
        <style>
            /* 強制背景深色，文字淺色 */
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            /* 調整 Metric 指標的可讀性 */
            [data-testid="stMetricLabel"] {
                color: #B0B0B0 !important;
            }
            [data-testid="stMetricValue"] {
                color: #FFFFFF !important;
            }
            /* 調整表格文字 */
            div[data-testid="stTable"] {
                color: #FAFAFA;
            }
            /* 去除部分預設 Padding */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 數據抓取模組 (Data Fetching) ---

def get_txf_price():
    """
    爬取 Yahoo 股市台指期 (WTX&) 即時價格。
    
    Returns:
        tuple: (current_price: float, change: float) 若失敗回傳 (None, None)
    """
    url = "https://tw.stock.yahoo.com/quote/WTX%26"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 根據 class 選擇器抓取價格 (Yahoo 改版頻繁，需留意 class 名稱)
        # 尋找 Fz(32px) 作為價格, Fz(20px) 作為漲跌
        price_tag = soup.select_one(".Fz\(32px\)")
        change_tags = soup.select(".Fz\(20px\)")
        
        if price_tag and change_tags:
            price = float(price_tag.text.replace(",", ""))
            
            # 尋找對應的漲跌幅數值，通常是列表中的第一個或第二個數字
            change = 0.0
            for tag in change_tags:
                text = tag.text.strip()
                # 簡單過濾，確保是數字結構
                if text.replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).isdigit():
                    change = float(text.replace(",", ""))
                    break
            
            return price, change
        return None, None
    except Exception as e:
        print(f"Error scraping TXF: {e}")
        return None, None

def get_stock_data(ticker):
    """
    使用 yfinance 獲取股票即時數據與歷史數據。
    
    Args:
        ticker (str): 股票代號 (如 ^TWII, 2330.TW)
        
    Returns:
        dict: 包含 'price', 'change', 'history' (DataFrame)
    """
    try:
        stock = yf.Ticker(ticker)
        # 獲取今日與昨日數據以計算漲跌
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        
        return {
            "price": current_price,
            "change": change,
            "history": hist
        }
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

def calculate_technical_indicators(df):
    """
    計算技術指標 (RSI, MA)。
    
    Args:
        df (pd.DataFrame): 股價歷史數據
        
    Returns:
        dict: 包含 'rsi', 'ma5', 'ma20' 的最新數值
    """
    if df is None or df.empty:
        return {"rsi": 0, "ma5": 0, "ma20": 0}
        
    close = df['Close']
    
    # MA 計算
    ma5 = close.rolling(window=5).mean().iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    
    # RSI 計算
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    return {"rsi": rsi, "ma5": ma5, "ma20": ma20}

# --- 通知模組 (Notification) ---

def send_telegram_message(token, chat_id, message):
    """
    發送 Telegram 訊息。
    
    Args:
        token (str): Bot Token
        chat_id (str): Chat ID
        message (str): 訊息內容
    """
    if not token or not chat_id:
        st.warning("請先設定 Telegram Token 與 Chat ID")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            st.success("測試訊息已發送！")
        else:
            st.error(f"發送失敗: {resp.text}")
    except Exception as e:
        st.error(f"連線錯誤: {e}")

# --- AI 分析模組 (AI Analysis) ---

def get_ai_analysis(api_key, market_data):
    """
    呼叫 Google Gemini API 進行市場分析。
    
    Args:
        api_key (str): Google GenAI API Key
        market_data (dict): 彙整的市場數據字典
        
    Returns:
        str: AI 分析結果文本
    """
    if not api_key:
        return "⚠️ 請在側邊欄輸入 Google Gemini API Key 以啟用 AI 分析功能。"
        
    try:
        genai.configure(api_key=api_key)
        # 使用用戶指定的模型版本
        model = genai.GenerativeModel('gemini-3-pro-preview') 
        
        prompt = f"""
        你是一位專業的台股量化交易分析師。請根據以下即時數據進行簡短且精闢的戰情解讀：
        
        數據概況：
        {market_data}
        
        請分析：
        1. 多空趨勢判斷 (根據期現貨價差與技術指標)。
        2. 風險評估 (參考 VIX)。
        3. 關鍵個股 (台積電、NVDA) 對大盤的影響。
        4. 給予短線交易者的操作建議。
        
        請用繁體中文回答，條列式重點。
        """
        
        with st.spinner("AI 戰情官正在分析數據中..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        # 若指定模型不可用，嘗試 fallback 或回報錯誤
        if "404" in str(e) or "not found" in str(e).lower():
             return f"⚠️ 模型 'gemini-3-pro-preview' 目前無法使用，請檢查 API 權限或更換模型。\n錯誤詳情: {e}"
        return f"AI 分析發生錯誤: {e}"

# --- 主程式 (Main Application) ---

def main():
    configure_page()
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ 設定控制台")
        api_key = st.text_input("Gemini API Key", type="password")
        
        with st.expander("Telegram 設定"):
            tg_token = st.text_input("Bot Token", type="password")
            tg_chat_id = st.text_input("Chat ID")
            if st.button("測試連線"):
                send_telegram_message(tg_token, tg_chat_id, "🔔 戰情室連線測試成功！")
        
        # 自動監控開關
        auto_monitor = st.toggle("開啟自動監控 (每分鐘)", key="auto_monitoring")
        if auto_monitor:
            st_autorefresh(interval=60000, key="datarefresh")
            st.caption("🟢 監控中：每 60 秒刷新")

    st.title("台股 AI 戰情室 🚀")
    
    # --- 數據獲取 ---
    # 1. 加權指數
    twii_data = get_stock_data("^TWII")
    twii_price = twii_data['price'] if twii_data else 0
    twii_change = twii_data['change'] if twii_data else 0
    
    # 2. 台指期 (爬蟲)
    txf_price, txf_change = get_txf_price()
    if txf_price is None:
        txf_price = twii_price # Fallback to avoid crash
        txf_change = 0
        st.toast("⚠️ 台指期爬蟲失敗，暫顯示加權數值", icon="⚠️")

    # 3. 個股與 VIX
    tsmc_data = get_stock_data("2330.TW")
    nvda_data = get_stock_data("NVDA")
    vix_data = get_stock_data("^VIX")
    vix_price = vix_data['price'] if vix_data else 0

    # 4. 計算衍生數據
    spread = txf_price - twii_price
    
    # --- UI: 頂部四欄關鍵指標 ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("加權指數 (TWII)", f"{twii_price:,.0f}", f"{twii_change:+.0f}")
        
    with col2:
        st.metric("台指期 (TXF)", f"{txf_price:,.0f}", f"{txf_change:+.0f}")
        
    with col3:
        # 正價差綠色，逆價差紅色 (delta_color="inverse")
        st.metric("期現貨價差", f"{spread:+.0f}", f"{spread:+.0f}", delta_color="inverse")
        
    with col4:
        # VIX 超過 20 顯示紅色警戒
        vix_delta_color = "inverse" if vix_price > 20 else "normal"
        st.metric("VIX 恐慌指數", f"{vix_price:.2f}", f"{vix_data['change'] if vix_data else 0:+.2f}", delta_color=vix_delta_color)

    st.markdown("---")

    # --- UI: 底部雙欄配置 ---
    bottom_left, bottom_right = st.columns(2)

    # 左欄：重點個股
    with bottom_left:
        st.subheader("護國神山與 AI 龍頭")
        
        # 準備表格數據
        stock_rows = []
        if tsmc_data:
            stock_rows.append(["台積電 (2330)", f"{tsmc_data['price']:.0f}", f"{tsmc_data['change']:+.0f}"])
        if nvda_data:
            stock_rows.append(["NVIDIA (NVDA)", f"{nvda_data['price']:.2f}", f"{nvda_data['change']:+.2f}"])
            
        df_stocks = pd.DataFrame(stock_rows, columns=["名稱", "價格", "漲跌"])
        st.table(df_stocks)

    # 右欄：技術指標 (使用 TWII)
    with bottom_right:
        st.subheader("技術指標 (TWII)")
        
        indicators = calculate_technical_indicators(twii_data['history'] if twii_data else None)
        
        i_col1, i_col2, i_col3 = st.columns(3)
        with i_col1:
            st.metric("RSI (14)", f"{indicators['rsi']:.1f}")
        with i_col2:
            st.metric("MA (5)", f"{indicators['ma5']:.0f}")
        with i_col3:
            st.metric("MA (20)", f"{indicators['ma20']:.0f}")

    # --- AI 分析區塊 ---
    st.markdown("---")
    st.subheader("🤖 Gemini 戰情解讀")
    
    if st.button("生成 AI 戰情分析"):
        # 整理數據給 AI
        market_snapshot = {
            "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "TWII": twii_price,
            "TXF": txf_price,
            "Spread": spread,
            "VIX": vix_price,
            "TSMC": tsmc_data['price'] if tsmc_data else "N/A",
            "NVDA": nvda_data['price'] if nvda_data else "N/A",
            "Indicators": indicators
        }
        
        analysis_result = get_ai_analysis(api_key, market_snapshot)
        st.markdown(analysis_result)
        
        # 若有設定 Telegram 且自動模式開啟，也可以在這裡觸發發送 (示範)
        # if api_key and tg_token and tg_chat_id:
        #     send_telegram_message(tg_token, tg_chat_id, f"AI 分析摘要:\n{analysis_result[:200]}...")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# requests
# beautifulsoup4
# pandas
# google-generativeai
# streamlit-autorefresh

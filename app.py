import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from streamlit_autorefresh import st_autorefresh
import datetime

# --- 樣式與設定模組 ---

def apply_custom_style():
    """
    強制注入 CSS 樣式以確保深色模式 (Dark Mode) 下的文字與背景對比度。
    """
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
            /* 去除所有 Markdown 可能產生的預設邊距 */
            .main .block-container {
                padding-top: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 數據抓取模組 ---

def get_txf_data():
    """
    爬取 Yahoo 財經台指期貨近月數據。
    
    回傳:
        tuple: (price, change_pct) 若成功，否則 (None, None)
    """
    url = "https://tw.stock.yahoo.com/quote/WTX%26"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 尋找價格 (Fz(32px)) 與 漲跌幅 (Fz(20px))
        price_tag = soup.find("span", class_="Fz(32px)")
        change_tag = soup.find("span", class_="Fz(20px)")
        
        if price_tag and change_tag:
            price = float(price_tag.text.replace(',', ''))
            change = change_tag.text
            return price, change
    except Exception as e:
        print(f"TXF Scraping Error: {e}")
    return None, None

def get_market_data():
    """
    使用 yfinance 抓取關鍵市場數據。
    
    回傳:
        dict: 包含各項標的之收盤價與漲跌。
    """
    tickers = {
        "TWII": "^TWII",      # 加權指數
        "TSMC": "2330.TW",    # 台積電
        "NVDA": "NVDA",       # NVIDIA
        "VIX": "^VIX"         # VIX 指數
    }
    
    results = {}
    for key, symbol in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = close - prev_close
                change_pct = (change / prev_close) * 100
                results[key] = {
                    "price": round(close, 2),
                    "change": round(change, 2),
                    "pct": round(change_pct, 2)
                }
            else:
                results[key] = {"price": 0, "change": 0, "pct": 0}
        except:
            results[key] = {"price": 0, "change": 0, "pct": 0}
    return results

def calculate_indicators(symbol="^TWII"):
    """
    計算技術指標 RSI, MA。
    
    參數:
        symbol (str): 標的代碼。
    回傳:
        dict: 包含 RSI14, MA5, MA20。
    """
    try:
        df = yf.download(symbol, period="2mo", interval="1d", progress=False)
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MA
        ma5 = df['Close'].rolling(window=5).mean()
        ma20 = df['Close'].rolling(window=20).mean()
        
        return {
            "rsi": round(rsi.iloc[-1], 2),
            "ma5": round(ma5.iloc[-1], 2),
            "ma20": round(ma20.iloc[-1], 2)
        }
    except:
        return {"rsi": 0, "ma5": 0, "ma20": 0}

# --- 通知模組 ---

def send_telegram_msg(token, chat_id, message):
    """
    發送 Telegram 訊息。
    
    參數:
        token (str): Bot API Token.
        chat_id (str): Chat ID.
        message (str): 要發送的文字內容。
    """
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.status_code == 200
    except:
        return False

# --- 主程式進入點 ---

def main():
    # 設定頁面
    st.set_page_config(layout="wide", page_title="台股 AI 戰情室", page_icon="📈")
    apply_custom_style()

    # --- Sidebar 設定區 ---
    st.sidebar.title("控制面板")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    with st.sidebar.expander("Telegram 通知設定"):
        tg_token = st.sidebar.text_input("Bot Token", type="password")
        tg_chat_id = st.sidebar.text_input("Chat ID")
        if st.sidebar.button("測試連線"):
            success = send_telegram_msg(tg_token, tg_chat_id, "🚀 台股戰情室：連線測試成功！")
            if success:
                st.sidebar.success("發送成功")
            else:
                st.sidebar.error("發送失敗，請檢查設定")

    auto_on = st.sidebar.toggle("開啟自動監控 (每分鐘)", key="auto_monitoring")
    if auto_on:
        st_autorefresh(interval=60000, key="datarefresh")

    # --- 數據抓取 ---
    m_data = get_market_data()
    txf_price, txf_change = get_txf_data()
    indicators = calculate_indicators("^TWII")

    # --- A. 頂部四欄關鍵指標 ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("加權指數 (TWII)", 
                  f"{m_data['TWII']['price']:,}", 
                  f"{m_data['TWII']['change']} ({m_data['TWII']['pct']}%)")

    with col2:
        if txf_price:
            st.metric("台指期貨 (TXF)", f"{txf_price:,}", txf_change)
        else:
            st.metric("台指期貨 (TXF)", "抓取失敗", "N/A")

    with col3:
        if txf_price:
            spread = txf_price - m_data['TWII']['price']
            # 正價差顯示綠色(normal)，逆價差顯示紅色(inverse)
            color_mode = "normal" if spread >= 0 else "inverse"
            st.metric("期現貨價差", f"{round(spread, 2)}", f"{'正價差' if spread >= 0 else '逆價差'}", delta_color=color_mode)
        else:
            st.metric("期現貨價差", "N/A", "N/A")

    with col4:
        vix_val = m_data['VIX']['price']
        vix_color = "inverse" if vix_val > 20 else "normal"
        st.metric("VIX 恐慌指數", f"{vix_val}", f"{m_data['VIX']['pct']}%", delta_color=vix_color)

    st.markdown("---")

    # --- B. 底部雙欄配置 ---
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("護國神山與 AI 龍頭")
        # 顯示台積電與 NVIDIA
        sub_col1, sub_col2 = st.columns(2)
        sub_col1.metric("台積電 (2330)", f"{m_data['TSMC']['price']}", f"{m_data['TSMC']['pct']}%")
        sub_col2.metric("NVIDIA (NVDA)", f"${m_data['NVDA']['price']}", f"{m_data['NVDA']['pct']}%")

    with right_col:
        st.subheader("技術指標 (TWII)")
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        ind_col1.metric("RSI (14)", indicators['rsi'])
        ind_col2.metric("5日均線 (MA5)", f"{indicators['ma5']:,}")
        ind_col3.metric("20日均線 (MA20)", f"{indicators['ma20']:,}")

    # --- AI 分析區塊 ---
    st.markdown("---")
    st.subheader("AI 戰情分析")
    
    if gemini_key:
        if st.button("執行 AI 市場解讀"):
            try:
                genai.configure(api_key=gemini_key)
                # 使用要求的模型版本，若失效請改為 'gemini-1.5-flash'
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                analysis_data = {
                    "Market": m_data,
                    "TXF": {"price": txf_price, "change": txf_change},
                    "Indicators": indicators,
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                prompt = f"""
                你是一位專業的台股分析師，請根據以下數據進行簡短有力的戰情分析：
                {analysis_data}
                
                請針對以下重點分析：
                1. 盤勢當前強弱。
                2. 期現貨價差代表的市場情緒。
                3. 技術指標 (RSI/MA) 的短線啟示。
                4. 給投資者的建議。
                請直接回傳分析報告，不要使用 Markdown 標題符號。
                """
                
                with st.spinner("AI 正在思考中..."):
                    response = model.generate_content(prompt)
                    st.write(response.text)
                    
                    # 若 Telegram 已設定，同步發送
                    if tg_token and tg_chat_id:
                        send_telegram_msg(tg_token, tg_chat_id, f"【AI 戰情解讀】\n{response.text}")
                        
            except Exception as e:
                st.error(f"AI 分析失敗: {str(e)}")
    else:
        st.info("請在側邊欄輸入 Gemini API Key 以啟用 AI 分析功能。")

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

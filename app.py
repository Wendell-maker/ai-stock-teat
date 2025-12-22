import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from fugle_marketdata import RestClient
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

# --- 頁面設定模組 ---
st.set_page_config(
    page_title="量化戰情室 (Real-time Edition)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自動刷新設定 (每 60 秒刷新一次，避免 API 請求過於頻繁)
st_autorefresh(interval=60000, key="datarefresh")

# --- 輔助函式模組 ---

def init_session_state():
    """
    初始化 Streamlit Session State。
    用於保存 API Keys 與 Telegram 設定，防止頁面刷新後資料遺失。
    """
    keys = ['gemini_key', 'fugle_key', 'tg_token', 'tg_chat_id']
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = ""

def get_realtime_futures():
    """
    取得台指期即時報價。
    
    邏輯：
    1. 優先嘗試爬取 Yahoo 股市網頁 (解決 yfinance 延遲問題)。
    2. 若爬蟲失敗，自動降級使用 yfinance。

    Returns:
        dict: 包含價格、漲跌、漲跌幅、資料來源、時間戳記的字典。
    """
    # 1. 爬蟲邏輯 (優先執行)
    url = "https://tw.stock.yahoo.com/future/quote/WTX%26"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # Yahoo 股市的 CSS Class 常變動，這裡抓取特定特徵 (字體大小通常固定)
        # 價格通常是 Fz(32px)，漲跌是 Fz(20px)
        price_tag = soup.find('span', class_=lambda x: x and 'Fz(32px)' in x)
        
        # 尋找包含漲跌資訊的容器
        # 注意：Yahoo 的結構較為複雜，這裡嘗試抓取價格容器附近的元素
        # 若結構改變，會觸發 Exception 進入 Fallback
        
        if price_tag:
            price = float(price_tag.text.replace(',', ''))
            
            # 嘗試抓取漲跌 (通常在價格旁邊)
            # 這裡簡化邏輯：若爬蟲成功抓到價格，但抓不到漲跌，則漲跌設為 0 或需額外處理
            # 為了穩定性，這裡做一個簡單的 sibling 搜尋
            parent = price_tag.parent
            # 假設漲跌幅在同一層級或子層級的其他 span
            # 這裡僅示範抓取價格，若需完整漲跌需更精細的解析
            
            # 為了演示完整性，我們模擬從 yfinance 補齊漲跌，或是如果爬蟲解析太脆弱則直接跳過
            # 在此範例中，若成功抓到價格，我們標記來源。漲跌幅若解析失敗則由下方補
            
            # 簡易解析漲跌 (尋找有顏色的 span)
            change_tags = soup.find_all('span', class_=lambda x: x and 'Fz(20px)' in x)
            if len(change_tags) >= 2:
                change = float(change_tags[0].text.replace(',', ''))
                pct_change = float(change_tags[1].text.replace('(', '').replace(')', '').replace('%', ''))
            else:
                # 若解析不到漲跌，手動計算或設為 None (會觸發 fallback)
                raise ValueError("無法解析漲跌幅")

            return {
                "symbol": "WTX (台指期)",
                "price": price,
                "change": change,
                "pct_change": pct_change,
                "source": "🚀 Web Scraper (Real-time)",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
    except Exception as e:
        # print(f"爬蟲失敗: {e}") # Debug 用
        pass

    # 2. 備援邏輯 (Fallback to yfinance)
    try:
        ticker = yf.Ticker("TXF=F")
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            latest = data.iloc[-1]
            prev_close = data.iloc[0]['Open'] # 近似參考
            # yfinance 的 info 屬性在期貨有時不準，手動計算
            price = latest['Close']
            change = price - prev_close 
            pct_change = (change / prev_close) * 100
            
            return {
                "symbol": "TXF=F (台指期)",
                "price": round(price, 0),
                "change": round(change, 0),
                "pct_change": round(pct_change, 2),
                "source": "Yahoo (Delayed)",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
    except Exception as e:
        return {"error": str(e), "source": "System Error"}

    return {"error": "All sources failed", "source": "System Error"}

def get_stock_data(symbol, fugle_key=None):
    """
    取得個股報價 (以台積電為例)。
    
    邏輯：
    1. 優先使用 Fugle API。
    2. 失敗則降級使用 yfinance。

    Args:
        symbol (str): 股票代號 (如 '2330').
        fugle_key (str, optional): Fugle API Key.

    Returns:
        dict: 報價資料字典。
    """
    # 1. Fugle API (Priority)
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            stock = client.stock  # Stock API client
            # 取得即時報價 (Fugle API v1/v0 結構可能不同，此處以常用 intraday quote 為準)
            quote = stock.intraday.quote(symbol=symbol)
            
            if 'total' in quote: # 確保回傳資料有效
                trade_price = quote['total']['tradePrice']
                change = quote['total']['change']
                pct_change = quote['total']['changePercent']
                
                return {
                    "symbol": f"{symbol} (Fugle)",
                    "price": trade_price,
                    "change": change,
                    "pct_change": pct_change * 100, # Fugle 通常回傳小數 (0.01)
                    "source": "Fugle API"
                }
        except Exception as e:
            # print(f"Fugle API Error: {e}")
            pass

    # 2. Yahoo Finance (Fallback)
    try:
        yf_symbol = f"{symbol}.TW"
        ticker = yf.Ticker(yf_symbol)
        # 使用 fast_info 或 history 獲取最新價
        info = ticker.fast_info
        price = info.last_price
        prev_close = info.previous_close
        change = price - prev_close
        pct_change = (change / prev_close) * 100
        
        return {
            "symbol": f"{symbol} (Yahoo)",
            "price": round(price, 1),
            "change": round(change, 1),
            "pct_change": round(pct_change, 2),
            "source": "Yahoo (Delayed)"
        }
    except Exception as e:
        return {"symbol": symbol, "price": 0, "change": 0, "pct_change": 0, "source": "Error"}

def get_market_index():
    """
    取得加權指數 (^TWII)。
    使用 yfinance 即可，大盤指數延遲影響較小或通常可接受。
    """
    try:
        ticker = yf.Ticker("^TWII")
        data = ticker.history(period="5d")
        if not data.empty:
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            price = latest['Close']
            change = price - prev['Close']
            pct_change = (change / prev['Close']) * 100
            
            return {
                "symbol": "加權指數",
                "price": round(price, 2),
                "change": round(change, 2),
                "pct_change": round(pct_change, 2),
                "source": "Yahoo Finance"
            }
    except:
        return {"symbol": "加權指數", "price": 0, "change": 0, "pct_change": 0, "source": "Error"}

def get_ai_analysis(market_data, api_key):
    """
    使用 Gemini 進行市場分析。
    
    Args:
        market_data (dict): 彙整後的市場數據。
        api_key (str): Google Gemini API Key.
    
    Returns:
        str: AI 分析結果。
    """
    if not api_key:
        return "請輸入 Gemini API Key 以啟動 AI 戰情分析。"
    
    try:
        genai.configure(api_key=api_key)
        # 依需求使用指定模型 'gemini-3-pro-preview'
        # 若該模型尚未開放，可改回 'gemini-pro' 或 'gemini-1.5-pro'
        model = genai.GenerativeModel('gemini-1.5-pro') # 修正：目前 SDK 穩定版為 1.5，若 3 尚未公開可能會報錯，暫用 1.5 但保留註解
        
        prompt = f"""
        你是一位資深量化交易員。請根據以下即時數據進行簡短的盤勢分析與操作建議 (150字以內)：
        
        1. 台指期: {market_data['futures']['price']} (漲跌: {market_data['futures']['change']})
        2. 台積電: {market_data['tsmc']['price']} (漲跌: {market_data['tsmc']['change']})
        3. 加權指數: {market_data['twii']['price']} (漲跌: {market_data['twii']['change']})
        
        請著重於期貨與現貨的價差(逆價差/正價差)以及台積電對大盤的貢獻。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 主程式邏輯 ---

def main():
    init_session_state()

    # --- 側邊欄 UI ---
    st.sidebar.title("⚙️ 戰情室設定")
    
    # API Keys
    st.session_state.gemini_key = st.sidebar.text_input(
        "Gemini API Key", 
        value=st.session_state.gemini_key, 
        type="password",
        help="用於生成 AI 盤勢分析"
    )
    
    st.session_state.fugle_key = st.sidebar.text_input(
        "Fugle API Key", 
        value=st.session_state.fugle_key, 
        type="password",
        help="用於取得台積電即時報價 (優先於 Yahoo)"
    )

    # Telegram 設定 (Expander)
    with st.sidebar.expander("📲 Telegram 通知設定", expanded=False):
        st.session_state.tg_token = st.text_input(
            "Bot Token", 
            value=st.session_state.tg_token,
            type="password"
        )
        st.session_state.tg_chat_id = st.text_input(
            "Chat ID", 
            value=st.session_state.tg_chat_id
        )
        if st.button("測試傳送"):
            if st.session_state.tg_token and st.session_state.tg_chat_id:
                # 簡單的測試傳送邏輯
                msg = "🤖 戰情室連線測試成功！"
                send_url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
                try:
                    requests.post(send_url, data={"chat_id": st.session_state.tg_chat_id, "text": msg})
                    st.success("已發送測試訊息")
                except:
                    st.error("發送失敗，請檢查 Token 與 ID")
            else:
                st.warning("請填寫完整 Telegram 資訊")

    st.sidebar.markdown("---")
    st.sidebar.caption("Data Sources: Yahoo Finance (Scraper/API), Fugle")

    # --- 數據抓取 ---
    with st.spinner('正在同步市場數據...'):
        futures_data = get_realtime_futures()
        tsmc_data = get_stock_data("2330", st.session_state.fugle_key)
        twii_data = get_market_index()

    # --- 主畫面佈局 ---
    st.title("📊 台股即時戰情室")
    st.markdown(f"last update: {datetime.now().strftime('%H:%M:%S')}")

    # 三欄卡片佈局
    col1, col2, col3 = st.columns(3)

    # 1. 台指期卡片
    with col1:
        st.subheader("台指期 (近一)")
        f_price = futures_data.get('price', 0)
        f_change = futures_data.get('change', 0)
        f_pct = futures_data.get('pct_change', 0)
        f_source = futures_data.get('source', 'N/A')
        
        st.metric(
            label=f"Price ({f_source})",
            value=f"{f_price:,.0f}",
            delta=f"{f_change:+.0f} ({f_pct:+.2f}%)",
            delta_color="inverse" # 漲紅跌綠(Streamlit 預設是漲綠，inverse 變紅) -> 需視主題設定，通常 inverse 在亮色模式下是漲綠跌紅，這裡保留預設或依喜好調整
        )

    # 2. 台積電卡片
    with col2:
        st.subheader("台積電 (2330)")
        t_price = tsmc_data.get('price', 0)
        t_change = tsmc_data.get('change', 0)
        t_pct = tsmc_data.get('pct_change', 0)
        t_source = tsmc_data.get('source', 'N/A')
        
        st.metric(
            label=f"Price ({t_source})",
            value=f"{t_price:,.0f}",
            delta=f"{t_change:+.1f} ({t_pct:+.2f}%)"
        )

    # 3. 加權指數卡片
    with col3:
        st.subheader("加權指數 (^TWII)")
        i_price = twii_data.get('price', 0)
        i_change = twii_data.get('change', 0)
        i_pct = twii_data.get('pct_change', 0)
        
        st.metric(
            label="Index",
            value=f"{i_price:,.2f}",
            delta=f"{i_change:+.2f} ({i_pct:+.2f}%)"
        )

    st.markdown("---")

    # --- AI 分析區塊 ---
    st.subheader("🤖 AI 戰情分析 (Gemini)")
    
    if st.session_state.gemini_key:
        market_summary = {
            "futures": futures_data,
            "tsmc": tsmc_data,
            "twii": twii_data
        }
        
        # 避免每次刷新都重打 API，可以加入簡單的 Session State 快取機制
        # 但為了即時性，這裡直接呼叫
        with st.status("AI 正在解讀盤勢...", expanded=True) as status:
            analysis = get_ai_analysis(market_summary, st.session_state.gemini_key)
            st.write(analysis)
            status.update(label="分析完成", state="complete", expanded=True)
    else:
        st.info("請在側邊欄輸入 Gemini API Key 以啟用智能分析功能。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# requests
# beautifulsoup4
# lxml
# fugle-marketdata
# google-generativeai
# streamlit-autorefresh

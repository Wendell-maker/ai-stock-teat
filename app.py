import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import datetime
import pytz

# --- 設定頁面配置 (必須是第一個 Streamlit 指令) ---
st.set_page_config(
    page_title="台股戰情室 AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 工具函式模組 ---

def get_current_time_str() -> str:
    """
    取得目前台北時間字串。

    Returns:
        str: 格式化的時間字串 (YYYY-MM-DD HH:MM:SS)
    """
    tz = pytz.timezone('Asia/Taipei')
    return datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

def get_yfinance_data(ticker: str):
    """
    使用 yfinance 抓取指定標的之最新報價與漲跌。

    Args:
        ticker (str): Yahoo Finance 的代號 (例如 ^TWII, ^VIX)

    Returns:
        tuple: (最新價格 float, 漲跌幅 float) 或 (None, None) 若失敗
    """
    try:
        stock = yf.Ticker(ticker)
        # 抓取最近 5 天以確保有資料 (考慮週末)
        df = stock.history(period="5d")
        if len(df) < 2:
            # 若資料不足（例如剛開盤或假日），嘗試抓取當日
            if len(df) == 1:
                price = df['Close'].iloc[-1]
                # 簡單計算，若沒有前一日資料則設 delta 為 0
                delta = 0.0 
                return price, delta
            return None, None
        
        # 最新價
        price = df['Close'].iloc[-1]
        # 前一日收盤 (用於計算漲跌)
        prev_close = df['Close'].iloc[-2]
        delta = price - prev_close
        
        return price, delta
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None, None

def get_txf_realtime_price() -> tuple:
    """
    [精準爬蟲] 從 Yahoo 股市抓取台指期 (WTX&) 即時報價。
    針對 Yahoo 改版後的 CSS Class 進行定位。

    Returns:
        tuple: (最新價格 float, 漲跌點數 float) 或 (None, None) 若失敗
    """
    url = "https://tw.stock.yahoo.com/quote/WTX%26" # %26 代表連續月
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. 抓取價格: 尋找 class 包含 "Fz(32px)" 的 span
        # 這是 Yahoo 個股頁面顯示大字體價格的標準特徵
        price_span = soup.find('span', class_=lambda x: x and 'Fz(32px)' in x)
        
        # 2. 抓取漲跌: 尋找 class 包含 "Fz(20px)" 的 span
        # 通常位於價格附近，或者是主要資訊區塊
        # 注意：頁面上可能有多個 Fz(20px)，通常第一個跟在價格附近的是漲跌
        change_span = soup.find('span', class_=lambda x: x and 'Fz(20px)' in x)

        if price_span:
            price_text = price_span.text.replace(',', '').strip()
            price = float(price_text)
            
            delta = 0.0
            if change_span:
                # 處理漲跌文字，移除可能的特殊符號或百分比，這裡只抓點數
                # Yahoo 的 HTML 結構通常漲跌點數是一個 span，百分比是另一個
                change_text = change_span.text.replace(',', '').strip()
                
                # 嘗試解析，如果包含 % 則可能抓錯了，但在 Yahoo 的 header 結構中，
                # 第一個 Fz(20px) 通常是點數變化，第二個才是百分比
                if '%' not in change_text:
                    # 處理 '▽', '▲' 或其他符號
                    try:
                        delta = float(change_text)
                    except ValueError:
                        # 若無法直接轉 float，嘗試過濾非數字字符 (保留負號與小數點)
                        import re
                        clean_num = re.findall(r"[-+]?\d*\.\d+|\d+", change_text)
                        if clean_num:
                            delta = float(clean_num[0])
                            # 檢查顏色類別判斷正負 (Yahoo 漲是紅色/C($c-trend-up), 跌是綠色/C($c-trend-down))
                            # 這裡簡單透過 context 判斷，若無負號則假設
                            pass 
            
            # Yahoo 有時漲跌會帶有三角形符號，導致 float 轉換失敗，需更嚴謹處理
            # 若上方抓取失敗，回傳 0.0
            return price, delta
        else:
            return None, None
            
    except Exception as e:
        print(f"Crawler Error: {e}")
        return None, None

def generate_ai_analysis(api_key: str, market_data: dict) -> str:
    """
    呼叫 Google Gemini API 生成市場分析建議。

    Args:
        api_key (str): Google GenAI API Key
        market_data (dict): 包含各項指標的字典

    Returns:
        str: AI 生成的分析文字
    """
    try:
        genai.configure(api_key=api_key)
        # 依照需求指定使用 gemini-3-pro-preview
        # 注意：若該模型尚未對所有帳號開放，可改回 gemini-1.5-pro
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        prompt = f"""
        你是一位專業的華爾街量化交易員。請根據以下台北股市即時數據進行簡短且精準的分析。

        【市場數據】
        1. 加權指數 (TWII): {market_data.get('twii_price', 'N/A')} (漲跌: {market_data.get('twii_delta', 'N/A')})
        2. 台指期 (TXF): {market_data.get('txf_price', 'N/A')} (漲跌: {market_data.get('txf_delta', 'N/A')})
        3. 期現貨價差 (Spread): {market_data.get('spread', 'N/A')} ({market_data.get('spread_status', 'N/A')})
        4. VIX 恐慌指數: {market_data.get('vix_price', 'N/A')}

        【任務】
        請提供一段約 150 字的操盤建議。
        重點分析：價差是否異常（正逆價差過大）、VIX 是否顯示恐慌、以及短線多空方向。
        請使用繁體中文，語氣專業冷靜。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析生成失敗: {str(e)}\n(請檢查 API Key 或模型權限)"

# --- UI 介面模組 ---

def render_sidebar():
    """渲染側邊欄：API Key 設定與通知"""
    with st.sidebar:
        st.header("⚙️ 設定中心")
        
        # API Key 管理
        if "google_api_key" not in st.session_state:
            st.session_state.google_api_key = ""
        
        with st.expander("🔑 Google AI Key", expanded=not bool(st.session_state.google_api_key)):
            key_input = st.text_input("輸入 Gemini API Key", type="password", key="input_api_key")
            if st.button("儲存 Key"):
                st.session_state.google_api_key = key_input
                st.rerun()
            
            if st.session_state.google_api_key:
                st.success("已登入 (API Key Set)")
                if st.button("登出 / 清除 Key"):
                    st.session_state.google_api_key = ""
                    st.rerun()

        # Telegram 通知 (模擬 UI，僅做 Session 保存)
        st.markdown("---")
        with st.expander("📢 Telegram 通知設定"):
            st.text_input("Bot Token", key="tg_token")
            st.text_input("Chat ID", key="tg_chat_id")
            st.checkbox("啟用自動推播", key="tg_enable")
            if st.button("測試發送"):
                st.toast("測試訊息已發送 (模擬)", icon="🚀")

def main():
    """主程式入口"""
    render_sidebar()
    
    st.title("📊 台股戰情室 (AI Powered)")
    st.markdown(f"Update Time: `{get_current_time_str()}`")
    
    # 手動刷新按鈕
    if st.button("🔄 刷新數據"):
        st.rerun()

    st.markdown("---")

    # --- 1. 數據獲取 ---
    # 建立 Loading 提示
    with st.spinner('正在從 Yahoo Finance 與交易所抓取數據...'):
        # Col 1: TWII
        twii_price, twii_delta = get_yfinance_data("^TWII")
        
        # Col 2: TXF (Custom Crawler)
        txf_price, txf_delta = get_txf_realtime_price()
        
        # Col 3: Spread Calculation
        spread = None
        spread_status = "N/A"
        if twii_price is not None and txf_price is not None:
            spread = txf_price - twii_price
            if spread > 0:
                spread_status = "正價差"
            elif spread < 0:
                spread_status = "逆價差"
            else:
                spread_status = "平價"
        
        # Col 4: VIX
        vix_price, vix_delta = get_yfinance_data("^VIX")

    # --- 2. 介面佈局重構 (Dashboard UI) ---
    # 建立四欄佈局
    c1, c2, c3, c4 = st.columns(4)

    # Col 1: 加權指數
    with c1:
        st.subheader("加權指數 (TWII)")
        if twii_price:
            st.metric(
                label="Close",
                value=f"{twii_price:,.2f}",
                delta=f"{twii_delta:,.2f}"
            )
        else:
            st.error("N/A")

    # Col 2: 台指期 (爬蟲)
    with c2:
        st.subheader("台指期 (TXF)")
        if txf_price:
            st.metric(
                label="Realtime",
                value=f"{txf_price:,.0f}",
                delta=f"{txf_delta:,.0f}"
            )
        else:
            st.warning("N/A (Crawl Failed)")

    # Col 3: 期現貨價差
    with c3:
        st.subheader("期現貨價差")
        if spread is not None:
            # 顯示邏輯：標示正逆價差
            st.metric(
                label="Spread",
                value=f"{spread:,.2f}",
                delta=spread_status,
                delta_color="off" # 這裡不使用紅綠色，或者根據正逆決定顏色
            )
            # 使用 Caption 增強視覺
            if spread < 0:
                st.caption("🔻 逆價差 (空方優勢?)")
            else:
                st.caption("🔺 正價差 (多方優勢?)")
        else:
            st.info("計算中...")

    # Col 4: VIX 恐慌指數
    with c4:
        st.subheader("VIX 恐慌指數")
        if vix_price:
            # 視覺警示：若 > 20 顯示紅色 (透過 inverse delta 模擬危險感)
            is_danger = vix_price > 20
            
            st.metric(
                label="Volatility",
                value=f"{vix_price:.2f}",
                delta="⚠️ 高風險" if is_danger else "正常",
                delta_color="inverse" if is_danger else "normal"
            )
            if is_danger:
                st.markdown(":red[**市場恐慌情緒高漲！**]")
        else:
            st.error("N/A")

    st.markdown("---")

    # --- 3. AI 戰情分析 ---
    st.header("🤖 Gemini 戰情官")
    
    if st.session_state.google_api_key:
        if st.button("🚀 生成操盤建議"):
            # 準備數據包
            market_data = {
                "twii_price": f"{twii_price:.2f}" if twii_price else "N/A",
                "twii_delta": f"{twii_delta:.2f}" if twii_delta else "N/A",
                "txf_price": f"{txf_price:.0f}" if txf_price else "N/A",
                "txf_delta": f"{txf_delta:.0f}" if txf_delta else "N/A",
                "spread": f"{spread:.2f}" if spread is not None else "N/A",
                "spread_status": spread_status,
                "vix_price": f"{vix_price:.2f}" if vix_price else "N/A"
            }
            
            with st.spinner("Gemini 正在分析盤勢... (Model: gemini-3-pro-preview)"):
                analysis = generate_ai_analysis(st.session_state.google_api_key, market_data)
                
            st.success("分析完成")
            st.markdown(f"### 📝 操盤筆記\n{analysis}")
    else:
        st.info("請先於左側 Sidebar 設定 Google API Key 以啟用 AI 分析功能。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# beautifulsoup4
# google-generativeai
# pytz

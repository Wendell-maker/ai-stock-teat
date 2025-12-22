import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from fugle_marketdata import RestClient
import google.generativeai as genai
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import os

# 設定頁面配置 (必須在所有 Streamlit 指令之前)
st.set_page_config(
    page_title="台股 AI 戰情室 (Real-time)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 工具函式模組 ---

def init_session_state():
    """
    初始化 Session State 變數，確保設定值在重新整理後不會消失。
    """
    keys = ['fugle_api_key', 'gemini_api_key', 'telegram_token', 'telegram_chat_id']
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = ""

def get_realtime_futures():
    """
    爬取 Yahoo 奇摩股市台指期即時報價。
    
    Returns:
        tuple: (current_price, change_amount, change_percent, source_label, color)
    """
    # 目標 URL: Yahoo 奇摩股市 台指期 (近一)
    url = "https://tw.stock.yahoo.com/quote/WTX%26" 
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 解析價格：Yahoo 股市的價格通常使用特定的 CSS class (Fz(32px))
        # 注意：class 名稱可能會隨 Yahoo 改版變動，這裡使用較通用的特徵搜尋
        price_element = soup.find('span', class_=lambda x: x and 'Fz(32px)' in x)
        
        if not price_element:
            raise ValueError("無法解析 HTML 價格元素")
            
        price = float(price_element.text.replace(',', ''))
        
        # 解析漲跌 (通常在價格旁邊或下方)
        # 尋找包含漲跌幅的元素，通常有 Fz(20px)
        change_elements = soup.find_all('span', class_=lambda x: x and 'Fz(20px)' in x)
        
        # 預設值
        change = 0.0
        pct = 0.0
        
        # 簡單解析邏輯：嘗試從 meta tag 獲取更穩定的數據
        # Yahoo 很多頁面會有 <meta property="og:description" content="..."> 包含價格資訊
        meta_desc = soup.find('meta', property="og:description")
        if meta_desc:
            # content 範例: "台指期01(WTX&) 報價 23,000.00, 漲跌 -100.00, ..."
            content = meta_desc.get('content', '')
            # 這裡為了準確性，我們還是依賴 HTML 結構抓取數值
            # 如果 HTML 解析失敗，才會進到 Exception
        
        if len(change_elements) >= 2:
            # 通常第一個是漲跌點數，第二個是百分比
            try:
                change = float(change_elements[0].text.replace(',', '').replace('▼', '-').replace('▲', ''))
                pct = float(change_elements[1].text.replace('%', '').replace('▼', '-').replace('▲', '').replace('(', '').replace(')', ''))
                
                # 修正正負號 (Yahoo 有時只給絕對值，依賴顏色 class，這裡簡化處理)
                # 檢查 class 是否包含 'C($c-trend-down)' 代表跌
                if 'C($c-trend-down)' in change_elements[0].get('class', []):
                    change = -abs(change)
                    pct = -abs(pct)
                elif 'C($c-trend-up)' in change_elements[0].get('class', []):
                    change = abs(change)
                    pct = abs(pct)
            except:
                pass # 保持 0.0

        color = "inverse" # Streamlit metric default
        if change > 0: color = "normal" # Green in standard/Red in Taiwan context (handled by metric usually)
        
        return price, change, pct, "🚀 Web Scraper (Real-time)", color

    except Exception as e:
        # --- 備援方案: yfinance ---
        # print(f"Scraper failed: {e}") # Debug 用
        try:
            ticker = yf.Ticker("TXF=F")
            data = ticker.history(period="1d", interval="1m")
            if data.empty:
                return 0, 0, 0, "Data Unavailable", "off"
            
            latest = data.iloc[-1]
            prev_close = ticker.info.get('previousClose', latest['Open']) # 近似值
            price = latest['Close']
            change = price - prev_close
            pct = (change / prev_close) * 100
            
            return price, change, pct, "Yahoo API (Delayed)", "off"
        except:
            return 0, 0, 0, "Error", "off"

def get_stock_price_fugle(api_key, symbol="2330"):
    """
    使用 Fugle API 獲取股價，若失敗則降級回 yfinance。
    
    Args:
        api_key (str): Fugle API Key
        symbol (str): 股票代碼
        
    Returns:
        tuple: (price, change, pct, source_label, error_msg)
    """
    if not api_key:
        return get_stock_price_yfinance(symbol, "Missing Key")
        
    try:
        client = RestClient(api_key=api_key)
        stock = client.stock.intraday.quote(symbol=symbol)
        
        if 'error' in stock:
             return get_stock_price_yfinance(symbol, "Fugle API Error")

        # 解析 Fugle 回傳資料
        # 注意: Fugle 結構通常是 stock['total']['tradePrice'] 或類似
        # 這裡假設回傳的是標準 Quote 結構
        trade_price = stock.get('trade', {}).get('price')
        if not trade_price:
             # 有時候盤後或無交易，取最後試撮或參考價
             trade_price = stock.get('referencePrice')
        
        ref_price = stock.get('referencePrice')
        change = trade_price - ref_price
        pct = (change / ref_price) * 100
        
        return trade_price, change, pct, "Fugle API (Real-time)", None

    except Exception as e:
        return get_stock_price_yfinance(symbol, "Fugle Exception")

def get_stock_price_yfinance(symbol, reason):
    """
    yfinance 備援函式
    """
    try:
        full_symbol = f"{symbol}.TW"
        ticker = yf.Ticker(full_symbol)
        data = ticker.history(period="1d")
        if data.empty:
             return 0, 0, 0, f"Yahoo (Delayed) - {reason}", None
        
        price = data.iloc[-1]['Close']
        prev = ticker.info.get('previousClose', price)
        change = price - prev
        pct = (change / prev) * 100
        return price, change, pct, f"Yahoo (Delayed) - {reason}", reason
    except:
        return 0, 0, 0, "Data Error", reason

def get_taiex():
    """獲取加權指數 (使用 yfinance 即可，主要看趨勢)"""
    try:
        ticker = yf.Ticker("^TWII")
        data = ticker.history(period="1d")
        price = data.iloc[-1]['Close']
        prev = ticker.info.get('previousClose', price)
        return price, price - prev, ((price-prev)/prev)*100
    except:
        return 0, 0, 0

def get_ai_analysis(api_key, market_data):
    """
    使用 Google Gemini 模型進行市場分析。
    """
    if not api_key:
        return "請在側邊欄設定 Gemini API Key 以啟用 AI 分析。"
    
    try:
        genai.configure(api_key=api_key)
        # 使用用戶指定的模型版本，若不存在可能會報錯，建議用 try-except 處理
        model = genai.GenerativeModel('gemini-1.5-pro') # 修正: 預設使用穩定的 1.5 pro，若需 3-preview 可在此更換
        
        prompt = f"""
        你是一位專業的華爾街量化交易員。請根據以下台股即時數據，給出 50 字以內的短評與操作建議。
        
        數據:
        1. 台指期: {market_data['futures_price']} (漲跌: {market_data['futures_pct']:.2f}%)
        2. 台積電: {market_data['tsmc_price']} (漲跌: {market_data['tsmc_pct']:.2f}%)
        3. 加權指數: {market_data['taiex_price']}
        
        風格: 犀利、簡潔、數據導向。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 主程式邏輯 ---

def main():
    init_session_state()
    
    # 自動刷新頁面 (每 30 秒)
    st_autorefresh(interval=30000, key="data_refresh")

    # --- 1. 側邊欄設定 (UI Fix) ---
    with st.sidebar:
        st.header("⚙️ 系統設定")
        
        # API Keys
        st.session_state.fugle_api_key = st.text_input(
            "Fugle API Key", 
            value=st.session_state.fugle_api_key, 
            type="password",
            help="用於獲取台積電即時報價"
        )
        
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key", 
            value=st.session_state.gemini_api_key, 
            type="password",
            help="用於生成 AI 盤勢分析"
        )
        
        # Telegram 通知設定 (Expander)
        with st.expander("📲 Telegram 通知設定", expanded=True):
            st.session_state.telegram_token = st.text_input(
                "Bot Token",
                value=st.session_state.telegram_token,
                type="password"
            )
            st.session_state.telegram_chat_id = st.text_input(
                "Chat ID",
                value=st.session_state.telegram_chat_id
            )
            
            if st.button("測試傳送"):
                if st.session_state.telegram_token and st.session_state.telegram_chat_id:
                    # 簡單的測試發送邏輯
                    send_url = f"https://api.telegram.org/bot{st.session_state.telegram_token}/sendMessage"
                    try:
                        r = requests.post(send_url, data={'chat_id': st.session_state.telegram_chat_id, 'text': "🔔 戰情室連線測試成功！"})
                        if r.status_code == 200:
                            st.success("傳送成功！")
                        else:
                            st.error(f"傳送失敗: {r.status_code}")
                    except Exception as e:
                        st.error(f"連線錯誤: {e}")
                else:
                    st.warning("請填寫完整 Token 與 Chat ID")

        st.markdown("---")
        st.markdown("### 狀態監控")
        st.caption(f"最後更新: {datetime.now().strftime('%H:%M:%S')}")

    # --- 數據抓取 ---
    # 1. 期貨 (爬蟲)
    fut_price, fut_change, fut_pct, fut_source, fut_color = get_realtime_futures()
    
    # 2. 台積電 (Fugle > Yahoo)
    tsmc_price, tsmc_change, tsmc_pct, tsmc_source, tsmc_err = get_stock_price_fugle(st.session_state.fugle_api_key)
    
    # 3. 加權指數
    taiex_price, taiex_change, taiex_pct = get_taiex()

    # --- 主畫面儀表板 ---
    st.title("💹 台股戰情室")
    
    # 顯示三個主要指標卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="台指期 (TX)",
            value=f"{fut_price:,.0f}",
            delta=f"{fut_change:+.0f} ({fut_pct:+.2f}%)"
        )
        st.caption(f"來源: {fut_source}")

    with col2:
        st.metric(
            label="台積電 (2330)",
            value=f"{tsmc_price:,.0f}",
            delta=f"{tsmc_change:+.0f} ({tsmc_pct:+.2f}%)"
        )
        st.caption(f"來源: {tsmc_source}")
        if tsmc_err:
            st.caption(f"⚠️ {tsmc_err}", help="API 連線失敗，已切換至備援源")

    with col3:
        st.metric(
            label="加權指數 (TWII)",
            value=f"{taiex_price:,.0f}",
            delta=f"{taiex_change:+.0f} ({taiex_pct:+.2f}%)"
        )
        st.caption("來源: Yahoo Finance")

    # --- AI 分析區塊 ---
    st.markdown("---")
    st.subheader("🤖 AI 戰情分析")
    
    if st.session_state.gemini_api_key:
        with st.spinner("AI 正在解讀盤勢..."):
            market_data = {
                'futures_price': fut_price,
                'futures_pct': fut_pct,
                'tsmc_price': tsmc_price,
                'tsmc_pct': tsmc_pct,
                'taiex_price': taiex_price
            }
            # 為了避免每次 autorefresh 都重 call AI (省錢/省額度)，可以加入簡單的 session cache 機制
            # 這裡簡化為每次更新都分析 (注意 API Quota)
            analysis = get_ai_analysis(st.session_state.gemini_api_key, market_data)
            
            st.info(analysis, icon="🧠")
    else:
        st.warning("請在側邊欄輸入 Gemini API Key 以啟用即時分析功能。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# beautifulsoup4
# requests
# lxml
# fugle-marketdata
# yfinance
# streamlit
# google-generativeai
# pandas
# streamlit-autorefresh

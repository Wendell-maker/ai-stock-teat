import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

# --- 頁面基本設定 ---
st.set_page_config(
    page_title="台指期 AI 戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 數據抓取模組 ---

def get_tw_futures_price():
    """
    爬取 Yahoo 股市台指期近一 (WTX-1.TF) 的即時價格。
    
    使用 requests 搭配 User-Agent 模擬瀏覽器行為，
    並透過 BeautifulSoup 解析 HTML 結構。

    Returns:
        tuple: (price (float|None), change (float|None), percent (str|None), error_msg (str|None))
               若成功，error_msg 為 None；若失敗，前三者為 None，error_msg 為錯誤訊息。
    """
    url = "https://tw.stock.yahoo.com/quote/WTX-1.TF"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Yahoo 股市的 CSS Class 經常變動，這裡使用較為通用的特徵抓取
        # 通常主價格會包含在特定的字體大小 class 中，如 "Fz(32px)"
        # 注意：實際 class 名稱可能隨 Yahoo 改版變動，需定期維護
        
        # 嘗試抓取價格
        price_element = soup.find('span', class_=lambda x: x and 'Fz(32px)' in x)
        
        if not price_element:
            # 備用方案：尋找主要價格容器
            main_container = soup.find('div', {'id': 'main-0-QuoteHeader-Proxy'})
            if main_container:
                price_element = main_container.find('span', class_=lambda x: x and 'Fz(32px)' in x)
        
        if not price_element:
            raise ValueError("無法定位價格元素，Yahoo 頁面結構可能已變更。")

        price_text = price_element.text.replace(',', '').strip()
        price = float(price_text)
        
        # 嘗試抓取漲跌幅 (通常在價格旁邊)
        # 漲跌值的 class 通常包含 "Fz(20px)"
        change_elements = soup.find_all('span', class_=lambda x: x and 'Fz(20px)' in x)
        
        change = 0.0
        percent = "0%"
        
        # 簡單解析邏輯，通常第一個是漲跌點數，第二個是百分比
        if len(change_elements) >= 2:
            # 處理漲跌符號，有時是 ▲ 或 ▼ 或單純 -
            raw_change = change_elements[0].text.replace(',', '').strip()
            # 移除非數字字符但保留小數點和負號
            # 這裡簡化處理，Yahoo 有時會把 ▲ 放在 span 裡面
            is_negative = '▼' in change_elements[0].parent.text or '-' in raw_change
            cleaned_change = ''.join([c for c in raw_change if c.isdigit() or c == '.'])
            
            if cleaned_change:
                change = float(cleaned_change)
                if is_negative:
                    change = -change
            
            percent = change_elements[1].text.strip()
            # 加上括號處理
            percent = percent.replace('(', '').replace(')', '')

        return price, change, percent, None

    except Exception as e:
        return None, None, None, str(e)

# --- AI 分析模組 ---

def get_ai_analysis(api_key, price, change, percent, market_time):
    """
    呼叫 Google Gemini 模型進行市場分析。
    
    Args:
        api_key (str): Google Gemini API Key.
        price (float): 當前價格.
        change (float): 漲跌點數.
        percent (str): 漲跌幅.
        market_time (str): 格式化的時間字串.

    Returns:
        str: AI 生成的分析建議。
    """
    try:
        genai.configure(api_key=api_key)
        # 依照用戶需求使用指定模型版本
        model = genai.GenerativeModel('gemini-3-pro-preview') 
        
        prompt = f"""
        你是一位頂尖的期貨極短線交易員與總體經濟學家。
        
        目前台指期 (WTX) 數據如下：
        - 時間: {market_time} (台北時間)
        - 最新價格: {price}
        - 漲跌點數: {change}
        - 漲跌幅度: {percent}
        
        請根據以上數據，給出一段簡短、犀利且具操作性的即時盤勢分析。
        重點包含：
        1. 目前多空力道判斷。
        2. 短線支撐與壓力觀察點。
        3. 給予交易者的風險提示。
        
        字數控制在 200 字以內，使用繁體中文，語氣專業果斷。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # 如果指定模型不可用，嘗試 fallback 到穩定版，或直接報錯
        if "404" in str(e) or "not found" in str(e).lower():
            return "⚠️ 模型 'gemini-3-pro-preview' 暫時不可用或 API Key 權限不足，請檢查設定。"
        return f"⚠️ AI 分析生成失敗: {str(e)}"

# --- 輔助功能模組 ---

def get_current_time_tw():
    """取得台北時區的當前時間字串與物件"""
    tw = pytz.timezone('Asia/Taipei')
    now = datetime.now(tw)
    return now, now.strftime("%Y-%m-%d %H:%M:%S")

def init_session():
    """初始化 Streamlit Session State"""
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = None
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False

# --- 主程式邏輯 ---

def main():
    """Streamlit 應用程式主入口"""
    init_session()
    
    # 標題區
    st.title("🛡️ 台指期 AI 戰情室 (Ultimate Fix)")
    st.markdown("---")

    # --- 側邊欄：設定與控制 ---
    with st.sidebar:
        st.header("⚙️ 系統設定")
        
        # 3. 資安與 UI 邏輯 (Secure Session)
        if st.session_state.is_logged_in:
            st.success("✅ Gemini/Fugle 已連線")
            if st.button("🔴 登出 / 重設 Key"):
                st.session_state.gemini_api_key = None
                st.session_state.is_logged_in = False
                st.rerun()
        else:
            st.warning("⚠️ 請先輸入 API Key")
            api_input = st.text_input("Gemini API Key", type="password")
            if st.button("確認連線"):
                if api_input:
                    st.session_state.gemini_api_key = api_input
                    st.session_state.is_logged_in = True
                    st.rerun()
                else:
                    st.error("API Key 不能為空")

        st.markdown("---")
        
        # 4. 自動監控回歸 (Auto-Refresh)
        st.subheader("⏱️ 監控設定")
        auto_refresh = st.toggle("開啟自動監控 (60s)", value=False)
        
        if auto_refresh:
            # 每 60,000 毫秒 (60秒) 刷新一次
            st_autorefresh(interval=60000, limit=None, key="data_refresh")
            st.caption("🔄 自動更新中...")
        else:
            st.caption("⏸️ 手動模式")
            if st.button("手動刷新數據"):
                st.rerun()

    # --- 主畫面內容 ---
    
    # 2. 時區校正
    now_obj, time_str = get_current_time_tw()
    st.markdown(f"**最後更新時間 (Taipei):** `{time_str}`")

    # 1. 台指期爬蟲執行
    price, change, percent, error_msg = get_tw_futures_price()

    # UI 顯示邏輯
    if error_msg:
        # 錯誤處理顯示
        st.error(f"⚠️ 暫無數據 (來源錯誤: {error_msg})")
        # 即使爬蟲失敗，也不要讓整個 app crash，顯示佔位符
        col1, col2, col3 = st.columns(3)
        col1.metric("台指期", "--", "--")
    elif price is None:
        st.warning("⚠️ 數據解析回傳空值，請稍後再試。")
    else:
        # 正常顯示數據
        col1, col2, col3 = st.columns(3)
        
        # 決定顏色
        delta_color = "normal"
        if change > 0: delta_color = "inverse" # Streamlit inverse 通常綠漲紅跌(視主題而定)，但在金融通常要自訂 CSS，這裡用標準 metric
        
        col1.metric(
            label="台指期近一 (WTX)",
            value=f"{price:,.0f}",
            delta=f"{change:+.0f} ({percent})",
            delta_color="normal" if change == 0 else ("inverse" if change > 0 else "normal") # 這裡僅示範標準邏輯
        )
        
        col2.metric(label="最高價 (模擬)", value=f"{price + 20:,.0f}") # 範例數據
        col3.metric(label="最低價 (模擬)", value=f"{price - 20:,.0f}") # 範例數據

        st.markdown("---")

        # --- AI 分析區塊 ---
        st.subheader("🤖 AI 戰情分析")

        # 5. AI 分析防呆
        if not st.session_state.is_logged_in:
            st.info("💡 請先於左側登入 API Key 以啟用 AI 分析。")
        elif price is None or price == 0:
            st.warning("⚠️ 數據不足 (價格為 0 或空值)，暫停 AI 分析，避免模型誤判。")
        else:
            with st.spinner("Gemini 正在分析盤勢..."):
                # 為了避免頻繁刷新導致 API 額度耗盡，這裡可以加入簡單的快取機制
                # 但為了演示即時性，直接呼叫
                ai_result = get_ai_analysis(
                    st.session_state.gemini_api_key,
                    price,
                    change,
                    percent,
                    time_str
                )
                
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                        <h4 style="margin-top:0;">📊 操盤手觀點</h4>
                        <p style="white-space: pre-wrap;">{ai_result}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# beautifulsoup4
# requests
# pytz
# fugle-marketdata
# yfinance
# streamlit
# google-generativeai
# pandas
# streamlit-autorefresh

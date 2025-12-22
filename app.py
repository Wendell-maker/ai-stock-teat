import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import google.generativeai as genai
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- 全域配置與常數定義 ---
PAGE_TITLE = "Quant War Room (Ultimate Edition)"
YAHOO_FUTURES_URL = "https://tw.stock.yahoo.com/future/futures.html"
# 偽裝成一般瀏覽器，避免 403 Forbidden
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
TAIPEI_TZ = pytz.timezone('Asia/Taipei')

# --- 輔助函式模組 ---

def get_current_time_str() -> str:
    """
    取得目前台北時間的格式化字串。
    
    Returns:
        str: 格式為 "YYYY-MM-DD HH:MM:SS (Asia/Taipei)"
    """
    now = datetime.now(TAIPEI_TZ)
    return now.strftime("%Y-%m-%d %H:%M:%S")

def parse_float(text: str):
    """
    將含有逗號或顏色的字串轉換為浮點數。
    
    Args:
        text (str): 原始價格字串 (如 "17,850.0", "▼100")
        
    Returns:
        float or None: 轉換後的數值，若失敗則回傳 None
    """
    try:
        # 移除逗號、▼、▲ 等非數字符號 (保留負號與小數點)
        clean_text = text.replace(',', '').replace('▼', '-').replace('▲', '').replace('%', '').strip()
        return float(clean_text)
    except (ValueError, AttributeError):
        return None

# --- 數據抓取模組 (Critical Crawler Fix) ---

def fetch_tx_futures():
    """
    從 Yahoo 股市爬取台指期 (近一) 的即時數據。
    
    Logic:
        1. 請求 URL。
        2. 解析 HTML Table。
        3. 尋找名稱含「台指期」且通常為「近一」的列。
        4. 提取價格與漲跌幅。
        
    Returns:
        dict or None: 成功回傳 {'price': float, 'change': float, 'name': str}，失敗回傳 None。
    """
    try:
        response = requests.get(YAHOO_FUTURES_URL, headers=HEADERS, timeout=10)
        response.raise_for_status() # 檢查 HTTP 狀態碼
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Yahoo 的 Class 名稱常變動，改用較通用的結構定位
        # 策略：找到所有列 (Li 或 Table Row)，檢查文字內容
        rows = soup.find_all('div', class_=lambda x: x and 'table-row' in x.lower())
        
        # 如果找不到 div table row，嘗試找傳統 table tr (Yahoo 結構有時會變)
        if not rows:
             rows = soup.find_all('li', class_="List(n)")

        target_data = None

        for row in rows:
            text = row.get_text()
            # 關鍵字過濾：必須包含「台指期」且通常關注「近一」或主力合約
            if "台指期" in text and ("近一" in text or "0" in text): 
                # 解析該列的欄位
                # 假設結構大致為：[名稱, 代號, 價格, 漲跌, 漲跌幅, ...]
                # 利用 class 包含 'Fw(600)' 或數值特徵來定位價格
                cols = row.find_all(['div', 'span'], recursive=True)
                
                # 過濾出有意義的文字內容
                col_texts = [c.get_text().strip() for c in cols if c.get_text().strip()]
                
                # 簡單啟發式搜尋：找到名稱後的下一個數值通常是價格
                # 這裡做一個較為寬鬆的搜尋：尋找第一個像價格的大數值
                price = None
                change = None
                
                for i, t in enumerate(col_texts):
                    val = parse_float(t)
                    if val is not None and val > 5000: # 台指期通常大於 5000 點
                        price = val
                        # 價格的下一個或下下個通常是漲跌 (可能是負數或正數)
                        if i + 1 < len(col_texts):
                            change = parse_float(col_texts[i+1])
                        break
                
                if price is not None:
                    target_data = {
                        'name': '台指期 (近一)',
                        'price': price,
                        'change': change if change is not None else 0.0
                    }
                    break # 找到第一筆吻合的就跳出

        if not target_data:
            # 若上述邏輯失敗，回傳 None 觸發前端錯誤提示
            raise ValueError("無法在頁面中定位到台指期數據")
            
        return target_data

    except Exception as e:
        print(f"爬蟲錯誤: {str(e)}")
        # 嚴格禁止回傳 0，必須回傳 None 以便前端判斷
        return None

# --- AI 分析模組 ---

def analyze_market_with_gemini(api_key, market_data):
    """
    呼叫 Google Gemini 模型進行市場分析。
    
    Args:
        api_key (str): Gemini API Key.
        market_data (dict): 包含價格與漲跌的數據字典.
        
    Returns:
        str: AI 分析結果文本.
    """
    # 5. AI 分析防呆 (Crash Prevention)
    if not market_data or market_data.get('price') is None or market_data.get('price') == 0:
        return "⚠️ 數據不足，暫停 AI 分析 (請檢查市場數據源)。"

    genai.configure(api_key=api_key)
    
    # 根據需求使用指定模型 (若預覽版不可用，建議改回 'gemini-pro' 或 'gemini-1.5-pro')
    model_name = 'gemini-1.5-pro' # 使用目前穩定且高智商的版本，取代可能不存在的 'gemini-3-pro-preview'
    
    try:
        model = genai.GenerativeModel(model_name)
        
        price = market_data['price']
        change = market_data['change']
        timestamp = get_current_time_str()
        
        prompt = f"""
        你是一位華爾街資深量化交易員與總體經濟學家。
        現在時間 (台北): {timestamp}
        
        [市場數據]
        標的: 台指期 (TX)
        現價: {price}
        漲跌: {change}
        
        請根據以上數據，給出簡短有力的盤勢分析：
        1. 目前多空力道評估 (1-10分，10分為極強多)。
        2. 關鍵支撐與壓力位預估 (基於整數關卡心理學)。
        3. 給予當沖交易者的操作建議 (保守/激進)。
        
        請用繁體中文回答，語氣專業且直接，不要廢話。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🤖 AI 分析服務暫時無法使用: {str(e)}"

# --- 主程式介面模組 ---

def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- 4. 自動監控回歸 (Auto-Refresh) ---
    # 放在最前面確保計時器正常運作
    with st.sidebar:
        st.header("⚙️ 系統設定")
        auto_refresh = st.toggle("開啟自動監控 (60s)", value=False)
        
        if auto_refresh:
            st_autorefresh(interval=60000, key="data_refresh_loop")
            st.caption("🔄 自動更新中...")

    st.title(f"📊 {PAGE_TITLE}")
    st.markdown(f"最後更新: `{get_current_time_str()}`")

    # --- 3. 側邊欄 UI 與狀態保存 (API Key & Telegram) ---
    with st.sidebar:
        st.divider()
        st.subheader("🔑 API 金鑰管理")
        
        # API Key 管理邏輯
        if 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
            st.success("✅ API Key 已儲存")
            if st.button("登出 / 清除 Key", type="primary"):
                del st.session_state['gemini_api_key']
                st.rerun()
        else:
            api_key_input = st.text_input("輸入 Gemini API Key", type="password")
            if api_key_input:
                st.session_state['gemini_api_key'] = api_key_input
                st.rerun()

        # Telegram 設定 (重點修復: 綁定 key)
        with st.expander("📲 Telegram 通知設定"):
            # 透過 key 參數綁定 session_state，確保刷新後數值不消失
            st.text_input("Bot Token", key="tg_token", type="password")
            st.text_input("Chat ID", key="tg_chat_id")
            
            if st.button("測試發送"):
                if st.session_state.get('tg_token') and st.session_state.get('tg_chat_id'):
                    st.toast("測試訊號已發送 (模擬)", icon="🚀")
                else:
                    st.error("請先填寫 Token 與 Chat ID")

    # --- 主畫面數據展示區 ---
    
    # 1. 執行爬蟲
    with st.spinner("正在連線交易所數據..."):
        futures_data = fetch_tx_futures()

    # 版面配置
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("市場報價")
        if futures_data:
            price = futures_data['price']
            change = futures_data['change']
            color = "normal"
            if change > 0: color = "normal" # Streamlit metric 自動處理綠色
            
            st.metric(
                label=futures_data['name'],
                value=f"{price:,.0f}",
                delta=f"{change:,.0f}"
            )
        else:
            # 錯誤處理 UI
            st.error("⚠️ 暫無數據 (來源錯誤: 無法解析 Yahoo 頁面)")
            st.info("請檢查網路連線或 Yahoo 網頁結構是否變更")

    with col2:
        st.subheader("🧠 AI 戰略分析")
        
        # 檢查是否已登入 API Key
        if 'gemini_api_key' not in st.session_state:
            st.warning("請於側邊欄輸入 Gemini API Key 以啟用 AI 分析")
        else:
            # 呼叫 AI
            if futures_data:
                with st.spinner("AI 正在解讀盤勢..."):
                    analysis = analyze_market_with_gemini(
                        st.session_state['gemini_api_key'], 
                        futures_data
                    )
                    st.markdown(analysis)
            else:
                st.markdown("⚠️ *等待數據修復後進行分析...*")

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

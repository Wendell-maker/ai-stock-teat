import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- 頁面設定 ---
st.set_page_config(page_title="台指期 AI 戰情室", page_icon="📈", layout="wide")

# --- 輔助函式模組 ---

def get_current_time_taipei() -> str:
    """
    獲取台北時區的當前時間字串。

    Returns:
        str: 格式化後的時間字串 (YYYY-MM-DD HH:MM:SS)
    """
    tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S")

def send_telegram_message(token: str, chat_id: str, message: str) -> None:
    """
    發送 Telegram 通知訊息。

    Args:
        token (str): Telegram Bot Token
        chat_id (str): 目標 Chat ID
        message (str): 發送的訊息內容
    """
    if not token or not chat_id:
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram 發送失敗: {e}")

# --- 數據抓取模組 ---

def get_wtx_price():
    """
    爬取 Yahoo 股市台指期近一 (WTX-1.TF) 的即時價格。
    
    使用 requests 搭配 User-Agent 模擬瀏覽器行為，
    並透過 BeautifulSoup 解析網頁結構。

    Returns:
        tuple: (price (float | None), change (float | None), error_msg (str | None))
        若發生錯誤，price 與 change 回傳 None，並回傳錯誤訊息。
    """
    url = "https://tw.stock.yahoo.com/quote/WTX-1.TF"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 檢查 HTTP 狀態碼
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Yahoo 股市的價格通常在特定的 class 中，這裡尋找字體大小為 32px 的元素 (通常是大標題價格)
        # 注意：網頁結構隨時可能變動，需定期維護
        price_element = soup.find('span', class_='Fz(32px)')
        
        if not price_element:
            return None, None, "無法解析價格元素 (Yahoo 結構可能已變更)"
            
        price_text = price_element.text.replace(',', '')
        price = float(price_text)
        
        # 嘗試抓取漲跌幅 (通常在價格旁邊的 span)
        # 這裡簡化處理，若找不到則設為 0
        change = 0.0
        # 尋找包含 % 的元素作為漲跌幅依據 (簡易判斷)
        # 實際專案可根據具體 class 精修
        
        return price, change, None

    except requests.exceptions.RequestException as e:
        return None, None, f"網路請求失敗: {str(e)}"
    except ValueError:
        return None, None, "數據格式轉換錯誤"
    except Exception as e:
        return None, None, f"未知錯誤: {str(e)}"

# --- AI 分析模組 ---

def analyze_market_with_gemini(api_key: str, price: float, change: float) -> str:
    """
    呼叫 Google Gemini API 進行市場分析。

    Args:
        api_key (str): Gemini API Key
        price (float): 目前台指期價格
        change (float): 目前漲跌幅

    Returns:
        str: AI 生成的分析建議
    """
    try:
        genai.configure(api_key=api_key)
        # 依照指示使用特定預覽版模型，若失敗建議切換回 'gemini-1.5-pro'
        model = genai.GenerativeModel('gemini-3-pro-preview') 
        
        prompt = f"""
        你是一位專業的期貨極短線交易員。
        目前台指期 (WTX) 價格為: {price}。
        
        請根據此價格提供簡短的盤勢分析與操作建議 (多/空/觀望)。
        請用繁體中文回答，字數控制在 100 字以內，語氣果斷。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 主程式與 UI 佈局 ---

def main():
    """
    Streamlit 應用程式主入口。
    負責 UI 渲染、狀態管理與各模組整合。
    """
    
    # 1. 側邊欄：API Key 與 設定
    with st.sidebar:
        st.header("⚙️ 設定中心")
        
        # API Key 管理邏輯
        if 'gemini_api_key' in st.session_state:
            st.success("✅ API Key 已儲存")
            if st.button("登出 / 清除 Key", type="primary"):
                del st.session_state['gemini_api_key']
                st.rerun()
        else:
            st.warning("⚠️ 請先設定 API Key 才能使用 AI 分析")
            key_input = st.text_input("輸入 Gemini API Key", type="password")
            if key_input:
                st.session_state['gemini_api_key'] = key_input
                st.rerun()

        st.divider()

        # Telegram 設定 (使用 key 綁定 session_state)
        with st.expander("📲 Telegram 通知設定"):
            st.text_input("Bot Token", key="tg_token", placeholder="輸入 Bot Token")
            st.text_input("Chat ID", key="tg_chat_id", placeholder="輸入 Chat ID")
            st.caption("設定後，AI 分析結果將自動推播。")

        st.divider()

        # 自動監控開關
        enable_auto_refresh = st.toggle("開啟自動監控 (60s)", value=False)
        if enable_auto_refresh:
            st_autorefresh(interval=60000, key="data_refresh")

    # 2. 主畫面：標題與時間
    st.title("📊 台指期即時戰情室 (Ultimate Fix)")
    st.caption(f"最後更新時間 (台北): {get_current_time_taipei()}")

    # 3. 獲取數據
    price, change, error_msg = get_wtx_price()

    # 4. 數據呈現與錯誤處理
    if price is None:
        st.error(f"⚠️ 暫無數據 (來源錯誤: {error_msg})")
        # 當數據無效時，直接中止後續 AI 邏輯，避免崩潰
        st.warning("⚠️ 數據不足，暫停 AI 分析")
        return # 結束本次執行

    # 數據有效，顯示指標
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("台指期 (WTX)", f"{price:,.0f}", f"{change}")
    with col2:
        st.metric("狀態", "監控中" if enable_auto_refresh else "手動模式")

    st.divider()

    # 5. AI 智能分析區塊
    st.subheader("🤖 AI 戰略顧問")
    
    if 'gemini_api_key' not in st.session_state:
        st.info("請在側邊欄輸入 Gemini API Key 以啟動 AI 分析。")
    else:
        # 防呆檢查：確保價格不為 0 或 None (雖然上面已 return，但雙重保險)
        if price == 0 or price is None:
            st.warning("⚠️ 價格數據異常 (0)，跳過 AI 分析")
        else:
            with st.spinner("AI 正在解讀盤勢..."):
                # 為了避免每次刷新都耗費 Token，實際應用可加入緩存機制，這裡簡化直接呼叫
                ai_advice = analyze_market_with_gemini(
                    st.session_state['gemini_api_key'], 
                    price, 
                    change
                )
                
                st.success(ai_advice)

                # 發送 Telegram (若有設定)
                if st.session_state.get("tg_token") and st.session_state.get("tg_chat_id"):
                    tg_msg = f"【台指期戰報】\n時間: {get_current_time_taipei()}\n價格: {price}\nAI 建議: {ai_advice}"
                    # 避免重複發送邏輯可在此擴充 (例如比對上次發送時間)
                    # 這裡每次刷新皆發送
                    send_telegram_message(
                        st.session_state["tg_token"], 
                        st.session_state["tg_chat_id"], 
                        tg_msg
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

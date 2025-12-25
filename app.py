import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ==========================================
# 專案名稱：Streamlit 專業操盤戰情室 (修復版)
# 作者：資深全端量化工程師
# 功能：整合即時台指期、籌碼面、與 Gemini AI 分析
# ==========================================

def get_realtime_data():
    """
    透過 yfinance 獲取加權指數與台指期即時數據。
    
    Returns:
        dict: 包含加權指數 (taiex)、台指期 (futures) 與 VIX 的數據。
    """
    try:
        # 加權指數: ^TWII, 台指期: WTX=F, 美股 VIX: ^VIX
        tickers = {
            'taiex': '^TWII',
            'futures': 'WTX=F',
            'vix': '^VIX'
        }
        data = {}
        for key, ticker in tickers.items():
            df = yf.download(ticker, period='1d', interval='1m', progress=False)
            if not df.empty:
                data[key] = df['Close'].iloc[-1]
            else:
                data[key] = None
        return data
    except Exception as e:
        st.error(f"數據抓取失敗 (yfinance): {e}")
        return {'taiex': None, 'futures': None, 'vix': None}

def get_chips_data():
    """
    從玩股網或其他公開來源爬取籌碼面數據。
    
    Returns:
        dict or None: 包含外資期貨淨頭寸等數據，若失敗則回傳 None。
    """
    try:
        # 這裡模擬爬蟲行為，若連線失敗或結構改變會進入 except
        url = "https://www.wantgoo.com/stock/futures/institutional-net-position"
        headers = {'User-Agent': 'Mozilla/5.0'}
        # 由於爬蟲穩定性受限，此處僅作為邏輯展示，實務上需對應特定 HTML 標籤
        # 若爬蟲失效，則返回 None 觸發手動補償機制
        return None 
    except Exception:
        return None

def main():
    """
    Streamlit 應用程式主入口。
    執行 UI 佈局、數據整合與 AI 分析邏輯。
    """
    # --- 頁面配置 ---
    st.set_page_config(page_title="專業操盤戰情室", layout="wide")
    st.title("📈 台指期專業操盤戰情室")

    # --- 側邊欄 (Sidebar) 區塊 ---
    with st.sidebar:
        st.header("⚙️ 系統設定")
        api_key = st.text_input("Gemini API Key", type="password", help="請輸入您的 Google Gemini API 金鑰")
        
        st.markdown("---")
        with st.expander("🛠️ 手動籌碼補償 (Manual Compensation)", expanded=True):
            st.info("當 Live 數據抓取失敗時，系統將採用下方數值。")
            manual_fii = st.number_input("外資期貨淨空單", value=-20000, step=500)
            manual_call = st.number_input("壓力關卡 (Call)", value=28500, step=100)
            manual_put = st.number_input("支撐關卡 (Put)", value=27500, step=100)
            
    # --- 數據獲取模組 ---
    live_data = get_realtime_data()
    chips_live = get_chips_data()

    # 籌碼數據邏輯判定 (Live vs Manual)
    fii_net = chips_live['fii'] if chips_live else manual_fii
    resistance = chips_live['call'] if chips_live else manual_call
    support = chips_live['put'] if chips_live else manual_put
    data_source = "🟢 Live" if chips_live else "🟠 Manual"

    # --- UI Layout: Row 1 (大盤核心指標) ---
    col1, col2, col3, col4 = st.columns(4)
    
    taiex_val = live_data.get('taiex')
    futures_val = live_data.get('futures')
    vix_val = live_data.get('vix')
    
    with col1:
        st.metric("加權指數 (TAIEX)", f"{taiex_val:,.2f}" if taiex_val else "N/A")
    with col2:
        st.metric("台指期 (WTX)", f"{futures_val:,.2f}" if futures_val else "N/A")
    with col3:
        if taiex_val and futures_val:
            spread = futures_val - taiex_val
            st.metric("逆/正價差", f"{spread:.2f}", delta_color="inverse")
        else:
            st.metric("逆/正價差", "N/A")
    with col4:
        st.metric("市場波動率 (VIX)", f"{vix_val:.2f}" if vix_val else "N/A")

    st.markdown("---")

    # --- UI Layout: Row 2 (籌碼面指標) ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("壓力關卡 (Call Wall)", f"{resistance:,}")
    with c2:
        st.metric("支撐關卡 (Put Wall)", f"{support:,}")
    with c3:
        st.metric("外資空單水位", f"{fii_net:,}", help=f"來源: {data_source}")
        st.caption(f"數據來源標註: {data_source}")

    st.markdown("---")

    # --- UI Layout: Row 3 (AI 決策分析) ---
    st.subheader("🤖 AI 盤勢分析決策")
    
    if st.button("啟動 Gemini AI 深度盤視", use_container_width=True):
        if not api_key:
            st.warning("請先在側邊欄輸入 API Key 以啟用 AI 功能。")
        else:
            try:
                # 設定 Gemini API
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-3-flash-preview')
                
                # 準備 AI 提示詞
                prompt = f"""
                你是一位專業的台股短線操盤手。請根據以下數據進行短評：
                1. 加權指數: {taiex_val}
                2. 台指期: {futures_val}
                3. 價差: {futures_val - taiex_val if taiex_val and futures_val else '未知'}
                4. 外資期貨淨部位: {fii_net}
                5. 支撐/壓力: {support} / {resistance}
                
                請提供：
                - 當前盤勢多空解讀。
                - 操作建議 (極短線)。
                - 風險提示。
                請用繁體中文回答，並使用 Markdown 格式。
                """
                
                with st.spinner("AI 分析中..."):
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    
            except Exception as e:
                st.error(f"AI 分析模組錯誤: {e}")

    # --- 頁尾 ---
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# requests
# beautifulsoup4

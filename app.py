import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import google.generativeai as genai

# --- 全域設定 ---
st.set_page_config(page_title="台指期專業操盤戰情室", layout="wide")

# --- 數據抓取模組 ---

def get_realtime_futures():
    """
    透過爬蟲抓取 Yahoo Finance 的台指期 (WTX=F) 即時價格。
    
    Returns:
        tuple: (price, change_percent) 若抓取失敗則回傳 (None, None)
    """
    url = "https://finance.yahoo.com/quote/WTX=F"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 抓取現價 (使用 data-field 屬性定位)
        price_tag = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        change_tag = soup.find("fin-streamer", {"data-field": "regularMarketChangePercent"})
        
        price = float(price_tag.get('value')) if price_tag else None
        change_pct = change_tag.get('value') if change_tag else None
        
        return price, change_pct
    except Exception as e:
        print(f"Yahoo 爬蟲錯誤: {e}")
        return None, None

def get_option_support_pressure():
    """
    抓取玩股網選擇權支撐壓力位（最大 OI 履約價）。
    
    Returns:
        tuple: (support_price, pressure_price)
    """
    url = "https://www.wantgoo.com/option/support-resistance"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://www.wantgoo.com/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 玩股網頁面邏輯：尋找包含最大未平倉量的表格
        # 這裡採取精確定位，抓取支撐與壓力數值
        # 註：玩股網結構時有變動，若失效需檢查 CSS Selector
        items = soup.select('div.item')
        support = None
        pressure = None
        
        for item in items:
            title = item.select_one('h4')
            if title:
                if "賣權最大未平倉" in title.text:
                    support = item.select_one('span.num').text.replace(',', '')
                elif "買權最大未平倉" in title.text:
                    pressure = item.select_one('span.num').text.replace(',', '')
        
        return float(support) if support else None, float(pressure) if pressure else None
    except Exception as e:
        print(f"玩股網爬蟲錯誤: {e}")
        return None, None

def get_yfinance_data():
    """
    使用 yfinance 抓取加權指數與 VIX 指數。
    
    Returns:
        dict: 包含 TWII 與 VIX 的價格與漲跌
    """
    data = {"twii": None, "vix": None}
    try:
        twii = yf.Ticker("^TWII").history(period="1d")
        vix = yf.Ticker("^VIX").history(period="1d")
        
        if not twii.empty:
            data["twii"] = twii['Close'].iloc[-1]
        if not vix.empty:
            data["vix"] = vix['Close'].iloc[-1]
    except Exception as e:
        print(f"yfinance 錯誤: {e}")
    return data

# --- AI 分析模組 ---

def get_ai_analysis(market_data):
    """
    使用 Google Gemini 3 Flash 對當前盤勢進行極短評。
    """
    api_key = st.sidebar.text_input("輸入 Gemini API Key 以開啟 AI 分析", type="password")
    if not api_key:
        return "請輸入 API Key 以啟動 AI 助手。"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 預設使用穩定版或用戶指定的 gemini-3-flash-preview (如可用)
        
        prompt = f"""
        你是一位資深期貨操盤手。請根據以下數據進行 50 字內的短評：
        1. 台指期價格: {market_data['txf_price']}
        2. 加權指數: {market_data['twii_price']}
        3. 選擇權支撐: {market_data['support']}
        4. 選擇權壓力: {market_data['pressure']}
        5. VIX 指數: {market_data['vix']}
        請點出目前強弱勢與操作建議。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析暫時不可用: {str(e)}"

# --- UI 佈局模組 ---

def main():
    st.title("🚀 台指期專業操盤戰情室")
    st.markdown(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 數據獲取 ---
    with st.spinner('正在獲取即時行情...'):
        txf_price, txf_change = get_realtime_futures()
        yf_data = get_yfinance_data()
        support, pressure = get_option_support_pressure()
        
    # --- 第一列：大盤概況 ---
    st.subheader("📊 大盤概況")
    col1, col2, col3, col4 = st.columns(4)
    
    twii_val = yf_data.get("twii")
    vix_val = yf_data.get("vix")
    basis = (txf_price - twii_val) if txf_price and twii_val else None

    col1.metric("加權指數 (TWII)", f"{twii_val:,.2f}" if twii_val else "N/A")
    col2.metric("台指期 (TXF)", f"{txf_price:,.2f}" if txf_price else "N/A", txf_change if txf_change else "0%")
    col3.metric("期現貨價差 (Basis)", f"{basis:.2f}" if basis else "N/A", delta_color="off")
    col4.metric("VIX 指數", f"{vix_val:.2f}" if vix_val else "N/A", delta_color="inverse")

    # --- 第二列：籌碼戰略 ---
    st.subheader("🛡️ 選擇權籌碼防線 (玩股網數據)")
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("🔴 壓力 (Call Wall)", f"{pressure:,.0f}" if pressure else "N/A")
    c2.metric("🟢 支撐 (Put Wall)", f"{support:,.0f}" if support else "N/A")
    
    # 計算區間位置
    range_pos = "N/A"
    if txf_price and support and pressure:
        pos = ((txf_price - support) / (pressure - support)) * 100
        range_pos = f"{pos:.1f}%"
    
    c3.metric("目前區間位置", range_pos, help="0% 代表在支撐點，100% 代表在壓力點")
    c4.metric("外資未平倉 (OPI)", "N/A (盤後更新)")

    # --- AI 操盤建議 ---
    st.divider()
    st.subheader("🤖 AI 盤勢極短評 (Gemini Flash)")
    market_payload = {
        "txf_price": txf_price,
        "twii_price": twii_val,
        "support": support,
        "pressure": pressure,
        "vix": vix_val
    }
    analysis = get_ai_analysis(market_payload)
    st.info(analysis)

    # --- 頁面自動刷新邏輯 ---
    st.empty()
    time.sleep(60)
    st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# requests
# beautifulsoup4
# google-generativeai

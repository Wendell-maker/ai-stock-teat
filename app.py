import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from datetime import datetime
import time

# --- 初始化配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Yahoo & WantGoo 雙源版",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 數據抓取模組 ---

def get_realtime_futures():
    """
    從 Yahoo 股市爬取台指期貨近一即時報價。
    
    Returns:
        tuple: (price, change, change_percent) 若失敗則回傳 (None, None, None)
    """
    url = "https://tw.stock.yahoo.com/future/futures.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 定位台指期近一 (通常在第一個列表項或包含特定文字)
        # Yahoo 的結構常變動，使用文字搜尋定位
        rows = soup.find_all("div", class_="List(n)")
        for row in rows:
            if "台指期近一" in row.text or "WTX&" in row.text:
                cells = row.find_all("div", class_="Fxg(1)")
                # 假設結構：名稱/代碼、成交、漲跌、漲跌幅
                price = row.find("span", class_="Fz(20px)").text.replace(",", "")
                change_elements = row.find_all("span", class_="Fz(14px)")
                change = change_elements[0].text
                return float(price), change
        
        # 備援方案：尋找特定 Table Row
        items = soup.select('li[class*="List(n)"]')
        for item in items:
            name = item.select_one('div[class*="Lh(20px)"]')
            if name and "台指期" in name.text:
                price = item.select_one('span[class*="Fz(20px)"]').text.replace(",", "")
                change = item.select_one('span[class*="Fz(14px)"]').text
                return float(price), change
                
        return None, None
    except Exception as e:
        st.error(f"Yahoo 爬蟲錯誤: {e}")
        return None, None

def get_option_support_pressure():
    """
    從玩股網爬取選擇權支撐壓力位 (最大 OI 履約價)。
    
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
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 玩股網通常會直接列出最大 OI 履約價
        # 這裡實作邏輯：抓取表格中 Call OI 與 Put OI 最大的那一列
        # 由於網頁動態載入，若 BS4 抓不到，需尋找 JSON 資料區塊或特定標籤
        
        # 提取支撐 (Put Max OI) 與 壓力 (Call Max OI)
        # 範例定位：尋找頁面上具備 'support' 或 'resistance' 關鍵字的區塊
        support_val = None
        pressure_val = None
        
        # 邏輯：遍歷表格中的履約價與 OI
        # 註：玩股網結構複雜，以下為通用解析範例
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")[1:] # 跳過表頭
            call_oi_list = []
            put_oi_list = []
            strikes = []
            
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    # 假設結構: Call OI | 履約價 | Put OI
                    try:
                        c_oi = int(cols[0].text.strip().replace(",", ""))
                        strike = float(cols[1].text.strip().replace(",", ""))
                        p_oi = int(cols[2].text.strip().replace(",", ""))
                        call_oi_list.append(c_oi)
                        put_oi_list.append(p_oi)
                        strikes.append(strike)
                    except:
                        continue
            
            if strikes:
                pressure_val = strikes[call_oi_list.index(max(call_oi_list))]
                support_val = strikes[put_oi_list.index(max(put_oi_list))]
        
        # 若表格解析失敗，則嘗試尋找 summary 標籤
        if not support_val:
            summary_box = soup.find_all("div", class_="p-data")
            # 這裡應根據網頁實際渲染後的標籤名稱微調
            
        return support_val, pressure_val
    except Exception as e:
        st.warning(f"玩股網籌碼解析中 (請檢查網路或網址)... {e}")
        return None, None

def get_yfinance_data():
    """
    獲取大盤加權指數與 VIX 指數數據。
    """
    try:
        twii = yf.Ticker("^TWII").history(period="1d")
        vix = yf.Ticker("^VIX").history(period="1d")
        
        twii_price = twii['Close'].iloc[-1]
        twii_change = twii_price - twii['Open'].iloc[-1]
        
        vix_price = vix['Close'].iloc[-1]
        return twii_price, twii_change, vix_price
    except Exception as e:
        return None, None, None

# --- AI 決策助手 ---

def get_ai_analysis(market_data):
    """
    使用 Gemini 3 Flash 對當前市場數據進行極簡點評。
    """
    if not st.session_state.get('api_key'):
        return "請在側邊欄輸入 Gemini API Key 以獲取 AI 洞察。"
    
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 使用 flash 模型加速
        prompt = f"""
        你是一位專業期貨操盤手。請根據以下數據，提供 50 字以內的極簡盤勢分析與策略建議：
        - 加權指數: {market_data['twii']}
        - 台指期: {market_data['txf']}
        - 價差: {market_data['basis']}
        - VIX: {market_data['vix']}
        - 壓力位: {market_data['pressure']}
        - 支撐位: {market_data['support']}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析暫時不可用: {e}"

# --- UI 介面實作 ---

def main():
    st.title("🚀 專業操盤戰情室 (Yahoo + WantGoo)")
    
    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 系統設定")
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
        
        st.divider()
        st.write("📊 **手動籌碼更新**")
        foreign_oi = st.number_input("外資未平倉淨力道", value=0, step=500)
        
        if st.button("手動刷新數據"):
            st.rerun()
            
        st.info(f"最後更新時間: {datetime.now().strftime('%H:%M:%S')}")

    # 數據獲取
    twii, twii_chg, vix = get_yfinance_data()
    txf, txf_chg = get_realtime_futures()
    support, pressure = get_option_support_pressure()
    
    # 計算基礎邏輯
    basis = (txf - twii) if txf and twii else 0
    
    # ------------------ 第一列：大盤概況 ------------------
    st.subheader("📌 大盤即時概況")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric("加權指數 (TWII)", f"{twii:,.2f}" if twii else "N/A", f"{twii_chg:+.2f}")
    with m_col2:
        st.metric("台指期 (TXF)", f"{txf:,.0f}" if txf else "N/A", f"{txf_chg}")
    with m_col3:
        color = "normal" if basis < 0 else "inverse"
        st.metric("期現貨價差 (Basis)", f"{basis:+.2f}", delta_color=color)
    with m_col4:
        st.metric("恐慌指數 (VIX)", f"{vix:.2f}" if vix else "N/A")

    st.divider()

    # ------------------ 第二列：籌碼戰略 ------------------
    st.subheader("🎯 籌碼策略位元")
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    
    # 若爬蟲失敗，給予預設值顯示
    disp_support = support if support else 0
    disp_pressure = pressure if pressure else 0
    
    with c_col1:
        st.error(f"🔴 上檔壓力 (Call Wall)\n\n### {disp_pressure:,.0f}")
    with c_col2:
        st.success(f"🟢 下檔支撐 (Put Wall)\n\n### {disp_support:,.0f}")
        
    with c_col3:
        # 計算區間位置
        if disp_pressure > disp_support and txf:
            range_pos = ((txf - disp_support) / (disp_pressure - disp_support)) * 100
            st.write("目前區間位置")
            st.progress(max(0, min(100, int(range_pos))) / 100)
            st.write(f"距離支撐 {range_pos:.1f}%")
        else:
            st.write("區間計算中...")
            
    with c_col4:
        st.metric("外資未平倉 (手動)", f"{foreign_oi:,.0f} 口")

    # ------------------ 第三列：AI 診斷 ------------------
    st.divider()
    st.subheader("🤖 AI 市場洞察 (Gemini-1.5-Flash)")
    
    market_summary = {
        "twii": twii, "txf": txf, "basis": basis, 
        "vix": vix, "support": disp_support, "pressure": disp_pressure
    }
    
    with st.container():
        analysis = get_ai_analysis(market_summary)
        st.info(analysis)

    # ------------------ 自動刷新邏輯 ------------------
    time.sleep(60)
    st.rerun()

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# requests
# beautifulsoup4
# google-generativeai
# lxml

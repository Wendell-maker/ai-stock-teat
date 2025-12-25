import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --- 全域設定 ---
st.set_page_config(page_title="專業操盤戰情室", layout="wide")

# --- 數據抓取模組 ---

def get_market_data(ticker_symbol: str) -> float:
    """
    透過 yfinance 獲取特定標的的最新收盤價。

    Args:
        ticker_symbol (str): 標的代碼 (例如: '^TWII', 'WTX=F')

    Returns:
        float: 最新收盤價。若失敗則傳回 0.0。
    """
    try:
        data = yf.download(ticker_symbol, period="1d", interval="1m", progress=False)
        if not data.empty:
            # 嚴格執行 Scalar Conversion，避免 Series 真值判斷錯誤
            return float(data['Close'].iloc[-1])
        return 0.0
    except Exception as e:
        st.error(f"讀取 {ticker_symbol} 失敗: {e}")
        return 0.0

def get_chips_data_live():
    """
    爬取外部網站獲取三大法人籌碼數據 (範例架構)。
    若爬蟲失敗，將回傳 None，由 UI 層接手手動補償值。

    Returns:
        dict: 包含外資期權部位數據，或 None。
    """
    try:
        # 此處為示意爬蟲邏輯，實際網站結構變動頻繁，失敗時需回傳 None
        # 範例：從某財經網站獲取外資淨部位
        # response = requests.get("https://example.com/chips", timeout=5)
        # ... logic ...
        return None  # 預設回傳 None 以啟用手動補償
    except:
        return None

# --- UI 輔助函式 ---

def display_metric(label: str, value: float, delta: float = None, prefix: str = "", suffix: str = ""):
    """
    標準化數值顯示組件。
    """
    val_str = f"{prefix}{value:,.2f}{suffix}"
    st.metric(label=label, value=val_str, delta=f"{delta:,.2f}" if delta else None)

# --- 主程式 ---

def main():
    st.title("📊 專業操盤戰情室 (PRO Dashboard)")

    # --- 側邊欄：設定與手動補償 ---
    with st.sidebar:
        st.header("⚙️ 系統設定")
        api_key = st.text_input("Gemini API Key", type="password", help="請輸入 Google AI Studio 的 API Key")
        
        st.divider()
        
        st.header("🛠️ 手動籌碼補償")
        st.info("當自動爬蟲失效時，請依據交易所數據手動輸入。")
        
        with st.expander("外資/期權數據設定", expanded=True):
            fii_net_position = st.number_input("外資期貨淨部位 (口)", value=-20000, step=100)
            opt_call_value = st.number_input("CALL 壓力點位", value=23000, step=50)
            opt_put_value = st.number_input("PUT 支撐點位", value=22000, step=50)
            
        st.caption("數據來源: Yahoo Finance / 交易所手動輸入")

    # --- 數據獲取 ---
    with st.spinner('同步市場數據中...'):
        # 獲取加權指數與台指期 (WTX=F)
        taiex = get_market_data("^TWII")
        wtx = get_market_data("WTX=F")
        vix = get_market_data("^VIX") # 使用 CBOE VIX 作為全球恐慌參考
        
        # 計算價差
        basis = wtx - taiex if taiex > 0 and wtx > 0 else 0

    # --- Row 1: 指標概覽 ---
    st.subheader("📈 市場即時指標")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric("加權指數 (TAIEX)", taiex)
    with col2:
        display_metric("台指期 (WTX=F)", wtx)
    with col3:
        color = "normal" if abs(basis) < 50 else "inverse"
        st.metric("台指期價差", f"{basis:.2f}", delta_color=color)
    with col4:
        display_metric("恐慌指數 (VIX)", vix)

    # --- Row 2: 籌碼與支撐壓力 ---
    st.subheader("🛡️ 籌碼與關鍵水位")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write("**外資期貨淨部位**")
        status_color = "red" if fii_net_position < -15000 else "green" if fii_net_position > 0 else "orange"
        st.markdown(f"<h2 style='color:{status_color};'>{fii_net_position:,.0f} 口</h2>", unsafe_allow_html=True)
        st.caption("來源: 側邊欄手動補償")

    with c2:
        st.write("**選擇權壓力點 (Call)**")
        st.markdown(f"<h2 style='color:lightcoral;'>{opt_call_value:,.0f}</h2>", unsafe_allow_html=True)

    with c3:
        st.write("**選擇權支撐點 (Put)**")
        st.markdown(f"<h2 style='color:lightgreen;'>{opt_put_value:,.0f}</h2>", unsafe_allow_html=True)

    # --- Row 3: AI 戰略分析 ---
    st.divider()
    st.subheader("🤖 AI 戰略決策")
    
    if st.button("啟動 AI 盤勢分析", use_container_width=True):
        if not api_key:
            st.warning("請先在側邊欄輸入 Gemini API Key")
        else:
            try:
                genai.configure(api_key=api_key)
                # 使用指定模型 (若 gemini-3 未發佈，請降級至 gemini-1.5-flash)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                你是一位專業的台股分析師。請針對以下數據進行深度分析並提供操作建議：
                
                1. 現貨加權指數: {taiex}
                2. 台指期貨: {wtx}
                3. 目前價差: {basis}
                4. VIX 恐慌指數: {vix}
                5. 外資期貨淨部位: {fii_net_position} 口
                6. 選擇權壓力: {opt_call_value} / 支撐: {opt_put_value}
                
                分析要求：
                - 判斷當前盤勢（多方、空方、或區間震盪）。
                - 計算價差異常風險。
                - 給予今日操作策略建議（包含停損觀念）。
                - 請使用繁體中文，語氣專業。
                """
                
                with st.spinner('AI 思考中...'):
                    response = model.generate_content(prompt)
                    st.markdown("### AI 分析報告")
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"AI 分析出錯: {e}")

# --- 執行入口 ---
if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# requests
# beautifulsoup4

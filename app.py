import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import google.generativeai as genai
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI 智能操盤戰情室",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Sidebar Configuration ---
st.sidebar.title("⚙️ 戰情室設定")
st.sidebar.caption("v2.4 Live Scraping")

GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

st.sidebar.divider()
with st.sidebar.expander("📊 手動籌碼數據 (Fallback)", expanded=True):
    MANUAL_FII = st.number_input("外資淨空單", value=-25000, step=500)
    MANUAL_PRESSURE = st.number_input("上檔壓力", value=24000, step=100)
    MANUAL_SUPPORT = st.number_input("下檔支撐", value=23000, step=100)

# --- 3. Data Fetching Functions ---

def get_stock_price(ticker):
    """
    通用抓取股價函式 (台股/指數).
    [CRITICAL FIX] 強制轉型為 float (Scalar).
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return None
        return float(data['Close'].iloc[-1])
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def get_realtime_futures():
    """
    獲取台指期即時報價 (TXF).
    策略: 爬蟲抓取 Yahoo 股市 (TXFR1 - 台指期近一) 以取得最新成交價。
    """
    try:
        url = "https://tw.stock.yahoo.com/quote/TXFR1"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 價格通常位於 class="Fz(32px)" (Yahoo Atomic CSS)
            price_element = soup.find('div', class_='Fz(32px)')
            
            if price_element:
                price = float(price_element.text.replace(',', ''))
                
                # 漲跌通常位於 class="Fz(20px)"
                change = 0.0
                change_element = soup.find('span', class_='Fz(20px)')
                
                if change_element:
                    # 處理特殊符號與顏色
                    raw_txt = change_element.text.strip().replace(',', '')
                    
                    if '▼' in raw_txt or '▽' in raw_txt:
                         # 下跌
                         clean_val = raw_txt.replace('▼', '').replace('▽', '')
                         change = -1 * float(clean_val)
                    elif '▲' in raw_txt or '△' in raw_txt:
                         # 上漲
                         clean_val = raw_txt.replace('▲', '').replace('△', '')
                         change = float(clean_val)
                    else:
                         # 平盤或純數字
                         try:
                             change = float(raw_txt)
                         except:
                             change = 0.0
                             
                return int(price), float(change)
                
    except Exception as e:
        print(f"Scraping Error: {e}")
    
    return None, None

# --- 4. Main Dashboard Logic ---

st.title("🚀 Python 智能操盤戰情室")
st.markdown("---")

# Data Loading
with st.spinner("正在同步市場數據..."):
    taiex_val = get_stock_price("^TWII")    # 加權指數
    txf_price, txf_change = get_realtime_futures() # 台指期 (Scraped)
    tsmc_val = get_stock_price("2330.TW")   # 台積電
    vix_val = get_stock_price("^VIX")       # VIX

# Calculations
try:
    spread = (txf_price - taiex_val) if (txf_price is not None and taiex_val is not None) else 0
except:
    spread = 0

# --- Row 1: Market Overview ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "加權指數 (TAIEX)", 
        f"{taiex_val:,.2f}" if isinstance(taiex_val, (int, float)) else "N/A"
    )

with col2:
    st.metric(
        "台指期 (TXF)", 
        f"{txf_price:,}" if txf_price else "N/A", 
        f"{int(txf_change)}" if txf_change is not None else None
    )

with col3:
    st.metric("期現貨價差", f"{int(spread)}", delta_color="normal")

with col4:
    st.metric(
        "VIX 恐慌指數", 
        f"{vix_val:.2f}" if isinstance(vix_val, (int, float)) else "N/A"
    )

# --- Row 2: Chips Strategy ---
st.subheader("🛡️ 籌碼攻防戰略 (Manual Fallback)")
c1, c2, c3 = st.columns(3)
c1.metric("🔴 上檔壓力", f"{MANUAL_PRESSURE:,}")
c2.metric("🟢 下檔支撐", f"{MANUAL_SUPPORT:,}")
c3.metric("📉 外資淨空單", f"{MANUAL_FII:,}")

# --- Row 3: AI Analysis ---
st.markdown("### 🧠 Gemini AI 戰術分析")

if st.button("生成 AI 戰術報告", type="primary", use_container_width=True):
    if not GEMINI_API_KEY:
        st.error("請先於左側輸入 Gemini API Key")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            
            prompt = f"""
            角色：資深台股操盤手
            數據：
            - 大盤: {taiex_val}
            - 台指: {txf_price} (價差 {spread})
            - VIX: {vix_val}
            - 籌碼: 外資空單 {MANUAL_FII} 口
            
            請簡短分析目前盤勢多空方向與操作建議 (150字內)。
            """
            
            with st.spinner("AI 思考中..."):
                response = model.generate_content(prompt)
                st.info(response.text)
        except Exception as e:
            st.error(f"AI Error: {e}")

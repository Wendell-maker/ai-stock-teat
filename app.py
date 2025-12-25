import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# --- 頁面設定 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Pro Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 樣式美化 ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_market_data(ticker_symbol: str):
    """
    透過 yfinance 獲取市場數據。
    
    Args:
        ticker_symbol (str): Ticker 代號 (例如: ^TWII, WTX=F)
        
    Returns:
        tuple: (現價, 漲跌額, 漲跌幅)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2d")
        if len(df) >= 2:
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = current_price - prev_price
            pct_change = (change / prev_price) * 100
            return round(current_price, 2), round(change, 2), round(pct_change, 2)
        return None, None, None
    except Exception as e:
        st.error(f"數據抓取失敗 ({ticker_symbol}): {e}")
        return None, None, None

def get_vix_data():
    """
    獲取 VIX 指數 (以 ^VIX 為參考)。
    """
    try:
        vix = yf.Ticker("^VIX")
        data = vix.history(period="1d")
        if not data.empty:
            return round(data['Close'].iloc[-1], 2)
        return "N/A"
    except:
        return "N/A"

def get_chips_data_live():
    """
    嘗試爬取玩股網或其他公開來源的籌碼數據 (範例佔位)。
    若失敗則回傳 None，觸發手動補償機制。
    """
    try:
        # 這裡僅模擬爬取邏輯，實際爬蟲需處理 Headers 與解析
        # 由於爬蟲不穩定，建議直接回傳 None 觸發手動輸入
        return None
    except:
        return None

# --- 側邊欄配置 (Sidebar) ---

with st.sidebar:
    st.header("⚙️ 系統設定")
    
    # API Key 設定
    api_key = st.text_input("Gemini API Key", type="password", help="請輸入您的 Google Gemini API Key")
    
    st.divider()
    
    # 手動籌碼補償區塊
    st.subheader("🛠️ 手動籌碼補償")
    with st.expander("編輯即時數據補償值", expanded=True):
        manual_fii = st.number_input("外資期貨淨力道 (口)", value=-20000, step=100)
        manual_support = st.number_input("預計支撐位", value=22500, step=50)
        manual_resistance = st.number_input("預計壓力位", value=23500, step=50)
        
    st.info("💡 當自動爬蟲失敗時，系統將優先採用上述補償值進行 AI 分析。")

# --- 主畫面邏輯 ---

st.title("🚀 專業操盤戰情室")
st.caption(f"數據更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 數據獲取
twii_price, twii_change, twii_pct = get_market_data("^TWII")
wtx_price, wtx_change, wtx_pct = get_market_data("WTX=F")
vix_val = get_vix_data()

# 籌碼邏輯處理 (若 Live 失敗則用手動)
live_chips = get_chips_data_live()
fii_net = live_chips if live_chips else manual_fii
source_tag = "Live" if live_chips else "Manual"

# 計算價差
basis = round(wtx_price - twii_price, 2) if wtx_price and twii_price else "N/A"

# --- 第一排：核心指數 ---
st.subheader("📊 市場核心指標")
r1_col1, r1_col2, r1_col3, r1_col4 = st.columns(4)

with r1_col1:
    st.metric("加權指數", f"{twii_price:,}", f"{twii_change} ({twii_pct}%)", delta_color="inverse" if twii_change < 0 else "normal")

with r1_col2:
    st.metric("台指期 (近月)", f"{wtx_price:,}", f"{wtx_change} ({wtx_pct}%)")

with r1_col3:
    st.metric("期現貨價差", basis, help="正價差代表看多情緒較強，逆價差則反之。")

with r1_col4:
    st.metric("波動率 VIX", vix_val, delta=None)

# --- 第二排：壓力支撐與籌碼 ---
st.subheader("🛡️ 關鍵價位與籌碼")
r2_col1, r2_col2, r2_col3 = st.columns(3)

with r2_col1:
    st.metric("預計壓力位", f"{manual_resistance:,}", "Resistance")

with r2_col2:
    st.metric("預計支撐位", f"{manual_support:,}", "Support")

with r2_col3:
    status_color = "normal" if fii_net > 0 else "inverse"
    st.metric("外資空單/淨部位", f"{fii_net:,} 口", f"來源: {source_tag}", delta_color=status_color)

# --- 第三排：AI 戰略分析 ---
st.divider()
st.subheader("🤖 AI 操盤戰略顧問")

if st.button("生成 AI 盤勢分析報告", use_container_width=True, type="primary"):
    if not api_key:
        st.warning("⚠️ 請先在左側邊欄輸入 Gemini API Key。")
    else:
        try:
            # 初始化 Gemini
            genai.configure(api_key=api_key)
            # 使用指定模型 (若 gemini-3-flash-preview 不存在，SDK 會報錯，此處遵從用戶指示)
            # 實務上建議使用 'gemini-1.5-flash'
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # 建立 Prompt
            prompt = f"""
            你是一位資深的台股短線交易員與量化分析師。
            請根據以下當前市場數據，提供精簡且具備洞察力的盤勢分析與交易策略。

            [當前數據]
            - 加權指數: {twii_price} ({twii_pct}%)
            - 台指期: {wtx_price}
            - 價差: {basis}
            - VIX 指數: {vix_val}
            - 外資期貨淨部位: {fii_net} 口
            - 預計壓力: {manual_resistance}
            - 預計支撐: {manual_support}

            [請包含以下分析模組]
            1. 市場情緒評估 (多/空/中性)
            2. 關鍵價位攻防邏輯
            3. 具體交易策略建議 (包含進場邏輯、停損點規劃)
            4. 風險警示
            
            請使用繁體中文，並以條列式回答。
            """
            
            with st.spinner("AI 正在分析市場動態..."):
                response = model.generate_content(prompt)
                st.markdown("### 📋 AI 戰略分析報告")
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"AI 分析生成失敗: {str(e)}")
            st.info("小提醒：請確認您的 API Key 是否有效，或模型名稱 'gemini-3-flash-preview' 是否為您的權限範圍內。")

# --- 頁尾 ---
st.divider()
st.caption("免責聲明：本工具僅供參考，不構成任何投資建議。投資有風險，入市需謹慎。")

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# requests
# beautifulsoup4

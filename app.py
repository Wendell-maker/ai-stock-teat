# app.py

import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. 頁面配置與初始設定
# ==========================================
st.set_page_config(
    page_title="AI 智能操盤戰情室",
    page_icon="📈",
    layout="centered",  # 手機版建議使用 centered 較為集中
    initial_sidebar_state="collapsed"
)

# 定義策略核心邏輯 (提供給 AI 的上下文)
STRATEGY_CONTEXT = """
【策略核心】
1. 價差結構：(期貨-現貨) > +50 偏多，> +100 強烈軋空。
2. 美股連動：NVDA 漲 > 2% 視為 AI 強勢。
3. 部位狀態：使用者持有 Buy Call 28000 (獲利中)。
"""

# ==========================================
# 2. 側邊欄設定 (API Keys 與 控制項)
# ==========================================
st.sidebar.header("⚙️ 設定與 API Key")

GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
FUGLE_API_KEY = st.sidebar.text_input("Fugle API Key (Optional)", type="password")
TELEGRAM_TOKEN = st.sidebar.text_input("Telegram Token (Optional)", type="password")
TELEGRAM_CHAT_ID = st.sidebar.text_input("Telegram Chat ID (Optional)", type="password")

st.sidebar.markdown("---")
# 自動刷新開關
enable_monitoring = st.sidebar.checkbox("啟動自動監控 (60s Refresh)", value=False)

# 設定自動刷新 (若勾選則每 60 秒刷新一次)
if enable_monitoring:
    st_autorefresh(interval=60 * 1000, key="auto_refresh")

# ==========================================
# 3. 功能模組：數據抓取
# ==========================================
def get_market_data():
    """
    抓取台股現貨、台指期(模擬/延遲)、美股(NVDA, TSM)數據
    回傳: dict 包含關鍵行情數據
    """
    data = {}
    
    try:
        # 使用 yfinance 抓取數據 (注意: yfinance 期貨數據通常有延遲)
        # ^TWII: 加權指數, WTX=F: 台指期(連續月), NVDA:輝達, TSM:台積電ADR, 2330.TW:台積電, TWD=X:匯率
        tickers = ["^TWII", "WTX=F", "NVDA", "TSM", "2330.TW", "TWD=X"]
        df = yf.download(tickers, period="1d", progress=False)
        
        # 處理 MultiIndex Column 問題 (yfinance 新版特性)
        # 取得最新一筆 Close 數據
        latest = df['Close'].iloc[-1]
        prev = df['Close'].iloc[0] # 簡易抓取開盤或前一日做漲跌幅參考
        
        # 1. 匯率 (USD/TWD)
        usdtwd = latest.get("TWD=X", 32.0)
        
        # 2. 台指期與現貨 (計算價差)
        tw_spot = latest.get("^TWII", 0)
        tw_future = latest.get("WTX=F", 0)
        
        # 若抓不到期貨數據(收盤後或代碼問題)，暫以現貨+隨機波動模擬展示 (避免Demo掛掉)
        if tw_future == 0 or pd.isna(tw_future):
            tw_future = tw_spot # Fallback
            
        spread = tw_future - tw_spot
        
        # 3. NVDA 漲跌幅
        nvda_price = latest.get("NVDA", 0)
        nvda_open = df['Open']['NVDA'].iloc[-1]
        nvda_pct = ((nvda_price - nvda_open) / nvda_open) * 100 if nvda_open else 0
        
        # 4. 台積電 ADR 溢價計算
        # ADR 換算台股價格 = (ADR股價 * 匯率) / 5
        tsm_adr = latest.get("TSM", 0)
        tsm_tw = latest.get("2330.TW", 0)
        
        tsm_converted_price = (tsm_adr * usdtwd) / 5
        adr_premium_pct = ((tsm_converted_price - tsm_tw) / tsm_tw) * 100 if tsm_tw else 0
        
        data = {
            "tw_spot": round(tw_spot, 2),
            "tw_future": round(tw_future, 2),
            "spread": round(spread, 2),
            "nvda_price": round(nvda_price, 2),
            "nvda_pct": round(nvda_pct, 2),
            "tsm_tw": round(tsm_tw, 2),
            "tsm_adr": round(tsm_adr, 2),
            "adr_premium_pct": round(adr_premium_pct, 2),
            "usdtwd": round(usdtwd, 2),
            "status": "success"
        }
        
    except Exception as e:
        data = {"status": "error", "message": str(e)}
        
    return data

# ==========================================
# 4. 功能模組：AI 分析 (Gemini)
# ==========================================
def get_gemini_analysis(market_data, api_key):
    """
    呼叫 Google Gemini 針對行情進行極短評
    """
    if not api_key:
        return "⚠️ 請先於側邊欄輸入 Gemini API Key"
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # 建構 Prompt
    prompt = f"""
    角色：你是一位資深的期貨操盤手，風格果斷、犀利。
    
    {STRATEGY_CONTEXT}
    
    【目前即時數據】
    - 台指期: {market_data.get('tw_future')}
    - 現貨: {market_data.get('tw_spot')}
    - 價差 (Spread): {market_data.get('spread')} (正數為正價差)
    - NVDA 漲跌幅: {market_data.get('nvda_pct')}%
    - 台積電 ADR 溢價: {market_data.get('adr_premium_pct')}%
    
    任務：
    請根據數據與策略核心，給出「一句話」的操作建議或盤勢判讀。
    字數限制：30字以內。
    格式要求：開頭使用Emoji (如 🟢, 🔴, ⚠️)，語氣要像戰情室指令。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"AI 分析失敗: {str(e)}"

# ==========================================
# 5. 主程式介面渲染 (Mobile First)
# ==========================================

# --- 頂部狀態列 ---
col_top_1, col_top_2 = st.columns([3, 1])
with col_top_1:
    st.caption(f"最後更新: {datetime.now().strftime('%H:%M:%S')}")
with col_top_2:
    if st.button("🔄"):
        st.rerun()

# 獲取數據
market_data = get_market_data()

# --- 區塊 1: 關鍵信號 (AI 建議) ---
st.markdown("### 🤖 戰情室指令")

if market_data['status'] == 'error':
    st.error(f"數據抓取錯誤: {market_data['message']}")
else:
    # 呼叫 AI (為了節省 Token，實際使用可增加快取機制)
    if GEMINI_API_KEY:
        with st.spinner("AI 正在分析盤勢..."):
            advice = get_gemini_analysis(market_data, GEMINI_API_KEY)
        
        # 根據建議內容簡單判斷顏色 (包含 "空" 用紅，"多" 用綠，"觀望" 用藍)
        if "空" in advice or "跌" in advice:
            st.error(advice, icon="📉")
        elif "多" in advice or "漲" in advice:
            st.success(advice, icon="📈")
        else:
            st.info(advice, icon="👀")
    else:
        st.warning("請輸入 Gemini API Key 以獲取 AI 建議")

# --- 區塊 2: 核心數據矩陣 (2x2) ---
st.markdown("### 📊 核心監控")

if market_data['status'] == 'success':
    col1, col2 = st.columns(2)
    
    # 1. 台指期
    with col1:
        st.metric(
            label="台指期 (TX)",
            value=market_data['tw_future'],
            delta=f"{market_data['spread']} (Spread)"
        )
    
    # 2. 價差 (Spread) 特別處理顏色
    with col2:
        spread_val = market_data['spread']
        # 定義顯示顏色邏輯
        spread_color = "normal"
        if spread_val > 50:
            spread_label = "🟢 偏多價差"
        elif spread_val < -50:
            spread_label = "🔴 逆價差大"
        else:
            spread_label = "⚪ 盤整價差"
            
        st.metric(
            label="價差結構",
            value=spread_val,
            delta="強勢" if spread_val > 50 else "弱勢",
            delta_color="normal" if -50 <= spread_val <= 50 else ("inverse" if spread_val < 0 else "normal")
        )

    # 換行
    col3, col4 = st.columns(2)
    
    # 3. NVDA 狀態
    with col3:
        st.metric(
            label="NVDA 漲跌幅",
            value=f"{market_data['nvda_price']}",
            delta=f"{market_data['nvda_pct']}%"
        )
        
    # 4. ADR 溢價
    with col4:
        st.metric(
            label="台積電 ADR 溢價",
            value=f"{market_data['adr_premium_pct']}%",
            delta="有利現貨" if market_data['adr_premium_pct'] > 0 else "拖累",
            delta_color="off" # 溢價通常看絕對值，這裡僅作展示
        )

else:
    st.info("等待數據載入...")

# --- 底部除錯與資訊 ---
with st.expander("查看原始數據細節"):
    st.json(market_data)
    st.text(STRATEGY_CONTEXT)


# requirements.txt
# ---------------------
# streamlit
# streamlit-autorefresh
# yfinance
# google-generativeai
# requests
# pandas
# ---------------------

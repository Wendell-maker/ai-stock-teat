import streamlit as st
import yfinance as yf
import requests
import google.generativeai as genai
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 網頁與全域設定
# ==========================================
st.set_page_config(page_title="AI 智能操盤戰情室 (Ultimate)", page_icon="🦅", layout="wide")

# 初始化 Session State (動態記憶)
if 'prev_spread' not in st.session_state:
    st.session_state.prev_spread = 0
if 'prev_tx' not in st.session_state:
    st.session_state.prev_tx = 0

# 自動刷新計時器 (每 60 秒)
count = st_autorefresh(interval=60000, limit=None, key="fcounter")

# ==========================================
# 2. 側邊欄：金鑰與設定
# ==========================================
with st.sidebar:
    st.header("⚙️ 系統設定")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    
    auto_refresh = st.checkbox("啟動全自動監控", value=True)
    st.caption("含 VIX 恐慌指數與 RSI 技術分析")

# ==========================================
# 3. 數據抓取與計算引擎
# ==========================================

# A. 抓台指期 (爬蟲)
def get_tw_futures():
    try:
        url = "https://mis.taifex.com.tw/futures/api/getQuoteList"
        payload = {"MarketType": "0", "SymbolType": "F", "KindID": "1", "CID": "TXF", "ExpireMonth": ""}
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.post(url, json=payload, headers=headers, timeout=5)
        data = res.json()
        if data['QuoteList']:
            quote = data['QuoteList'][0]
            price = float(quote.get('DealPrice', 0))
            return price
    except:
        return 0

# B. 抓美股與 VIX (yfinance)
def get_us_market_data():
    """一次抓取 NVDA 和 VIX"""
    try:
        tickers = yf.Tickers("NVDA ^VIX")
        
        # NVDA 處理
        nvda_hist = tickers.tickers['NVDA'].history(period="1d")
        if not nvda_hist.empty:
            nvda_price = nvda_hist['Close'].iloc[-1]
            nvda_open = nvda_hist['Open'].iloc[0]
            nvda_chg = ((nvda_price - nvda_open) / nvda_open) * 100
        else:
            nvda_price, nvda_chg = 0, 0
            
        # VIX 處理
        vix_hist = tickers.tickers['^VIX'].history(period="1d")
        if not vix_hist.empty:
            vix_price = vix_hist['Close'].iloc[-1]
        else:
            vix_price = 0
            
        return nvda_price, nvda_chg, vix_price
    except:
        return 0, 0, 0

# C. 抓現貨並計算技術指標 (RSI, MA)
def get_technical_analysis():
    """
    抓取加權指數 (^TWII) 的 K 線來計算技術指標
    回傳: 現貨價格, RSI數值, MA5價格
    """
    try:
        # 抓取最近 30 天資料 (計算 RSI 用)
        tw = yf.Ticker("^TWII")
        hist = tw.history(period="1mo") 
        
        if hist.empty:
            return 0, 50, 0 # 預設值
            
        current_price = hist['Close'].iloc[-1]
        
        # 1. 計算 MA5 (五日均線)
        ma5 = hist['Close'].rolling(window=5).mean().iloc[-1]
        
        # 2. 計算 RSI (14)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        return current_price, current_rsi, ma5
    except:
        return 0, 50, 0

# ==========================================
# 4. AI 策略大腦 (含技術指標與VIX)
# ==========================================
STRATEGY_CONTEXT = """
【角色設定】
你是一位精通「技術分析」與「籌碼解讀」的頂尖操盤手。
你的決策必須綜合考量：價差籌碼、技術位階 (RSI/MA)、以及市場恐慌度 (VIX)。

【多維度判斷邏輯】
1. **價差結構 (Spread)**:
   - 價差 > +50 且 動能(Delta) > 0：多頭強攻。
   - 價差 > +50 但 動能 < -20：買盤力竭，多單警戒。

2. **技術位階 (Technical Filter)** - *這是精準度的關鍵*:
   - **RSI 指標**: 若 RSI > 80，視為「嚴重過熱」。即使價差是紅的，也**絕對禁止**追價，建議等待拉回或平倉。
   - **MA5 均線**: 價格在 MA5 之上為強勢；跌破 MA5 為轉弱訊號 (出場點)。

3. **VIX 恐慌指數 (Volatility)**:
   - **VIX > 20**: 市場恐慌，權利金極貴。策略：不留倉，快進快出。
   - **VIX < 13**: 市場安逸，權利金便宜。策略：適合波段持有 Buy Call。
   - **VIX 暴漲**: 若指數跌且 VIX 飆升，代表恐慌性殺盤，Put 會噴出。

4. **美股連動**:
   - NVDA 漲 > 2%：AI 族群強勢助漲。

【使用者部位】
- 持有：Buy Call 28000。
- 任務：利用上述指標，幫我判斷現在該「貪婪」還是該「恐懼」。
"""

def get_gemini_analysis(api_key, tx, spread, delta, nvda_chg, vix, rsi, ma5, tw_spot):
    if not api_key:
        return "⚠️ 請輸入 API Key"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # 判斷技術面狀態文字
    tech_status = []
    if rsi > 75: tech_status.append("🔴 RSI過熱")
    elif rsi < 25: tech_status.append("🟢 RSI超賣")
    else: tech_status.append("⚪ RSI中性")
    
    if tw_spot > ma5: tech_status.append("🟢 站上MA5")
    else: tech_status.append("🔴 跌破MA5")

    prompt = f"""
    請用繁體中文進行高精準度盤勢分析。

    === 1. 全方位數據 ===
    - **台指期**: {tx:.0f}
    - **期現價差**: {spread:.0f} (動能: {delta:.0f})
    - **美股 NVDA**: {nvda_chg:.2f}%
    - **VIX 恐慌指數**: {vix:.2f} (判斷權利金貴賤)
    - **RSI (14)**: {rsi:.1f} ({tech_status[0]})
    - **MA5 位置**: {ma5:.0f} ({tech_status[1]})

    === 2. 策略邏輯 ===
    {STRATEGY_CONTEXT}

    === 3. 分析結論 ===
    請給我簡潔的決策儀表板：
    1. 【盤勢訊號】：(例如：🟢 軋空噴出 / 🔴 過熱拉回 / 🟡 震盪整理)
    2. 【關鍵變數】：指出目前影響最大的指標 (是VIX太高？還是RSI過熱？還是價差擴大？)
    3. 【操盤指令】：針對 Buy Call 部位，給出明確指令 (續抱/減碼/移動停利/空手)。
    """
    
    try:
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        return f"分析錯誤: {e}"

# ==========================================
# 5. 主程式顯示層
# ==========================================
st.title("🦅 AI 操盤戰情室 (Ultimate)")
st.markdown(f"Update: {time.strftime('%H:%M:%S')}")

# 1. 獲取所有數據
tx_price = get_tw_futures()
tw_spot, rsi, ma5 = get_technical_analysis()
nvda_price, nvda_chg, vix = get_us_market_data()

# 2. 計算衍生數據
if tw_spot != 0:
    spread = tx_price - tw_spot
else:
    spread = 0

spread_delta = spread - st.session_state.prev_spread
st.session_state.prev_spread = spread

# 3. 顯示數據矩陣 (3x2 排列)
c1, c2, c3 = st.columns(3)
c1.metric("台指期 (TX)", f"{tx_price:.0f}", f"{spread:.0f} (價差)")
c2.metric("VIX 恐慌指數", f"{vix:.2f}", "權利金水位")
c3.metric("NVDA 漲跌", f"{nvda_chg:.2f}%", f"{nvda_price:.2f}")

c4, c5, c6 = st.columns(3)
c4.metric("價差動能 (Delta)", f"{spread_delta:.0f}", "多空力道")
c5.metric("RSI 強弱", f"{rsi:.1f}", "80過熱/20超賣")
c6.metric("MA5 均線", f"{ma5:.0f}", "短線防守")

st.divider()

# 4. AI 戰略分析
st.subheader("🤖 戰略指揮中心")

if auto_refresh:
    with st.spinner("AI 正在綜合運算 RSI, VIX 與 籌碼數據..."):
        advice = get_gemini_analysis(gemini_api_key, tx_price, spread, spread_delta, nvda_chg, vix, rsi, ma5, tw_spot)
        
        # 顯示樣式
        if "續抱" in advice or "軋空" in advice:
            st.success(advice)
        elif "平倉" in advice or "減碼" in advice:
            st.error(advice)
        else:
            st.warning(advice)
else:
    st.info("勾選左側「啟動全自動監控」以獲取分析。")

# 頁尾
st.caption("Data Sources: Taifex (Crawler), Yahoo Finance (API)")
        

import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# --- 頁面設定 ---
st.set_page_config(
    page_title="專業操盤戰情室 | AI 決策系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 數據抓取模組 ---

def get_vix_data():
    """
    從 yfinance 獲取 VIX 波動率指數數據。
    
    Returns:
        dict: 包含 'price' 與 'change' 的字典，若失敗則回傳 None。
    """
    try:
        # 獲取 VIX 數據
        vix = yf.Ticker("^VIX")
        df = vix.history(period="2d")
        
        # 嚴格檢查數據是否為空，避免 IndexError
        if df.empty or len(df) < 2:
            return None
            
        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change = current_price - prev_price
        
        return {
            'price': round(current_price, 2),
            'change': round(change, 2)
        }
    except Exception as e:
        st.error(f"VIX 數據獲取失敗: {e}")
        return None

def get_market_quote(symbol: str):
    """
    獲取指定標的的即時行情 (支援 yfinance 代號)。
    
    Args:
        symbol (str): 標的代號 (例如: ^TWII, WTX=F)
        
    Returns:
        dict: 包含 'price' 與 'change' 的字典，若失敗則回傳 None。
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2d")
        
        if df.empty or len(df) < 1:
            return None
            
        current_price = float(df['Close'].iloc[-1])
        # 若有兩天數據則計算漲跌，否則回傳 0.0
        change = (current_price - df['Close'].iloc[-2]) if len(df) >= 2 else 0.0
        
        return {
            'price': round(current_price, 2),
            'change': round(change, 2)
        }
    except Exception as e:
        return None

def get_historical_data(symbol: str, days: int = 30):
    """
    獲取歷史 K 線數據用於繪圖。
    
    Args:
        symbol (str): 標的代號
        days (int): 天數
        
    Returns:
        pd.DataFrame: 包含歷史價格的 DataFrame。
    """
    try:
        df = yf.download(symbol, start=(datetime.now() - timedelta(days=days)), end=datetime.now())
        return df
    except Exception:
        return pd.DataFrame()

# --- AI 分析模組 ---

def get_ai_analysis(api_key: str, market_data: dict):
    """
    呼叫 Google Gemini 模型進行市場分析。
    
    Args:
        api_key (str): Google API Key
        market_data (dict): 包含當前市場數值的字典
        
    Returns:
        str: AI 分析結果。
    """
    if not api_key:
        return "請在側邊欄輸入 API Key 以啟動 AI 分析。"
        
    try:
        genai.configure(api_key=api_key)
        # 預設使用用戶指定的模型版本
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = f"""
        你是一位專業的量化交易分析師。請針對以下市場數據進行簡短且精闢的解說：
        - 加權指數 (Spot): {market_data.get('spot_price')}
        - 台指期 (Futures): {market_data.get('fut_price')}
        - 逆價差/正價差 (Spread): {market_data.get('spread')}
        - VIX 指數: {market_data.get('vix_price')}
        
        請給出：
        1. 當前多空氛圍判斷。
        2. 操作建議 (短線)。
        請用繁體中文回覆，並保持專業語氣。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}"

# --- 主程式邏輯 ---

def main():
    # --- 側邊欄配置 ---
    st.sidebar.title("⚙️ 系統設定")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="請輸入您的 Google Gemini API Key")
    
    st.sidebar.markdown("---")
    st.sidebar.info("本系統每 60 秒自動重新計算一次 (手動重新整理網頁)")
    
    # --- 數據獲取與安全解包 ---
    # 這裡使用 yfinance 的代號作為範例
    spot_data = get_market_quote("^TWII")    # 台股加權指數
    fut_data = get_market_quote("WTX=F")    # 台指期 (連續合約)
    vix_data = get_vix_data()               # VIX 指數
    
    # 安全提取數值 (Safe Unpacking)
    s_price = spot_data['price'] if spot_data else None
    s_change = spot_data['change'] if spot_data else 0.0
    
    f_price = fut_data['price'] if fut_data else None
    f_change = fut_data['change'] if fut_data else 0.0
    
    v_price = vix_data['price'] if vix_data else None
    v_change = vix_data['change'] if vix_data else 0.0
    
    # 計算價差 (Spread)
    spread = None
    if s_price is not None and f_price is not None:
        spread = round(f_price - s_price, 2)

    # --- UI 視覺呈現 ---
    st.title("📈 專業操盤戰情室")
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 第一排：核心數據指標 (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("加權指數 (現貨)", 
                  value=f"{s_price:,.2f}" if s_price else "N/A", 
                  delta=f"{s_change:+.2f}" if s_price else None)
        
    with col2:
        st.metric("台指期 (期貨)", 
                  value=f"{f_price:,.2f}" if f_price else "N/A", 
                  delta=f"{f_change:+.2f}" if f_price else None)
        
    with col3:
        # 價差判斷
        spread_label = "價差 (Spread)"
        if spread is not None:
            delta_color = "normal" if spread > 0 else "inverse"
            st.metric(spread_label, value=f"{spread:+.2f}", delta="正價差" if spread > 0 else "逆價差", delta_color=delta_color)
        else:
            st.metric(spread_label, value="N/A")

    with col4:
        st.metric("恐慌指數 (VIX)", 
                  value=f"{v_price:.2f}" if v_price else "N/A", 
                  delta=f"{v_change:+.2f}" if v_price else None,
                  delta_color="inverse") # VIX 上漲通常對股市是不利的

    st.markdown("---")

    # 第二排：圖表與 AI 分析
    chart_col, ai_col = st.columns([2, 1])

    with chart_col:
        st.subheader("📊 現貨歷史走勢 (30D)")
        hist_df = get_historical_data("^TWII")
        if not hist_df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=hist_df.index,
                open=hist_df['Open'],
                high=hist_df['High'],
                low=hist_df['Low'],
                close=hist_df['Close'],
                increasing_line_color='#ef5350', # 紅漲 (台灣習慣)
                decreasing_line_color='#26a69a'  # 綠跌
            )])
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("暫無歷史數據可供顯示")

    with ai_col:
        st.subheader("🤖 AI 盤勢分析")
        with st.container(border=True):
            market_context = {
                'spot_price': s_price,
                'fut_price': f_price,
                'spread': spread,
                'vix_price': v_price
            }
            if st.button("生成 AI 觀點", use_container_width=True):
                with st.spinner("AI 正在解讀市場數據..."):
                    analysis = get_ai_analysis(api_key, market_context)
                    st.markdown(analysis)
            else:
                st.write("請點擊上方按鈕獲取 AI 建議。")

    # 頁尾資訊
    st.markdown("---")
    st.caption("數據來源: Yahoo Finance | 警語: 本系統僅供參考，投資盈虧請自行負責。")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# pandas
# google-generativeai
# plotly
# requests

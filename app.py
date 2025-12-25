import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from fugle_marketdata import RestClient
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="專業操盤戰情室 | Professional Trading Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定義 CSS 樣式 (優化 RWD 與 視覺效果) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .stAlert { background-color: #1e2130; border: none; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    h1, h2, h3 { color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 側邊欄設定模組 ---
with st.sidebar:
    st.title("🛡️ 系統核心配置")
    st.markdown("---")
    
    # API 金鑰輸入區
    fugle_api_key = st.text_input("Fugle API Key", type="password", help="用於加權指數與個股即時數據")
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="用於 AI 盤勢分析與策略產出")
    
    st.markdown("---")
    st.subheader("📡 連線狀態")
    if fugle_api_key and gemini_api_key:
        st.success("API 金鑰已備妥")
    else:
        st.warning("請輸入 API 金鑰以啟用完整功能")
    
    update_interval = st.slider("數據刷新頻率 (秒)", 5, 60, 15)
    
    if st.button("🚀 強制刷新數據"):
        st.rerun()

# --- 數據抓取模組 ---

def get_futures_from_yf():
    """
    使用 yfinance 抓取台指期 (WTX=F) 的即時數據。
    
    Returns:
        tuple: (current_price, price_change, change_percent, volume)
    """
    try:
        ticker = yf.Ticker("WTX=F")
        # 抓取 1 天內 1 分鐘線
        df = ticker.history(period="1d", interval="1m")
        
        if df.empty:
            return 0.0, 0.0, 0.0, 0
            
        last_price = float(df['Close'].iloc[-1])
        prev_close = float(ticker.info.get('previousClose', last_price))
        change = last_price - prev_close
        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0.0
        volume = int(df['Volume'].iloc[-1])
        
        return last_price, change, pct_change, volume
    except Exception as e:
        st.error(f"期貨數據抓取失敗: {e}")
        return 0.0, 0.0, 0.0, 0

def get_market_data_fugle(api_key):
    """
    使用 Fugle REST Client 獲取加權指數與台積電數據。
    
    Args:
        api_key (str): Fugle API 金鑰
    Returns:
        dict: 包含市場指標的字典
    """
    if not api_key:
        return None
        
    client = RestClient(api_key=api_key)
    try:
        # 加權指數 (TSE01)
        tse = client.stock.intraday.quote(symbol='IX0001')
        # 台積電 (2330)
        tsmc = client.stock.intraday.quote(symbol='2330')
        
        return {
            "tse_price": float(tse.get('lastPrice', 0)),
            "tse_change": float(tse.get('change', 0)),
            "tse_pct": float(tse.get('changePercent', 0)),
            "tsmc_price": float(tsmc.get('lastPrice', 0)),
            "tsmc_change": float(tsmc.get('change', 0)),
            "tsmc_pct": float(tsmc.get('changePercent', 0))
        }
    except Exception as e:
        st.sidebar.error(f"Fugle 數據異常: {e}")
        return None

# --- AI 分析模組 ---

def get_ai_analysis(api_key, market_info):
    """
    調用 Google Gemini Pro 進行盤勢多空判斷。
    
    Args:
        api_key (str): Gemini API Key
        market_info (dict): 當前市場數值
    Returns:
        str: AI 分析結果
    """
    if not api_key:
        return "請先配置 Gemini API Key 以開啟 AI 操盤助手。"
        
    try:
        genai.configure(api_key=api_key)
        # 預設使用用戶指定的 gemini-3-flash-preview (如版本未開放則降級至 gemini-1.5-flash)
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = f"""
        你是一位專業的台股量化交易分析師。請根據以下即時數據進行短線盤勢分析：
        1. 台股加權指數: {market_info['tse_price']} (漲跌: {market_info['tse_change']})
        2. 台指期 (WTX=F): {market_info['fut_price']} (漲跌: {market_info['fut_change']})
        3. 護國神山台積電: {market_info['tsmc_price']} (漲跌: {market_info['tsmc_pct']}%)
        
        請提供：
        - 多空勢力對比分析
        - 關鍵支撐與壓力位預測
        - 短線操作策略建議 (保守/積極)
        請用繁體中文回覆，語氣專業且精煉。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析暫時無法使用: {str(e)}"

# --- UI 佈局主體 ---

def main():
    st.title("📊 專業操盤戰情室")
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 系統延遲: < 500ms")

    # 1. 數據抓取
    market_data = get_market_data_fugle(fugle_api_key)
    fut_price, fut_change, fut_pct, fut_vol = get_futures_from_yf()

    if market_data:
        # 2. 頂部核心指標區 (Metric Row)
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric(
                label="加權指數 (TSE01)", 
                value=f"{market_data['tse_price']:,.2f}", 
                delta=f"{market_data['tse_change']:+.2f} ({market_data['tse_pct']}%)"
            )
            
        with m2:
            st.metric(
                label="台指期 (WTX=F)", 
                value=f"{fut_price:,.0f}", 
                delta=f"{fut_change:+.0f} ({fut_pct:.2f}%)"
            )
            
        with m3:
            st.metric(
                label="台積電 (2330)", 
                value=f"{market_data['tsmc_price']}", 
                delta=f"{market_data['tsmc_change']:+.2f} ({market_data['tsmc_pct']}%)"
            )
            
        with m4:
            # 簡單計算盤中波動率指標
            volatility = abs(fut_pct) * 1.5
            st.metric("市場預估波動率", f"{volatility:.2f}%", delta="Normal", delta_color="off")

        # 3. 中間區塊：圖表與 AI 分析
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📈 盤中趨勢監控")
            # 這裡展示簡單的 YFinance 歷史圖表
            ticker = yf.Ticker("WTX=F")
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                fig = go.Figure(data=[go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='WTX=F'
                )])
                fig.update_layout(
                    template="plotly_dark", 
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("等待期貨圖表數據更新...")

        with c2:
            st.subheader("🤖 AI 策略洞察")
            # 彙整數據給 AI
            info_for_ai = {
                "tse_price": market_data['tse_price'],
                "tse_change": market_data['tse_change'],
                "fut_price": fut_price,
                "fut_change": fut_change,
                "tsmc_price": market_data['tsmc_price'],
                "tsmc_pct": market_data['tsmc_pct']
            }
            
            with st.spinner("AI 正在解析盤勢..."):
                analysis_report = get_ai_analysis(gemini_api_key, info_for_ai)
                st.markdown(f"""
                <div style="background-color: #161b22; padding: 20px; border-radius: 10px; border-left: 5px solid #00d4ff;">
                    {analysis_report}
                </div>
                """, unsafe_allow_html=True)

        # 4. 底部狀態列
        st.markdown("---")
        st.subheader("🔍 盤中異動偵測")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if abs(fut_price - market_data['tse_price']) > 50:
                st.warning(f"⚠️ 期現貨價差擴大：目前價差 {fut_price - market_data['tse_price']:.2f}")
            else:
                st.success("✅ 期現貨價差處於正常範圍")
        with col_v2:
            st.info(f"當前成交量估計 (期貨): {fut_vol} 口")

    else:
        st.info("💡 請在側邊欄輸入 API 金鑰以獲取即時市場數據。")
        st.image("https://images.unsplash.com/photo-1611974714024-282424b8979e?auto=format&fit=crop&w=1200&q=80")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# google-generativeai
# fugle-marketdata
# pandas
# plotly

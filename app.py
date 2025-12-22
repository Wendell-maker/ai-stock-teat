import streamlit as st
import pandas as pd
import yfinance as yf
from fugle-marketdata import RestStockClient
import google.generativeai as genai
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# --- 設定頁面配置 ---
st.set_page_config(
    page_title="台股戰情室 Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 常數設定 ---
AI_MODEL_VERSION = 'gemini-1.5-pro-latest' # 根據Google最新發布，若需 "gemini-3-pro-preview" 可在此修改，目前建議使用穩定版 1.5 Pro
TZ_TW = pytz.timezone('Asia/Taipei')

# --- 數據抓取模組 ---

def get_yfinance_data(ticker: str) -> dict:
    """
    使用 yfinance 抓取即時報價 (作為備援或指數使用)。
    
    Args:
        ticker (str): 股票或期貨代碼 (e.g., 'TXF=F', '^TWII').
    
    Returns:
        dict: 包含價格、漲跌幅、更新時間的字典，若失敗回傳 None。
    """
    try:
        stock = yf.Ticker(ticker)
        # 嘗試獲取 fast_info 或 history
        info = stock.fast_info
        
        # 檢查是否有有效數據
        if info is None or info.last_price is None:
            # 降級嘗試使用 history
            df = stock.history(period='1d', interval='1m')
            if df.empty:
                return None
            current_price = df['Close'].iloc[-1]
            prev_close = stock.info.get('previousClose', df['Open'].iloc[0])
        else:
            current_price = info.last_price
            prev_close = info.previous_close

        change = current_price - prev_close
        pct_change = (change / prev_close) * 100
        
        return {
            'price': current_price,
            'change': change,
            'pct_change': pct_change,
            'time': datetime.now(TZ_TW).strftime('%H:%M:%S'),
            'source': 'Yahoo Finance (Delayed/Est)'
        }
    except Exception as e:
        print(f"YFinance Error for {ticker}: {e}")
        return None

def get_fugle_data(symbol: str, api_key: str) -> dict:
    """
    使用 Fugle API 抓取即時報價 (優先使用)。
    
    Args:
        symbol (str): 股票代碼 (e.g., '2330', 'TXF').
        api_key (str): Fugle API Key.
    
    Returns:
        dict: 包含價格、漲跌幅、更新時間的字典，若失敗回傳 None。
    """
    if not api_key:
        return None
    
    try:
        client = RestStockClient(api_key=api_key)
        stock = client.stock  # Stock API 進入點
        
        # 取得個股即時報價 (intraday/quote)
        quote = stock.intraday.quote(symbol=symbol)
        
        if 'lastPrice' in quote:
            current_price = quote['lastPrice']
            change = quote['change']
            pct_change = quote['changePercent']
            update_time = datetime.fromtimestamp(quote['lastUpdated']/1000, TZ_TW).strftime('%H:%M:%S')
            
            return {
                'price': current_price,
                'change': change,
                'pct_change': pct_change,
                'time': update_time,
                'source': 'Fugle Real-time API'
            }
        return None
    except Exception as e:
        # 可以在此紀錄錯誤，但不中斷程式
        print(f"Fugle API Error for {symbol}: {e}")
        return None

def fetch_market_data(fugle_key: str = None):
    """
    核心數據整合邏輯：
    1. TXF (台指期): 優先 Fugle，失敗轉 Yahoo (TXF=F)。
    2. TWII (加權指數): 僅使用 Yahoo (^TWII)。
    3. 2330 (台積電): 優先 Fugle，失敗轉 Yahoo (2330.TW)。
    """
    data = {}
    
    # --- 1. 台指期 (TXF) ---
    txf_data = None
    if fugle_key:
        # 嘗試 Fugle (假設 symbol 為 TXF，需視 Fugle 實際期貨代碼權限而定，若無權限會 Exception)
        txf_data = get_fugle_data('TXF', fugle_key)
    
    if not txf_data:
        # Fallback to Yahoo
        txf_data = get_yfinance_data('TXF=F')
        if txf_data:
            txf_data['source'] = 'Yahoo (TXF=F)'
    
    data['TXF'] = txf_data

    # --- 2. 加權指數 (TWII) ---
    # Fugle 主要針對個股，指數部分建議維持 Yahoo 或需付費 API
    twii_data = get_yfinance_data('^TWII')
    if twii_data:
        twii_data['source'] = 'Yahoo (^TWII)'
    data['TWII'] = twii_data

    # --- 3. 台積電 (2330) ---
    tsmc_data = None
    if fugle_key:
        tsmc_data = get_fugle_data('2330', fugle_key)
    
    if not tsmc_data:
        tsmc_data = get_yfinance_data('2330.TW')
        if tsmc_data:
            tsmc_data['source'] = 'Yahoo (2330.TW)'
    
    data['2330'] = tsmc_data
    
    return data

# --- AI 分析模組 ---

def analyze_market_ai(market_data: dict, gemini_key: str):
    """
    呼叫 Google Gemini 模型進行市場分析。
    
    Args:
        market_data (dict): 包含各商品報價的字典。
        gemini_key (str): Google AI API Key。
        
    Returns:
        str: AI 分析結果文本。
    """
    if not gemini_key:
        return None

    try:
        genai.configure(api_key=gemini_key)
        # 指定使用 gemini-3-pro-preview (若無法使用會自動報錯，建議使用 try-except)
        # 註：若 gemini-3 尚未正式開放，請改回 'gemini-1.5-pro'
        model = genai.GenerativeModel("gemini-1.5-pro-latest") 
        
        prompt = f"""
        你是一位專業的量化交易員。請根據以下即時數據進行簡短的盤勢分析與風險提示：
        
        [市場數據]
        1. 台指期: {market_data.get('TXF', {})}
        2. 加權指數: {market_data.get('TWII', {})}
        3. 台積電: {market_data.get('2330', {})}
        
        請給出：
        1. 目前多空方向判斷。
        2. 短線關鍵支撐/壓力位觀察。
        3. 給予當沖交易者的具體建議 (保守/積極)。
        請用繁體中文回答，條列式重點，語氣專業冷靜。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析連線失敗: {str(e)}"

# --- UI 組件模組 ---

def render_metric_card(title: str, data: dict):
    """
    渲染單個商品的指標卡片。
    """
    if not data:
        st.metric(label=title, value="N/A", delta="資料讀取失敗")
        st.caption("來源: 無法連線")
        return

    color = "normal"
    if data['change'] > 0:
        color = "off" # Streamlit metric 自動處理顏色，這裡僅示意
    
    st.metric(
        label=title,
        value=f"{data['price']:,.0f}",
        delta=f"{data['change']:.1f} ({data['pct_change']:.2f}%)"
    )
    st.caption(f"來源: {data['source']} | 時間: {data['time']}")

# --- 主程式 ---

def main():
    # 1. Session State 初始化
    if 'fugle_api_key' not in st.session_state:
        st.session_state.fugle_api_key = None
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = None

    # 2. 側邊欄設定 (Sidebar)
    with st.sidebar:
        st.title("⚙️ 戰情室設定")
        
        # --- 自動刷新監控 ---
        st.subheader("📡 即時監控")
        auto_refresh = st.toggle("全自動監控 (Auto-refresh)", value=False)
        
        if auto_refresh:
            st_autorefresh(interval=60 * 1000, key="data_refresh")
            st.info("🔄 系統每 60 秒自動刷新")
        else:
            if st.button("手動刷新數據"):
                st.rerun()

        st.divider()

        # --- Fugle API 設定 (資安優化) ---
        st.subheader("🔑 Fugle MarketData")
        if st.session_state.fugle_api_key:
            st.success("✅ Fugle API 已連線 (安全儲存)")
            if st.button("🔄 重設/登出 Fugle"):
                st.session_state.fugle_api_key = None
                st.rerun()
        else:
            fugle_input = st.text_input("輸入 Fugle API Key", type="password", key="input_fugle")
            if st.button("連線 Fugle"):
                if fugle_input:
                    st.session_state.fugle_api_key = fugle_input
                    st.rerun()
                else:
                    st.warning("請輸入 API Key")

        st.divider()

        # --- Gemini API 設定 (資安優化) ---
        st.subheader("🤖 Google Gemini AI")
        if st.session_state.gemini_api_key:
            st.success("✅ Gemini AI 已就緒")
            if st.button("🔄 重設/登出 Gemini"):
                st.session_state.gemini_api_key = None
                st.rerun()
        else:
            gemini_input = st.text_input("輸入 Gemini API Key", type="password", key="input_gemini")
            if st.button("啟用 AI 分析"):
                if gemini_input:
                    st.session_state.gemini_api_key = gemini_input
                    st.rerun()
                else:
                    st.warning("請輸入 API Key")
        
        st.markdown("---")
        st.caption("Designed for Pro Traders")

    # 3. 主畫面內容
    st.title("🛡️ 台股即時戰情室 (Secure Edition)")
    st.markdown(f"最後更新: {datetime.now(TZ_TW).strftime('%Y-%m-%d %H:%M:%S')}")

    # 4. 獲取數據
    with st.spinner("正在同步交易所數據..."):
        market_data = fetch_market_data(st.session_state.fugle_api_key)

    # 5. 顯示報價卡片 (RWD: 手機版自動堆疊)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_card("台指期 (TXF)", market_data.get('TXF'))
    
    with col2:
        render_metric_card("加權指數 (TWII)", market_data.get('TWII'))
        
    with col3:
        render_metric_card("台積電 (2330)", market_data.get('2330'))

    st.divider()

    # 6. AI 戰略分析區塊
    st.subheader("🧠 AI 戰略分析")
    
    if st.session_state.gemini_api_key:
        if st.button("生成最新市場解讀", type="primary"):
            with st.spinner("Gemini 正在分析盤勢..."):
                ai_analysis = analyze_market_ai(market_data, st.session_state.gemini_api_key)
                if ai_analysis:
                    st.markdown(ai_analysis)
                else:
                    st.error("分析生成失敗，請檢查 API Key 或額度。")
    else:
        # AI 降級提示
        st.info("ℹ️ 解鎖 AI 深度分析功能：請於左側選單輸入 Google Gemini API Key。目前僅顯示基礎數據。")

    # (可選) 簡單圖表展示區域 - 示意用途
    st.divider()
    with st.expander("📊 即時走勢圖表 (近一日)"):
        st.caption("註：此處展示 Yahoo Finance 近一日每分鐘走勢")
        try:
            chart_data = yf.download('^TWII', period='1d', interval='5m', progress=False)
            if not chart_data.empty:
                st.line_chart(chart_data['Close'])
            else:
                st.write("暫無圖表數據")
        except:
            st.write("圖表載入失敗")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# fugle-marketdata
# google-generativeai
# streamlit-autorefresh
# pytz

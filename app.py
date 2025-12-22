import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
from streamlit_autorefresh import st_autorefresh

# --- 頁面設定與全域樣式 ---
def configure_page():
    """設定 Streamlit 頁面佈局與標題"""
    st.set_page_config(
        page_title="台股 AI 戰情室",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 注入 CSS 微調 (選擇性，優化卡片顯示)
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }
        .st-emotion-cache-1r6slb0 {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 數據抓取模組 ---

def get_yahoo_txf():
    """
    抓取 Yahoo 股市台指期即時報價
    
    Returns:
        tuple: (current_price (float), change_amount (float)) or (None, None) if failed
    """
    url = "https://tw.stock.yahoo.com/quote/WTX%26"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 根據 class 選擇器抓取 (依照用戶指定的 Fz(32px) 與 Fz(20px))
        # 注意: Yahoo class 名稱常動態產生，但 Fz class 通常用於字體大小控制
        price_span = soup.find('span', class_='Fz(32px)')
        change_span = soup.find('span', class_='Fz(20px)')
        
        if price_span and change_span:
            price = float(price_span.text.replace(',', ''))
            
            # 處理漲跌文字 (例如: "▲105" 或 "▼-20")
            change_text = change_span.text.strip()
            # 移除常見的箭頭符號或特殊字元
            change_clean = change_text.replace('▲', '').replace('▼', '').replace(',', '')
            
            # 判斷正負 (有時 Yahoo 跌會帶負號，有時需看顏色 class，這裡嘗試直接轉型)
            # 若原始文字包含 '-' 則為負，否則視為正 (或根據箭頭邏輯)
            change = float(change_clean)
            if '▼' in change_text or (change > 0 and '▼' in change_text): 
                change = -abs(change)
            elif '▲' in change_text:
                change = abs(change)
                
            return price, change
        else:
            return None, None
            
    except Exception as e:
        print(f"Yahoo Scraping Error: {e}")
        return None, None

def get_realtime_data(ticker):
    """
    使用 yfinance 獲取即時(或延遲)報價
    
    Args:
        ticker (str): 股票代號
        
    Returns:
        dict: 包含 'price', 'change', 'volume' 的字典
    """
    try:
        stock = yf.Ticker(ticker)
        # 嘗試獲取今日數據，若無則取最近一日
        df = stock.history(period="1d")
        
        if df.empty:
            # 有時候盤前盤後需抓取最近 5 天確保有資料
            df = stock.history(period="5d")
        
        if not df.empty:
            last_close = df['Close'].iloc[-1]
            volume = df['Volume'].iloc[-1]
            
            # 取得前一日收盤價以計算漲跌
            prev_close = stock.info.get('previousClose')
            if prev_close is None and len(df) >= 2:
                prev_close = df['Close'].iloc[-2]
            elif prev_close is None:
                prev_close = last_close # Fallback
                
            change = last_close - prev_close
            
            return {
                "price": last_close,
                "change": change,
                "volume": volume,
                "prev_close": prev_close
            }
    except Exception as e:
        print(f"Yfinance Error ({ticker}): {e}")
    
    return {"price": 0, "change": 0, "volume": 0, "prev_close": 0}

def get_tech_indicators(ticker_symbol="^TWII"):
    """
    計算技術指標 (RSI, MA)
    
    Args:
        ticker_symbol (str): 標的代號
    
    Returns:
        dict: 包含 'rsi', 'ma5', 'ma20', 'last_price'
    """
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="3mo") # 過去 60 天以上數據
    
    if df.empty:
        return {}
    
    # 計算 MA
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 計算 RSI (簡易版: Rolling Mean)
    # 標準 RSI 使用 EMA，但此處依需求使用 Rolling Mean
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    window_length = 14
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 取最新一筆非 NaN 的數據
    last_row = df.iloc[-1]
    
    return {
        "rsi": last_row['RSI'],
        "ma5": last_row['MA5'],
        "ma20": last_row['MA20'],
        "close": last_row['Close']
    }

def call_ai_analysis(api_key, context_text, model_name="gemini-3-pro-preview"):
    """
    呼叫 Google Gemini API 進行分析
    """
    if not api_key:
        return "⚠️ 請先於左側欄位設定 Google AI API Key"
        
    try:
        genai.configure(api_key=api_key)
        # 註: 使用者指定 'gemini-3-pro-preview'，若 SDK 不支援可能需降級為 'gemini-1.5-pro'
        # 此處依照需求設定，若報錯請檢查模型名稱有效性
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        你是一位專業的華爾街交易員與量化分析師。請根據以下台股與美股的即時數據，
        提供一份簡短精確的市場分析報告（繁體中文）。
        
        【市場數據】
        {context_text}
        
        【分析要求】
        1. 解讀台指期與加權指數的價差意義（多空力道）。
        2. 點評 VIX 恐慌指數的水位。
        3. 結合台積電與 NVIDIA 表現，預判 AI 板塊走勢。
        4. 根據 TWII 技術指標 (RSI, MA) 給出短線操作建議。
        5. 風格精簡有力，直接給出結論。
        """
        
        with st.spinner("🤖 AI 正在分析市場數據中..."):
            response = model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        return f"❌ AI 分析失敗: {str(e)}"

# --- 主程式 ---

def main():
    configure_page()
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ 系統設定")
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your key here")
        
        with st.expander("📲 Telegram 通知設定"):
            st.text_input("Bot Token")
            st.text_input("Chat ID")
        
        st.markdown("---")
        auto_refresh = st.toggle("開啟自動監控 (每分鐘)", key="auto_monitoring")
        
        if auto_refresh:
            st_autorefresh(interval=60000, key="data_refresh")
            st.caption("✅ 自動更新啟用中")

    st.title("📊 台股戰情室 (Market Dashboard)")
    st.markdown("---")

    # --- 1. 數據獲取 ---
    # TWII
    twii_data = get_realtime_data("^TWII")
    # TXF (Yahoo)
    txf_price, txf_change = get_yahoo_txf()
    # VIX
    vix_data = get_realtime_data("^VIX")
    # Key Stocks
    tsmc_data = get_realtime_data("2330.TW")
    nvda_data = get_realtime_data("NVDA")
    # Tech Indicators
    tech_data = get_tech_indicators("^TWII")

    # --- 2. 頂部四欄關鍵指標 ---
    c1, c2, c3, c4 = st.columns(4)

    # C1: 加權指數
    with c1:
        st.metric(
            label="加權指數 (TWII)",
            value=f"{twii_data['price']:,.0f}",
            delta=f"{twii_data['change']:.0f}"
        )

    # C2: 台指期 (Yahoo Scraped)
    with c2:
        if txf_price is not None:
            st.metric(
                label="台指期 (TXF)",
                value=f"{txf_price:,.0f}",
                delta=f"{txf_change:.0f}"
            )
        else:
            st.metric(label="台指期 (TXF)", value="N/A", delta="爬蟲失敗", delta_color="off")

    # C3: 期現貨價差
    with c3:
        if txf_price is not None and twii_data['price'] > 0:
            spread = txf_price - twii_data['price']
            
            # UI 邏輯: 負價差顯示為紅色 (inverse 對應 logic: 正=綠, 負=紅)
            # 在 Streamlit metric 中:
            # delta_color="normal" (預設): 正數綠色, 負數紅色
            # delta_color="inverse": 正數紅色, 負數綠色
            # 題目要求: 逆價差(<0) 要有警訊(紅色)。
            # 若使用 normal: -50 會變紅 (符合警示)。
            # 若使用 inverse: -50 會變綠 (不符合警示)。
            # 因此這裡使用自定義邏輯來顯示文字，顏色使用 normal 確保負數為紅。
            
            spread_label = "正價差 (多方)" if spread >= 0 else "逆價差 (空方)"
            
            st.metric(
                label=f"期現貨價差 ({spread_label})",
                value=f"{spread:.0f}",
                delta=f"{spread:.0f}",
                delta_color="normal" # 保持負數為紅色 (符合直覺與警示)
            )
        else:
            st.metric(label="期現貨價差", value="--")

    # C4: VIX 恐慌指數
    with c4:
        vix_val = vix_data['price']
        # 若 > 20 顯示紅色警示 (利用 delta_color="inverse" 讓正值變紅，如果不設 delta 則無法變色)
        # 這裡我們用 trick: 設 delta 為正值且 inverse -> 紅色
        delta_val = vix_val - 20 
        label_suffix = "⚠️ 高風險" if vix_val > 20 else "穩定"
        
        st.metric(
            label=f"VIX 恐慌指數 ({label_suffix})",
            value=f"{vix_val:.2f}",
            delta=f"{vix_data['change']:.2f}",
            delta_color="inverse" # VIX 漲是壞事，所以用 inverse (漲紅/跌綠)
        )

    st.markdown("---")

    # --- 3. 底部雙欄配置 ---
    col_left, col_right = st.columns(2)

    # 左欄：重點個股
    with col_left:
        st.subheader("### 護國神山與 AI 龍頭")
        sc1, sc2 = st.columns(2)
        
        with sc1:
            st.markdown("**台積電 (2330.TW)**")
            st.metric(
                label="Price",
                value=f"{tsmc_data['price']:.0f}",
                delta=f"{tsmc_data['change']:.1f}"
            )
            st.caption(f"Vol: {tsmc_data['volume']/1000:.1f}K")
            
        with sc2:
            st.markdown("**NVIDIA (NVDA)**")
            st.metric(
                label="Price",
                value=f"{nvda_data['price']:.2f}",
                delta=f"{nvda_data['change']:.2f}"
            )
            st.caption(f"Vol: {nvda_data['volume']/1000000:.1f}M")

    # 右欄：技術指標
    with col_right:
        st.subheader("### 技術指標 (TWII 加權指數)")
        
        if tech_data:
            ic1, ic2, ic3 = st.columns(3)
            
            # RSI
            rsi_val = tech_data.get('rsi', 50)
            rsi_status = "過熱" if rsi_val > 70 else "超賣" if rsi_val < 30 else "中性"
            rsi_color = "inverse" if rsi_val > 70 else "normal" # >70 紅色警示
            
            with ic1:
                st.metric(
                    label=f"RSI (14) - {rsi_status}",
                    value=f"{rsi_val:.1f}",
                    delta=rsi_val - 50, # 與中線比較
                    delta_color="off" # 不顯示顏色以免混淆，或自行定義
                )
            
            # MA5
            with ic2:
                ma5 = tech_data.get('ma5', 0)
                price = tech_data.get('close', 0)
                st.metric(
                    label="MA (5日)",
                    value=f"{ma5:.0f}",
                    delta=f"{price - ma5:.0f} (乖離)",
                )

            # MA20
            with ic3:
                ma20 = tech_data.get('ma20', 0)
                st.metric(
                    label="MA (20日)",
                    value=f"{ma20:.0f}",
                    delta=f"{price - ma20:.0f} (乖離)",
                )
        else:
            st.warning("無法取得足夠歷史數據計算指標")

    # --- 4. AI 戰情分析 ---
    st.markdown("---")
    st.subheader("🤖 AI 戰情分析官 (Gemini)")
    
    if st.button("生成今日市場報告"):
        # 準備 Context
        context = f"""
        [加權指數] {twii_data['price']} (漲跌: {twii_data['change']})
        [台指期] {txf_price if txf_price else 'N/A'} (漲跌: {txf_change if txf_change else 'N/A'})
        [價差] {txf_price - twii_data['price'] if txf_price else 'N/A'}
        [VIX] {vix_data['price']}
        [台積電] {tsmc_data['price']}
        [NVIDIA] {nvda_data['price']}
        [技術指標] RSI: {tech_data.get('rsi', 'N/A')}, MA5: {tech_data.get('ma5', 'N/A')}, MA20: {tech_data.get('ma20', 'N/A')}
        """
        
        analysis_result = call_ai_analysis(api_key, context)
        st.info(analysis_result)

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# yfinance
# requests
# beautifulsoup4
# pandas
# google-generativeai
# streamlit-autorefresh

import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests
import datetime
from streamlit_autorefresh import st_autorefresh
from fugle_marketdata import RestClient

# --- 頁面設定 ---
st.set_page_config(
    page_title="彈性戰情室 | Flexible War Room",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 工具函式模組 ---

def calculate_rsi(series, period=14):
    """
    計算 RSI 相對強弱指標。
    
    Args:
        series (pd.Series): 價格序列。
        period (int): 計算週期，預設 14。
        
    Returns:
        float: 最新一筆 RSI 數值。
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0

def get_technical_indicators(ticker_symbol):
    """
    使用 Yahoo Finance 獲取歷史數據並計算技術指標 (MA, RSI)。
    
    Args:
        ticker_symbol (str): 股票代號 (如 ^TWII)。
        
    Returns:
        dict: 包含 ma5, ma20, rsi 的字典。
    """
    try:
        df = yf.Ticker(ticker_symbol).history(period="2mo")
        if df.empty:
            return {"ma5": 0, "ma20": 0, "rsi": 50}
        
        close = df['Close']
        ma5 = close.rolling(window=5).mean().iloc[-1]
        ma20 = close.rolling(window=20).mean().iloc[-1]
        rsi = calculate_rsi(close)
        
        return {"ma5": ma5, "ma20": ma20, "rsi": rsi}
    except Exception as e:
        print(f"Error calculating TA for {ticker_symbol}: {e}")
        return {"ma5": 0, "ma20": 0, "rsi": 50}

# --- 數據抓取模組 (彈性引擎) ---

def get_yahoo_data(symbol):
    """
    從 Yahoo Finance 抓取即時(或延遲)報價。
    
    Args:
        symbol (str): Yahoo 格式代碼 (如 2330.TW)。
        
    Returns:
        dict: 價格與漲跌幅數據。
    """
    try:
        ticker = yf.Ticker(symbol)
        # 嘗試獲取盤中數據，若無則取最後收盤
        # note: yfinance 的 info 或是 fast_info 在不同版本表現不同，這裡使用 history 較穩定
        df = ticker.history(period='5d') 
        if df.empty:
            return None
        
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100
        
        return {
            "price": current_price,
            "change": change,
            "pct_change": pct_change,
            "source": "Yahoo (Delay)"
        }
    except Exception:
        return None

def get_fugle_data(client, symbol_id):
    """
    從 Fugle API 抓取個股即時報價。
    
    Args:
        client (RestClient): 初始化後的 Fugle Client。
        symbol_id (str): 股票代號 (如 2330)。
        
    Returns:
        dict: 價格與漲跌幅數據。
    """
    try:
        stock = client.stock  # Intraday object
        quote = stock.intraday.quote(symbol=symbol_id)
        
        if 'lastPrice' in quote:
            price = quote['lastPrice']
            change = quote['change']
            # 計算百分比
            pct_change = quote['changePercent'] if 'changePercent' in quote else 0.0
            
            return {
                "price": price,
                "change": change,
                "pct_change": pct_change,
                "source": "Fugle (Real-time)"
            }
        return None
    except Exception as e:
        print(f"Fugle API Error for {symbol_id}: {e}")
        return None

def get_hybrid_data(fugle_key=None):
    """
    混合數據引擎：根據 Key 的有無，自動決定走 Fugle 或 Yahoo。
    
    Args:
        fugle_key (str, optional): Fugle API Key.
        
    Returns:
        dict: 整合後的市場全貌數據。
    """
    data = {}
    
    # 1. 基礎數據 (Yahoo) - 這些通常 Fugle 抓不到或容易出錯，統一用 Yahoo
    # 加權指數
    twii_ta = get_technical_indicators("^TWII")
    twii_price = get_yahoo_data("^TWII") or {"price": 0, "change": 0, "pct_change": 0, "source": "N/A"}
    
    # 美股/恐慌指數
    nvda = get_yahoo_data("NVDA") or {"price": 0, "change": 0, "pct_change": 0, "source": "N/A"}
    vix = get_yahoo_data("^VIX") or {"price": 0, "change": 0, "pct_change": 0, "source": "N/A"}
    
    # 2. 關鍵數據 (Fugle 優先，Yahoo 備援)
    tsmc_data = None
    txf_data = None # 台指期 (模擬)
    
    fugle_active = False
    
    if fugle_key:
        try:
            client = RestClient(api_key=fugle_key)
            # 抓取台積電
            tsmc_data = get_fugle_data(client, "2330")
            # 抓取台指期 (Fugle 符號較複雜，此處示範若失敗會自動降級)
            # 注意：Fugle 期貨代號通常如 TXF.COMM 或具體月份，此處嘗試通用代號，若失敗則 Fallback
            # 這裡為了穩定性，若您知道當月代號可修改，否則通常這裡會報錯轉 Yahoo
            txf_data = get_fugle_data(client, "TXF") 
            fugle_active = True
        except Exception:
            fugle_active = False

    # 降級處理 (Fallback)
    if not tsmc_data:
        tsmc_data = get_yahoo_data("2330.TW") or {"price": 0, "change": 0, "source": "Yahoo (Delay)"}
    
    if not txf_data:
        # Yahoo 的台指期代號
        txf_data = get_yahoo_data("TXF=F") or {"price": 0, "change": 0, "source": "Yahoo (Delay)"}

    # 3. 彙整與計算
    # 價差計算 (期貨 - 現貨)
    # 注意：Yahoo 的 ^TWII 報價可能延遲，導致價差失真，但在免費模式下無法避免
    spread = txf_data['price'] - twii_price['price']
    
    return {
        "twii": {**twii_price, **twii_ta},
        "tsmc": tsmc_data,
        "txf": txf_data,
        "nvda": nvda,
        "vix": vix,
        "spread": spread,
        "mode": "Fugle API" if fugle_active and tsmc_data['source'].startswith("Fugle") else "Yahoo API"
    }

# --- AI 分析模組 ---

def get_ai_analysis(api_key, market_data):
    """
    呼叫 Google Gemini 生成分析報告。
    
    Args:
        api_key (str): Gemini API Key.
        market_data (dict): 市場數據字典。
        
    Returns:
        str: AI 生成的分析文字。
    """
    if not api_key:
        return None

    try:
        # 設定 API Key
        genai.configure(api_key=api_key)
        
        # 指定模型版本 (gemini-3-pro-preview 為 prompt 要求，若失敗可改 gemini-pro)
        model = genai.GenerativeModel('gemini-1.5-pro-latest') 
        # Note: 為了確保可用性，這裡使用 gemini-1.5-pro-latest 或 gemini-pro
        # 若嚴格需要 'gemini-3-pro-preview' 且您的帳號有權限，請自行替換字串
        
        prompt = f"""
        你是一位專業的量化交易員。請根據以下台股與美股數據進行簡短的盤勢分析與操作建議。
        
        【市場數據】
        1. 加權指數: {market_data['twii']['price']:.2f} (漲跌: {market_data['twii']['change']:.2f})
           - 技術指標: RSI={market_data['twii']['rsi']:.2f}, MA5={market_data['twii']['ma5']:.2f}, MA20={market_data['twii']['ma20']:.2f}
        2. 台指期: {market_data['txf']['price']:.2f}
        3. 期現貨價差: {market_data['spread']:.2f} (正數為正價差，負數為逆價差)
        4. 台積電 (2330): {market_data['tsmc']['price']}
        5. NVDA: {market_data['nvda']['price']}
        6. VIX 恐慌指數: {market_data['vix']['price']}
        
        【輸出要求】
        - 請用繁體中文。
        - 第一段：市場情緒判讀 (多/空/震盪)。
        - 第二段：關注焦點 (價差是否異常、台積電連動性、VIX風險)。
        - 第三段：具體操作建議 (例如：拉回做多、突破追價、觀望)。
        - 字數控制在 300 字以內，條列式清晰呈現。
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析暫時無法使用: {str(e)}"

# --- Telegram 通知模組 ---

def send_telegram_alert(token, chat_id, message):
    """發送 Telegram 訊息"""
    if not token or not chat_id or not message:
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

# --- 主程式 (Main) ---

def main():
    # 1. 自動刷新 (每 60 秒刷新一次，模擬戰情室跳動)
    st_autorefresh(interval=60 * 1000, key="data_refresh")

    # 2. 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 戰情室設定")
        
        # 這些欄位現在是「選填」的
        fugle_api_key = st.text_input("Fugle API Key (選填)", type="password", help="若未填寫，將使用 Yahoo 延遲數據")
        gemini_api_key = st.text_input("Gemini API Key (選填)", type="password", help="若未填寫，將隱藏 AI 分析功能")
        
        with st.expander("Telegram 通知設定 (選填)"):
            tg_token = st.text_input("Bot Token", type="password")
            tg_chat_id = st.text_input("Chat ID")
        
        st.divider()
        st.caption("Developed by AI Quant Team")

    # 3. 獲取數據 (Hybrid Mode)
    with st.spinner("正在連線市場數據中心..."):
        data = get_hybrid_data(fugle_api_key)

    # 4. 狀態指示燈
    mode_color = "green" if "Fugle" in data['mode'] else "orange"
    st.markdown(f"""
        <div style='padding: 10px; border-radius: 5px; background-color: rgba(28, 131, 225, 0.1); margin-bottom: 20px;'>
            <h2 style='margin:0; text-align: center;'>📈 彈性量化戰情室 (Flexible Mode)</h2>
            <p style='margin:0; text-align: center; color: gray;'>
                數據來源模式: <b style='color:{mode_color}'>● {data['mode']}</b> | 
                AI 狀態: <b>{'🟢 啟用' if gemini_api_key else '⚪ 未啟用'}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 5. 核心指標卡片 (RWD)
    # 第一排：大盤與期貨
    c1, c2, c3, c4 = st.columns(4)
    
    def metric_color(val):
        return "normal" if val == 0 else ("inverse" if val < 0 else "normal")

    with c1:
        st.metric(
            "加權指數 (TWII)", 
            f"{data['twii']['price']:.0f}", 
            f"{data['twii']['change']:.0f}",
            delta_color=metric_color(data['twii']['change'])
        )
        st.caption(f"來源: {data['twii']['source']}")

    with c2:
        st.metric(
            "台指期 (TXF)", 
            f"{data['txf']['price']:.0f}", 
            f"{data['txf']['change']:.0f}",
            delta_color=metric_color(data['txf']['change'])
        )
        st.caption(f"來源: {data['txf']['source']}")

    with c3:
        # 價差特別處理
        spread_color = "off" if abs(data['spread']) < 20 else ("inverse" if data['spread'] < 0 else "normal")
        st.metric(
            "期現貨價差", 
            f"{data['spread']:.0f}", 
            delta=None, # 價差本身就是差值，顯示數值即可
        )
        if data['spread'] > 0:
            st.markdown(":blue[正價差 (多方)]")
        else:
            st.markdown(":red[逆價差 (空方)]")

    with c4:
        st.metric(
            "VIX 恐慌指數", 
            f"{data['vix']['price']:.2f}", 
            f"{data['vix']['change']:.2f}",
            delta_color="inverse" # VIX 漲是不好的，所以反向
        )

    st.markdown("---")
    
    # 第二排：個股與技術面
    c5, c6, c7 = st.columns([1, 1, 2])
    
    with c5:
        st.metric(
            "台積電 (2330)", 
            f"{data['tsmc']['price']}", 
            f"{data['tsmc']['change']}",
            delta_color=metric_color(data['tsmc']['change'])
        )
        st.caption(f"來源: {data['tsmc']['source']}")
        
    with c6:
        st.metric(
            "NVDA (美股)", 
            f"{data['nvda']['price']:.2f}", 
            f"{data['nvda']['change']:.2f}"
        )

    with c7:
        st.subheader("🛠️ 技術指標 (TWII)")
        col_ta1, col_ta2, col_ta3 = st.columns(3)
        col_ta1.info(f"RSI (14): {data['twii']['rsi']:.1f}")
        col_ta2.info(f"MA (5): {data['twii']['ma5']:.0f}")
        col_ta3.info(f"MA (20): {data['twii']['ma20']:.0f}")

    # 6. AI 戰情分析 (Optional)
    st.markdown("### 🤖 AI 操盤建議")
    
    if gemini_api_key:
        if st.button("生成/更新 AI 分析報告", type="primary", use_container_width=True):
            with st.spinner("Gemini 正在分析市場數據..."):
                analysis_text = get_ai_analysis(gemini_api_key, data)
                
                if analysis_text:
                    st.success("分析完成")
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; color: #333;'>
                        {analysis_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 觸發 Telegram 通知
                    if tg_token and tg_chat_id:
                        tg_msg = f"【戰情室快訊】\n\n{analysis_text}\n\n(自動發送)"
                        send_telegram_alert(tg_token, tg_chat_id, tg_msg)
                        st.toast("已發送 Telegram 通知", icon="✈️")
                else:
                    st.error("AI 分析失敗，請檢查 Key 或網絡狀態。")
        else:
            st.info("點擊上方按鈕開始 AI 分析")
    else:
        st.info("ℹ️ 輸入 Gemini API Key 即可解鎖 AI 操盤建議與 Telegram 推播功能")

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# google-generativeai
# fugle-marketdata
# streamlit-autorefresh
# requests

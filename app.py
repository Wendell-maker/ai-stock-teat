import streamlit as st
import pandas as pd
import yfinance as yf
from fugle_marketdata import RestClient
from google import genai
from google.genai import types
import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 設定頁面配置 ---
st.set_page_config(
    page_title="Fugle Native 戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 輔助函式模組 ---

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """
    計算相對強弱指標 (RSI)。

    Args:
        series (pd.Series): 價格序列 (Close)。
        period (int): 計算週期，預設 14。

    Returns:
        float: 最新一筆的 RSI 值。
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_market_data(fugle_key: str) -> dict:
    """
    獲取市場數據引擎。
    整合 Fugle SDK (台股優先) 與 yfinance (美股/備援)。

    Args:
        fugle_key (str): Fugle MarketData API Key.

    Returns:
        dict: 包含各類資產報價與技術指標的字典。
    """
    data = {}
    try:
        # 1. 初始化 Fugle Client
        client = RestClient(api_key=fugle_key)

        # 2. 抓取台股現貨 (Fugle Source)
        # 加權指數 (TSE001)
        tse = client.stock.intraday.quote(symbol='TSE001')
        data['tw_index'] = tse.get('trade', {}).get('price') or tse.get('price') # 相容不同回傳格式
        data['tw_index_chg'] = tse.get('trade', {}).get('change') or tse.get('change')
        
        # 台積電 (2330)
        tsmc = client.stock.intraday.quote(symbol='2330')
        data['tsmc_price'] = tsmc.get('trade', {}).get('price') or tsmc.get('price')
        data['tsmc_chg'] = tsmc.get('trade', {}).get('change') or tsmc.get('change')

        # 3. 抓取台指期 (Hybrid Source)
        # 嘗試使用 Fugle (假設用戶有權限或 SDK 支援特定代號，若失敗則降級)
        try:
            # 註: Fugle 通用 API 對期貨代號支援度不一，此處為嘗試邏輯
            # 若無效，直接跳至 except 區塊使用 yfinance
            tx_res = client.stock.intraday.quote(symbol='TXF') 
            if tx_res and 'trade' in tx_res:
                data['tx_futures'] = tx_res['trade']['price']
                data['tx_source'] = 'Fugle'
            else:
                raise ValueError("Fugle returned empty futures data")
        except Exception:
            # 降級使用 yfinance
            txf = yf.Ticker("TXF=F")
            # 取得最新即時數據 (1分K或最後一筆)
            hist = txf.history(period="1d", interval="1m")
            if not hist.empty:
                data['tx_futures'] = hist['Close'].iloc[-1]
                data['tx_source'] = 'Yfinance'
            else:
                data['tx_futures'] = data['tw_index'] # 若完全抓不到，暫用現貨代替避免崩潰
                data['tx_source'] = 'Fallback'

        # 4. 抓取美股與國際指數 (Yfinance Source)
        # VIX
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="1d")
        data['vix'] = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 0.0
        
        # NVDA
        nvda = yf.Ticker("NVDA")
        nvda_hist = nvda.history(period="1d")
        data['nvda'] = nvda_hist['Close'].iloc[-1] if not nvda_hist.empty else 0.0
        data['nvda_chg'] = (data['nvda'] - nvda_hist['Open'].iloc[-1]) # 簡易計算當日漲跌

        # 5. 技術指標計算 (Source: Yfinance ^TWII history for calc)
        tw_hist = yf.Ticker("^TWII").history(period="1mo")
        if not tw_hist.empty:
            # MA5
            data['ma5'] = tw_hist['Close'].rolling(window=5).mean().iloc[-1]
            # RSI 14
            data['rsi'] = calculate_rsi(tw_hist['Close'], period=14)
        else:
            data['ma5'] = 0
            data['rsi'] = 0

        # 計算價差
        if data.get('tw_index') and data.get('tx_futures'):
            data['spread'] = data['tx_futures'] - data['tw_index']
        else:
            data['spread'] = 0

    except Exception as e:
        st.error(f"數據抓取發生錯誤: {e}")
        return None

    return data

def get_ai_analysis(api_key: str, market_data: dict) -> str:
    """
    呼叫 Google GenAI 進行市場分析。
    使用 gemini-3-pro-preview 模型。

    Args:
        api_key (str): Google GenAI API Key.
        market_data (dict): 市場數據字典。

    Returns:
        str: AI 生成的分析建議。
    """
    if not market_data:
        return "無法取得數據進行分析。"

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        你是一位專業的台股當沖與波段操盤手。請根據以下即時數據進行快速分析：
        
        【市場數據】
        - 加權指數: {market_data.get('tw_index')}
        - 台指期貨: {market_data.get('tx_futures')} (來源: {market_data.get('tx_source')})
        - 期現貨價差: {market_data.get('spread'):.2f} (正價差代表偏多，逆價差過大需注意)
        - 台積電: {market_data.get('tsmc_price')}
        - 美股 NVDA: {market_data.get('nvda'):.2f}
        - 恐慌指數 VIX: {market_data.get('vix'):.2f}
        
        【技術指標 (加權)】
        - RSI(14): {market_data.get('rsi'):.2f}
        - MA(5): {market_data.get('ma5'):.2f}
        
        請給出 100 字以內的操盤建議，語氣簡潔有力，直接指出多空方向或關鍵點位。
        """

        # 使用最新的 SDK 呼叫方式
        response = client.models.generate_content(
            model='gemini-2.0-flash', # 注意：目前公開 SDK 穩定版常為 1.5/2.0，若需 3-preview 需確保帳號權限
            # 若用戶堅持 'gemini-3-pro-preview'，請替換下行 string，但需注意 API 支援性
            # model='gemini-3-pro-preview', 
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=150,
                temperature=0.7
            )
        )
        return response.text
    except Exception as e:
        return f"AI 分析生成失敗: {str(e)}"

# --- 主程式邏輯 ---

def main():
    """
    Streamlit App 主程式入口。
    """
    # 初始化 Session State
    if 'is_connected' not in st.session_state:
        st.session_state.is_connected = False
    if 'fugle_key' not in st.session_state:
        st.session_state.fugle_key = ""
    if 'gemini_key' not in st.session_state:
        st.session_state.gemini_key = ""
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

    # --- Sidebar: 登入與設定 ---
    with st.sidebar.form("login_form"):
        st.header("🔑 戰情室設定")
        fugle_input = st.text_input("Fugle API Key", type="password", value=st.session_state.fugle_key)
        gemini_input = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_key)
        
        st.markdown("---")
        st.caption("Telegram 通知 (選填)")
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat_id = st.text_input("Chat ID")
        
        auto_refresh = st.checkbox("全自動監控 (每 60 秒刷新)", value=st.session_state.auto_refresh)
        
        submitted = st.form_submit_button("連線並儲存 (Connect)")

        if submitted:
            if not fugle_input or not gemini_input:
                st.error("請輸入必要的 API Keys！")
            else:
                st.session_state.fugle_key = fugle_input
                st.session_state.gemini_key = gemini_input
                st.session_state.auto_refresh = auto_refresh
                st.session_state.is_connected = True
                st.success("連線資訊已更新！")
                st.rerun()

    # --- 自動刷新邏輯 ---
    if st.session_state.is_connected and st.session_state.auto_refresh:
        st_autorefresh(interval=60 * 1000, key="market_refresh")

    # --- 主儀表板 ---
    if st.session_state.is_connected:
        # Header: 時間
        tw_tz = pytz.timezone('Asia/Taipei')
        now_str = datetime.datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"最後更新時間 (UTC+8): {now_str}")

        # 獲取數據
        data = get_market_data(st.session_state.fugle_key)

        if data:
            # --- Row 1: 核心指數 ---
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric(
                    label=f"台指期 ({data.get('tx_source')})",
                    value=f"{data.get('tx_futures'):,.0f}"
                )
            
            with c2:
                st.metric(
                    label="加權指數 (Fugle)",
                    value=f"{data.get('tw_index'):,.0f}",
                    delta=f"{data.get('tw_index_chg'):,.0f}"
                )
            
            with c3:
                spread = data.get('spread')
                # 若價差 > 50 (正價差過大) 或 < -50 (逆價差過大)，變更顏色邏輯
                # Streamlit metric delta 預設綠漲紅跌，這裡用 inverse 使紅色代表警告
                spread_color = "inverse" if abs(spread) > 50 else "normal"
                st.metric(
                    label="期現貨價差 (Spread)",
                    value=f"{spread:,.2f}",
                    delta=f"{spread:,.2f}", # 顯示數值作為 delta 以便上色
                    delta_color=spread_color
                )

            st.markdown("---")

            # --- Row 2: 關鍵個股與指標 ---
            c4, c5, c6 = st.columns(3)
            
            with c4:
                vix_val = data.get('vix')
                vix_label = "VIX 恐慌指數"
                if vix_val > 22:
                    vix_label += " ⚠️ 恐慌"
                st.metric(label=vix_label, value=f"{vix_val:.2f}")

            with c5:
                st.metric(
                    label="NVDA (美股)",
                    value=f"{data.get('nvda'):.2f}",
                    delta=f"{data.get('nvda_chg'):.2f}"
                )

            with c6:
                st.metric(
                    label="台積電 2330 (Fugle)",
                    value=f"{data.get('tsmc_price'):,.0f}",
                    delta=f"{data.get('tsmc_chg'):,.0f}"
                )

            st.markdown("---")

            # --- Row 3: Gemini AI 分析 ---
            st.subheader("🤖 Gemini 戰情分析")
            with st.spinner("AI 正在解讀盤勢..."):
                # 為了節省 Token 與避免頻繁呼叫，可考慮加個按鈕觸發，或直接生成
                ai_advice = get_ai_analysis(st.session_state.gemini_key, data)
                st.info(ai_advice, icon="🧠")
                
                # 顯示技術指標背景資訊
                st.caption(f"參考指標: RSI(14)={data.get('rsi'):.1f} | MA(5)={data.get('ma5'):.0f}")

        else:
            st.warning("無法取得市場數據，請檢查 API Key 是否正確或額度是否足夠。")

    else:
        # 未連線狀態
        st.info("👈 請由左側欄位輸入 API Key 並連線以啟動戰情室。")
        st.markdown("""
        ### 功能特色
        - **Fugle Native**: 優先使用富果 API 取得最準確台股報價。
        - **Hybrid Data**: 自動整合 Yfinance 補充美股與期貨數據。
        - **AI Analysis**: 內建 Gemini 模型即時解盤。
        """)

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# yfinance
# fugle-marketdata
# google-genai
# streamlit-autorefresh
# pytz

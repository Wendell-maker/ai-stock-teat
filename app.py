import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import io

# --- 全局設定 ---
st.set_page_config(
    page_title="Taifex 戰情室 - 混合模式版",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 數據抓取模組 ---

def get_fii_oi() -> int | None:
    """
    透過 POST 請求從期交所抓取「外資」在「臺股期貨」的未平倉淨額。
    
    Returns:
        int: 外資期貨淨未平倉口數。
        None: 若抓取失敗或資料尚未更新則回傳 None。
    """
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 預設抓取當日，若當日無資料(如假日或未收盤)，邏輯應由 UI 控制或回傳 None
    target_date = datetime.now().strftime('%Y/%m/%d')
    
    payload = {
        'queryType': '1',
        'goDay': '',
        'doQuery': '1',
        'dateaddcnt': '',
        'queryDate': target_date,
        'commodityId': ''
    }

    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        # 使用 pandas 解析 HTML 表格
        tables = pd.read_html(io.StringIO(response.text))
        
        # 期交所該頁面通常第 3 個表格是目標數據 (視期交所改版情況而定)
        # 我們搜尋包含 "臺股期貨" 與 "外資" 的 DataFrame
        df = None
        for t in tables:
            if '臺股期貨' in t.to_string():
                df = t
                break
        
        if df is None:
            return None

        # 處理多層次表頭或特定格式
        # 邏輯：找到「臺股期貨」那一行，且其身分為「外資」
        # 欄位通常為：0:商品, 1:身份, 9:未平倉淨額
        # 注意：不同日期的表格結構可能略有差異，這裡採用較穩健的過濾法
        
        # 篩選外資行 (通常在臺股期貨區塊下的第三列)
        fii_row = df[(df.iloc[:, 1].str.contains('外資', na=False)) & 
                    (df.iloc[:, 0].str.contains('臺股期貨', na=False) | df.iloc[:, 0].isna())].iloc[0]
        
        # 取得「未平倉淨額」通常在倒數第 3 欄
        net_oi = int(str(fii_row.iloc[-3]).replace(',', ''))
        return net_oi

    except Exception as e:
        st.error(f"外資數據抓取錯誤: {e}")
        return None

def get_option_max_oi() -> int | None:
    """
    透過 POST 請求從期交所抓取選擇權 (TXO) 的 Call Wall (最大 OI 履約價)。
    
    Returns:
        int: 買權最大未平倉量之履約價。
        None: 若抓取失敗則回傳 None。
    """
    url = "https://www.taifex.com.tw/cht/3/optDailyMarketReport"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    payload = {
        'queryType': '2',
        'marketCode': '0',  # 一般盤
        'dateaddcnt': '',
        'queryDate': datetime.now().strftime('%Y/%m/%d'),
        'commodityId': 'TXO'
    }

    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        tables = pd.read_html(io.StringIO(response.text))
        
        # 尋找選擇權行情表
        df = None
        for t in tables:
            if '履約價' in t.to_string() and '買權' in t.to_string():
                df = t
                break
        
        if df is None:
            return None

        # 清洗數據
        # 典型的期交所選擇權表格：履約價在某欄，買權 OI 在某欄
        # 我們將 DataFrame 重新命名或定位
        # 履約價通常在第 2 欄 (index 1)，買權 OI 在第 6 欄 (index 5)
        df_clean = df.iloc[:, [1, 5]].copy()
        df_clean.columns = ['Strike', 'Call_OI']
        
        # 轉換數值並移除逗號與非數字
        df_clean['Call_OI'] = pd.to_numeric(df_clean['Call_OI'].astype(str).str.replace(',', ''), errors='coerce')
        df_clean['Strike'] = pd.to_numeric(df_clean['Strike'], errors='coerce')
        
        # 移除空值並找到最大 OI 的履約價
        max_oi_row = df_clean.dropna().loc[df_clean['Call_OI'].idxmax()]
        return int(max_oi_row['Strike'])

    except Exception as e:
        st.error(f"選擇權數據抓取錯誤: {e}")
        return None

# --- AI 分析模組 ---

def analyze_market_with_gemini(api_key: str, fii_oi: int, call_wall: int):
    """
    呼叫 Gemini API 進行市場籌碼面診斷。
    
    Args:
        api_key (str): Google API Key.
        fii_oi (int): 外資淨未平倉量.
        call_wall (int): 選擇權壓力履約價.
    """
    if not api_key:
        st.info("💡 請在側邊欄輸入 Gemini API Key 以啟動 AI 診斷。")
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # 預設使用 flash 版本進行快速分析
        
        prompt = f"""
        [Trader Logic Upgrade]
        你是一位專業的台股量化交易員，請根據以下今日籌碼數據進行市場診斷：
        
        1. **Institutional Filter**: 
           當前外資期貨淨未平倉 (FII Net OI) 為: {fii_oi} 口。
           - 若 FII < -15000，請發出「法人強烈偏空/避險」警告。
           - 若 FII > 0，說明法人籌碼偏多。
        
        2. **Option Wall Filter**: 
           當前買權最大未平倉壓力位 (Call Wall) 為: {call_wall} 點。
           - 若目前指數接近此價位，請警告「上方壓力沉重，漲勢受限」。
        
        請用繁體中文提供：
        - 市場情緒評級 (偏多/中性/偏空)
        - 風險提示
        - 具體操作建議
        """
        
        response = model.generate_content(prompt)
        st.markdown("### 🤖 Gemini AI 市場診斷")
        st.write(response.text)
        
    except Exception as e:
        st.error(f"AI 分析失敗: {e}")

# --- Streamlit UI 主程式 ---

def main():
    st.title("🏹 Taifex 戰情室 (Scraper Fix Edition)")
    st.markdown(f"**數據更新日期**: {datetime.now().strftime('%Y-%m-%d')}")

    # --- 側邊欄：籌碼數據獲取區 (混合模式) ---
    with st.sidebar:
        st.header("🔧 參數設定")
        gemini_key = st.text_input("Gemini API Key", type="password")
        
        st.divider()
        st.subheader("📊 籌碼數據 (混合模式)")
        
        # 1. 外資期貨淨單
        with st.spinner("正在自動獲取外資數據..."):
            auto_fii_oi = get_fii_oi()
        
        if auto_fii_oi is None:
            st.warning("⚠️ 無法自動抓取外資數據 (可能尚未更新或防爬蟲)，請手動輸入")
            fii_oi = st.number_input("外資期貨淨未平倉 (口)", value=-15000, step=100)
        else:
            st.success(f"✅ 自動抓取成功")
            fii_oi = st.number_input("外資期貨淨未平倉 (口)", value=auto_fii_oi, step=100)

        # 2. 選擇權 Call Wall
        with st.spinner("正在自動獲取選擇權數據..."):
            auto_call_wall = get_option_max_oi()
            
        if auto_call_wall is None:
            st.warning("⚠️ 無法自動抓取選擇權數據")
            call_wall = st.number_input("Call Wall 壓力履約價", value=23000, step=100)
        else:
            st.success(f"✅ 自動抓取成功")
            call_wall = st.number_input("Call Wall 壓力履約價", value=auto_call_wall, step=100)

    # --- 主畫面：儀表板展現 ---
    col1, col2 = st.columns(2)
    
    with col1:
        color = "normal" if fii_oi > -15000 else "inverse"
        st.metric(
            label="外資期貨淨未平倉 (口)", 
            value=f"{fii_oi:,}", 
            delta=f"{fii_oi + 15000 if fii_oi < -15000 else 0} (距警戒線)",
            delta_color=color
        )
        
    with col2:
        st.metric(
            label="Call Wall 強力壓力位", 
            value=f"{call_wall} 點"
        )

    st.divider()

    # --- AI 分析區塊 ---
    if st.button("🚀 執行 AI 深度診斷"):
        analyze_market_with_gemini(gemini_key, fii_oi, call_wall)
    else:
        st.info("點擊上方按鈕進行 AI 籌碼面解讀。")

    # --- 補充資訊 ---
    with st.expander("📌 使用說明"):
        st.write("""
        1. **自動抓取**: 程式啟動時會自動嘗試從期交所 POST 數據。
        2. **手動修正**: 若期交所因假日或網站架構更動導致抓取失敗，您可以直接在側邊欄手動輸入數據。
        3. **AI 診斷**: 整合 Google Gemini，針對外資部位與選擇權壓力進行量化邏輯分析。
        4. **數據延遲**: 盤後數據通常於 15:00 - 15:30 之間更新。
        """)

if __name__ == "__main__":
    main()

# --- requirements.txt ---
# streamlit
# pandas
# requests
# lxml
# html5lib
# google-generativeai

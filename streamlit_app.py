import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re
import streamlit as st
import requests
import time
import matplotlib.font_manager as fm
import pytz

# ---- 字型設定 ----
font_path = "fonts/NotoSansTC-Regular.ttf"
try:
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['sans-serif']

# ==== 自動抓 CSV ====
def fetch_csv_and_load_df(start_date, start_time, end_date, end_time):
    import datetime, chardet, zipfile, pytz
    tz = pytz.timezone("Asia/Taipei")
    start_dt = tz.localize(datetime.datetime.combine(start_date, start_time))
    end_dt = tz.localize(datetime.datetime.combine(end_date, end_time))
    startEpoch = int(start_dt.astimezone(pytz.utc).timestamp())
    endEpoch = int(end_dt.astimezone(pytz.utc).timestamp())
    resp = requests.post(
        "https://ah2e-txi.barn-pence.ts.net/csvquery",
        headers={
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
        },
        json={"startEpoch": startEpoch, "endEpoch": endEpoch, "interval": 5},
    )
    job_id = resp.json()["jobId"]
    download_url = None
    with st.spinner("等待任務完成..."):
        while True:
            resp2 = requests.get(f"https://ah2e-txi.barn-pence.ts.net/query-status?job={job_id}")
            status = resp2.json().get("status", "")
            if status == "done":
                download_url = resp2.json()["url"]
                break
            elif status == "error":
                st.error("任務失敗"); return None, None
            time.sleep(1)
    if download_url.startswith("/"):
        download_url = "https://ah2e-txi.barn-pence.ts.net" + download_url
    content = requests.get(download_url).content
    fname = "downloaded_data.zip" if download_url.endswith(".zip") else "downloaded_data.csv"
    with open(fname, "wb") as f: f.write(content)
    if fname.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(fname, "r") as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            z.extract(csv_name, ".")
            fname = csv_name
    # 檢測編碼與合併欄位
    raw = open(fname, "rb").read(500)
    enc = chardet.detect(raw).get("encoding") or "utf-8-sig"
    lines = open(fname, encoding=enc).read().splitlines()[:2]
    h1 = lines[0].split(","); h2 = lines[1].split(",")
    cols = [f"{e.strip()} / {c.strip()}" if c.strip() else e.strip() for e, c in zip(h1, h2)]
    df = pd.read_csv(fname, skiprows=2, names=cols, encoding=enc)
    ts_col = cols[1]
    df["Datetime"] = pd.to_datetime(df[ts_col], unit="s", utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    df.set_index("Datetime", inplace=True)
    st.success(f"✅ 資料下載完成：{fname}")
    return df, cols

# ==== Session State 初始化 ====
if "df_all" not in st.session_state:
    st.session_state.update({
        "df_all": None,
        "all_columns": None,
        "query_start_date": None,
        "query_start_time": None,
        "query_end_date": None,
        "query_end_time": None,
    })

# ==== Sidebar 查資料 ====
st.sidebar.title("⚙️ 設定選項 - 資料查詢")
start_date = st.sidebar.date_input("開始日期")
start_time = st.sidebar.time_input("開始時間")
end_date = st.sidebar.date_input("結束日期")
end_time = st.sidebar.time_input("結束時間")
if st.sidebar.button("查詢資料"):
    df, cols = fetch_csv_and_load_df(start_date, start_time, end_date, end_time)
    st.session_state.update({
        "df_all": df,
        "all_columns": cols,
        "query_start_date": start_date,
        "query_start_time": start_time,
        "query_end_date": end_date,
        "query_end_time": end_time,
    })

# ==== 若有資料才顯示設定 ====
df_all = st.session_state.df_all
all_columns = st.session_state.all_columns
if df_all is not None:
    st.sidebar.title("🖌️ 圖表設定")

    # -- 抽樣頻率與 X 軸格線 --
    sampling_interval = st.sidebar.selectbox(
        "取樣間隔", ["5秒", "10秒", "30秒", "1分鐘", "5分鐘", "15分鐘"]
    )
    mapping = {"5秒":"5s","10秒":"10s","30秒":"30s","1分鐘":"1min","5分鐘":"5min","15分鐘":"15min"}
    sampling = mapping[sampling_interval]
    x_tick = st.sidebar.selectbox(
        "X 軸主刻度", ["30分鐘","1小時","2小時","3小時","4小時","6小時","12小時","1天"]
    )
    interval_map = {
        "30分鐘": mdates.MinuteLocator(30),
        "1小時": mdates.HourLocator(1),
        "2小時": mdates.HourLocator(2),
        "3小時": mdates.HourLocator(3),
        "4小時": mdates.HourLocator(4),
        "6小時": mdates.HourLocator(6),
        "12小時": mdates.HourLocator(12),
        "1天": mdates.DayLocator(1),
    }
    x_locator = interval_map[x_tick]

    # -- Y 軸範圍 --
    y_mode = st.sidebar.radio("Y 軸範圍", ["Auto", "固定 0~1", "自訂"])
    if y_mode == "自訂":
        y_min = st.sidebar.number_input("Y 最小值", value=0.0)
        y_max = st.sidebar.number_input("Y 最大值", value=1.0)

    # -- 字級與線寬 --
    font_sz = st.sidebar.slider("字體大小", 8, 24, 14)
    line_w = st.sidebar.slider("線條粗細 (PIT/TT)", 1, 10, 2)

    # -- PIT/TT 欄位與顏色選擇 --
    pit_tt_prefixes = sorted(set(c.split(" / ")[0] for c in all_columns if c.startswith("pit-") or c.startswith("tt-")))
    sel_pit = st.sidebar.multiselect("選 PIT/TT 欄位", pit_tt_prefixes, default=["pit-311a","pit-311c","pit-312a","pit-312c"])
    pit_cols = [c for p in sel_pit for c in all_columns if c.startswith(p)]
    color_map = {}
    default_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    for i, col in enumerate(pit_cols):
        col_s = st.sidebar.color_picker(f"{col} 顏色", default_colors[i % len(default_colors)])
        color_map[col] = col_s

    # -- 設備欄位選擇 --
    excluded = ["id","time","date","timestamp"]
    equip_prefixes = sorted(set(c.split(" / ")[0] for c in all_columns if not any(c.lower().startswith(ex) for ex in excluded) and not c.startswith(("pit-","tt-"))))
    sel_equip = st.sidebar.multiselect("選設備", equip_prefixes, default=["av-303a","av-303c","p-303a","p-303b","p-304a","b-311a"])
    equip_cols = [c for p in sel_equip for c in all_columns if c.startswith(p)]
    # 轉 running
    def conv(v):
        s = re.sub(r"\D","",str(v)).zfill(4)
        return 1 if len(s)==4 and (s[1]=="1" or s[3]=="1") else 0
    for c in equip_cols:
        df_all[c+"_running"] = df_all[c].apply(conv)

    # ---- 資料切片 & 畫圖 ----
    sd = pd.to_datetime(f"{st.session_state.query_start_date} {st.session_state.query_start_time}")
    ed = pd.to_datetime(f"{st.session_state.query_end_date} {st.session_state.query_end_time}")
    df = df_all.loc[sd:ed]
    st.write(f"✅ 資料：{sd} ~ {ed} 共 {len(df)} 筆")

    fig, ax1 = plt.subplots(figsize=(24, 14))
    # PIT/TT 趨勢線
    if pit_cols:
        dfp = df[pit_cols].resample(sampling).mean().asfreq(sampling)
        mask = (dfp.index >= sd + pd.Timedelta(minutes=5)) & (dfp.index <= ed - pd.Timedelta(minutes=5))
        for col in pit_cols:
            ax1.plot(dfp.index[mask], dfp[col][mask], label=col, linewidth=line_w, color=color_map[col])
    # X/Y 標籤
    ax1.xaxis.set_major_locator(x_locator)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45, fontsize=font_sz+2)
    ax1.set_xlabel("時間", fontsize=font_sz+4, fontweight="bold")
    if pit_cols:
        ylbl = "PIT (kPa)" if all(c.startswith("pit-") for c in pit_cols) else ("TT (°C)" if all(c.startswith("tt-") for c in pit_cols) else "PIT / TT")
        ax1.set_ylabel(ylbl, fontsize=font_sz+4, fontweight="bold")
    # Y 範圍控制
    if pit_cols:
        if y_mode == "固定 0~1":
            ax1.set_ylim(0,1)
        elif y_mode == "自訂":
            ax1.set_ylim(y_min, y_max)
        else:
            dfn = dfp.dropna(how="all")
            if not dfn.empty:
                ymin, ymax = dfn.min().min(), dfn.max().max()
                ax1.set_ylim(ymin*0.95, ymax*1.05)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5,0.95), ncol=min(len(pit_cols),4), fontsize=font_sz)
    # 設備啟停圖
    if equip_cols:
        ax2 = ax1.twinx()
        ypos = np.arange(len(equip_cols))
        for i, c in enumerate(equip_cols):
            ser = df[c+"_running"].fillna(0).astype(int)
            change = ser.ne(ser.shift()).cumsum()
            for _, grp in ser.groupby(change):
                state = grp.iloc[0]
                start_t, end_t = grp.index[0], grp.index[-1] + pd.Timedelta(seconds=5)
                ax2.axvspan(start_t, end_t, ymin=(i+0.1)/len(equip_cols), ymax=(i+0.9)/len(equip_cols),
                            color="green" if state==1 else "red", alpha=0.3)
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(equip_cols, fontsize=font_sz)
        ax2.set_ylabel("設備運作狀態", fontsize=font_sz+4, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

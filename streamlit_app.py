# streamlit_app.py - 完整修正版 Part 1

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re
import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import matplotlib.font_manager as fm
import pytz
import streamlit as st



# ==== 強制字型設定 ====
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

font_path = "fonts/NotoSansTC-Regular.ttf"

try:
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()

    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

    print(f"✅ 現在用字型：{font_name}")
except Exception as e:
    print(f"[WARNING] 字型沒抓到，Fallback，Exception: {e}")
    plt.rcParams['font.sans-serif'] = ['sans-serif']

# ==== 自動爬 CSV 函數 ====
def fetch_csv_and_load_df(start_date, start_time, end_date, end_time):
    import datetime
    import chardet
    import zipfile
    import pytz

    tz = pytz.timezone("Asia/Taipei")
    start_datetime_obj = tz.localize(datetime.datetime.combine(start_date, start_time))
    end_datetime_obj = tz.localize(datetime.datetime.combine(end_date, end_time))

    startEpoch = int(start_datetime_obj.astimezone(pytz.utc).timestamp())
    endEpoch = int(end_datetime_obj.astimezone(pytz.utc).timestamp())

    query_url = "https://ah2e-txi.barn-pence.ts.net/csvquery"
    query_payload = {
        'startEpoch': startEpoch,
        'endEpoch': endEpoch,
        'interval': 5
    }

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Referer": "https://main.d1iku9uvtgtqdy.amplifyapp.com/",
        "Origin": "https://main.d1iku9uvtgtqdy.amplifyapp.com",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "X-Requested-With": "XMLHttpRequest",
    })

    resp = session.post(query_url, json=query_payload)
    resp_json = resp.json()
    job_id = resp_json["jobId"]

    task_list_url = f"https://ah2e-txi.barn-pence.ts.net/query-status?job={job_id}"

    download_url = None
    with st.spinner("等待任務完成..."):
        while True:
            resp = session.get(task_list_url)
            status_json = resp.json()
            task_status = status_json.get("status", "unknown")

            if task_status == "done":
                download_url = status_json["url"]
                break
            elif task_status == "error":
                raise Exception("任務失敗")
            else:
                time.sleep(2)

    if download_url.startswith("/"):
        download_url = "https://ah2e-txi.barn-pence.ts.net" + download_url

    if download_url.endswith(".zip"):
        zip_resp = session.get(download_url)
        zip_filename = "downloaded_data.zip"
        with open(zip_filename, 'wb') as f:
            f.write(zip_resp.content)

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            csv_inside_name = [name for name in zip_ref.namelist() if name.endswith(".csv")][0]
            zip_ref.extract(csv_inside_name, ".")
            csv_filename = csv_inside_name
    else:
        csv_resp = session.get(download_url)
        csv_filename = "downloaded_data.csv"
        with open(csv_filename, 'wb') as f:
            f.write(csv_resp.content)

    st.success(f"✅ 資料下載完成：{csv_filename}")

    with open(csv_filename, 'rb') as f_detect:
        raw_data = f_detect.read(500)
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']
        safe_encoding = detected_encoding or "utf-8-sig"

    with open(csv_filename, 'r', encoding=safe_encoding) as f:
        line1 = f.readline().strip().split(",")
        line2 = f.readline().strip().split(",")
        max_len = max(len(line1), len(line2))
        line1 += [""] * (max_len - len(line1))
        line2 += [""] * (max_len - len(line2))
        combined_columns = [f"{eng.strip()} / {chi.strip()}" if chi.strip() else eng.strip() for eng, chi in zip(line1, line2)]

    df = pd.read_csv(csv_filename, skiprows=2, names=combined_columns, low_memory=False, encoding=safe_encoding)

    timestamp_col = combined_columns[1]

    df["Datetime"] = pd.to_datetime(df[timestamp_col], unit="s", utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)

    df.set_index("Datetime", inplace=True)


    return df, combined_columns
# ==== 初始化 Session State ====
if "df_all" not in st.session_state:
    st.session_state.df_all = None
    st.session_state.all_columns = None
    st.session_state.query_start_date = None
    st.session_state.query_start_time = None
    st.session_state.query_end_date = None
    st.session_state.query_end_time = None






st.set_page_config(
    page_title="台以乾式厭氧醱酵數據分析",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# ==== Tabs ==== 加上 query_params 控制目前在哪個 tab
query_params = st.experimental_get_query_params()
selected_tab = query_params.get("tab", ["首頁"])[0]

tab_names = ["首頁", "分析功能", "PIT/TT日對日比對"]
tab_idx = tab_names.index(selected_tab) if selected_tab in tab_names else 0

tabs = st.tabs(tab_names)

# 寫回目前 tab (保持同步，讓手動點 tab 也會寫入網址)
st.experimental_set_query_params(tab=tab_names[tab_idx])

# ==== 首頁 ====
with tabs[0]:
    st.title("台以乾式厭氧醱酵數據分析")
    st.markdown("""
    ### 使用說明
    歡迎使用本應用程式！  
    以下是基本操作指南：

    1. 點擊左上方小箭頭  
    2. 選擇需要分析的時間區間 
    3. 再到分析功能頁面調整圖表顯示參數  
    4. 查看結果並存檔 🎉

    **注意事項：**  
    - 查詢時間區間需大於20分鐘才看的到圖喔（因程式預設掐頭去尾各5分鐘）

    ---
    """)

with tabs[1]:
    st.title("分析功能")    

    st.sidebar.title("⚙️ 設定選項 - 資料查詢")
    
    start_date = st.sidebar.date_input("開始日期")
    start_time = st.sidebar.time_input("開始時間")
    end_date = st.sidebar.date_input("結束日期")
    end_time = st.sidebar.time_input("結束時間")

    if st.sidebar.button("查詢資料"):
        df, columns = fetch_csv_and_load_df(start_date, start_time, end_date, end_time)
        st.session_state.df_all = df
        st.session_state.all_columns = columns
        st.session_state.query_start_date = start_date
        st.session_state.query_start_time = start_time
        st.session_state.query_end_date = end_date
        st.session_state.query_end_time = end_time
    # 🚀 這一行 → 強制跳到分析功能 tab
        st.experimental_set_query_params(tab="分析功能")


    # 用 session_state 的資料
    df_all = st.session_state.get("df_all")
    all_columns = st.session_state.get("all_columns")
    query_start_date = st.session_state.get("query_start_date")
    query_start_time = st.session_state.get("query_start_time")
    query_end_date = st.session_state.get("query_end_date")
    query_end_time = st.session_state.get("query_end_time")



    

    if df_all is not None and all_columns is not None:
    # 進入畫圖段

        # === 分析功能頁專屬 sidebar 設定 ===
        st.sidebar.title("🖌️ 圖表設定")
        # → 接下來你的 sampling_interval_display 到畫圖的整段 code 全部放在這邊
            # ==== 取樣間隔 ====
        sampling_interval_display = st.sidebar.selectbox(
            "取樣間隔 (Resample)",
            ["5秒", "10秒", "30秒", "1分鐘", "5分鐘", "15分鐘"],
            index=4
        )
        sampling_interval_map = {
            "5秒": "5s",
            "10秒": "10s",
            "30秒": "30s",
            "1分鐘": "1min",
            "5分鐘": "5min",
            "15分鐘": "15min",
        }
        sampling_interval = sampling_interval_map[sampling_interval_display]
        # ==== X 軸主刻度 ====
        x_axis_interval = st.sidebar.selectbox(
            "X 軸主刻度間距",
            ["10分鐘","30分鐘", "1小時", "2小時", "3小時", "4小時", "6小時", "12小時", "1天", "7天"],
            index=2  # 0 是 "10分鐘"
        )
        interval_map = {
            "10分鐘": mdates.MinuteLocator(interval=10),
            "30分鐘": mdates.MinuteLocator(interval=30),
            "1小時": mdates.HourLocator(interval=1),
            "2小時": mdates.HourLocator(interval=2),
            "3小時": mdates.HourLocator(interval=3),
            "4小時": mdates.HourLocator(interval=4),
            "6小時": mdates.HourLocator(interval=6),
            "12小時": mdates.HourLocator(interval=12),
            "1天": mdates.DayLocator(interval=1),
            "7天": mdates.DayLocator(interval=7),
        }
        x_major_locator = interval_map.get(x_axis_interval, mdates.HourLocator(interval=1))

        # ==== Y 軸區間設定 ====
        pit_tt_y_axis_mode = st.sidebar.radio("PIT / TT Y 軸區間", ["Auto", "固定 0~1", "自訂 min/max"])
        y_min_custom = None
        y_max_custom = None
        if pit_tt_y_axis_mode == "自訂 min/max":
            y_min_custom = st.sidebar.number_input("自訂 Y 軸最小值", value=0.0)
            y_max_custom = st.sidebar.number_input("自訂 Y 軸最大值", value=1.0)

        # ==== 字體大小 ====
        font_size = st.sidebar.slider("字體大小", 10, 26, 18)
        line_w = st.sidebar.slider("線條粗細 (PIT/TT)", 1, 10, 2)

        # ==== PIT/TT 選擇 ====
        available_pit_tt_prefixes = sorted(list(set(
            [col.split(" / ")[0] for col in all_columns if col.startswith("pit-") or col.startswith("tt-")]
        )))
        default_pit_columns = ["pit-311a", "pit-311c", "pit-312a", "pit-312c"]

        selected_pit_tt_prefixes = st.sidebar.multiselect(
            "選擇 PIT / TT 欄位 (可複選)",
            available_pit_tt_prefixes,
            default=default_pit_columns
        )

        pit_cols = []
        for col_prefix in selected_pit_tt_prefixes:
            full_col = [col for col in all_columns if col.startswith(col_prefix)][0]
            pit_cols.append(full_col)

        # ==== 設備選擇 ====
        excluded_prefixes = ["id", "time", "date", "timestamp"]
        available_equipment_prefixes = sorted(list(set(
            [
                col.split(" / ")[0]
                for col in all_columns
                if not (col.startswith("pit-") or col.startswith("tt-"))
                and not any(col.lower().startswith(ex_prefix) for ex_prefix in excluded_prefixes)
            ]
        )))
        default_equipment_cols = ["av-303a", "av-303c", "p-303a", "p-304a", "b-311a"]

        selected_equipment_prefixes = st.sidebar.multiselect("選擇設備 (可複選)", available_equipment_prefixes, default=default_equipment_cols)

        equipment_cols_full = []
        for col_prefix in selected_equipment_prefixes:
            full_col = [col for col in all_columns if col.startswith(col_prefix)][0]
            equipment_cols_full.append(full_col)



    

        # ==== 轉換設備狀態欄位 ====
        def convert_running_state(val):
            val_str = str(val).strip().replace("'", "")
            val_str = re.sub(r"[^\d]", "", val_str)
            val_str = val_str.zfill(4)
            if len(val_str) != 4:
                return 0
            return 1 if (val_str[1] == "1" or val_str[3] == "1") else 0

        for col in equipment_cols_full:
            df_all[col + "_running"] = df_all[col].apply(convert_running_state)

        # 🚫 不要加 pytz / 不要加 tz_localize

        start_datetime = pd.to_datetime(f"{query_start_date} {query_start_time}")
        end_datetime = pd.to_datetime(f"{query_end_date} {query_end_time}")


        df_plot = df_all.loc[(df_all.index >= start_datetime) & (df_all.index <= end_datetime)]
        st.write(f"✅ 擷取時間段：{start_datetime} ～ {end_datetime}，總筆數：{len(df_plot)}")





        # 額外欄位：FIT-311-VOL 顯示
        fit_col_candidates = [col for col in all_columns if col.startswith("fit-311-vol")]
        fit_col = fit_col_candidates[0] if len(fit_col_candidates) > 0 else None
        show_fit = st.sidebar.checkbox("顯示 fit-311-vol / 沼氣總量", value=False, key="show_fit_checkbox")



        # ==== 繪圖 ====
        fig, ax1 = plt.subplots(figsize=(24, 14))

        resample_cols = pit_cols.copy()
        if show_fit and fit_col and fit_col in df_plot.columns:
            resample_cols.append(fit_col)

        if len(resample_cols) > 0:
            df_pit_resampled = df_plot[resample_cols].resample(sampling_interval).mean()
            df_pit_resampled = df_pit_resampled.asfreq(sampling_interval)

            trim_delta = pd.Timedelta(minutes=5)
            trim_start = start_datetime + trim_delta
            trim_end = end_datetime - trim_delta
            trim_mask = (df_pit_resampled.index >= trim_start) & (df_pit_resampled.index <= trim_end)
        else:
            df_pit_resampled = pd.DataFrame()  # 空 df
            trim_mask = pd.Series([False])  # 避免後面畫圖報錯


        # ==== 標題 & Y 標籤 ====
        sampling_interval_display_map = {
            "5s": "5 秒",
            "10s": "10 秒",
            "30s": "30 秒",
            "1min": "1 分鐘",
            "5min": "5 分鐘",
            "15min": "15 分鐘",
        }
        sampling_interval_display = sampling_interval_display_map.get(sampling_interval, sampling_interval)

        if all(col.startswith("pit-") for col in selected_pit_tt_prefixes):
            y_label = "PIT 趨勢圖 (kPa)"
            plot_title = f"PIT 趨勢圖 (取樣間隔：{sampling_interval_display})\n{start_datetime} ~ {end_datetime}"
        elif all(col.startswith("tt-") for col in selected_pit_tt_prefixes):
            y_label = "TT 趨勢圖 (°C)"
            plot_title = f"TT 趨勢圖 (取樣間隔：{sampling_interval_display})\n{start_datetime} ~ {end_datetime}"
        else:
            y_label = "PIT / TT 趨勢圖"
            plot_title = f"PIT / TT 趨勢圖 (取樣間隔：{sampling_interval_display})\n{start_datetime} ~ {end_datetime}"

        if len(equipment_cols_full) > 0:
            plot_title = plot_title.replace("趨勢圖", "趨勢及設備起停圖")

        # ==== PIT/TT 趨勢線圖 ====（含第 4 點 & 第 5 點）====
        default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

        # 第4點：線條是否顯示 checkbox
        show_line_map = {}
        for col in pit_cols:
            show_line = st.sidebar.checkbox(f"顯示 {col}", value=True)
            show_line_map[col] = show_line

        # 第5點：線條透明度
        line_alpha = st.sidebar.slider("線條透明度 (PIT/TT)", 0.0, 1.0, 1.0, step=0.05)


        # 線條顏色
        color_map_per_line = {}
        for i, col in enumerate(pit_cols):
            default_color = default_colors[i % len(default_colors)]
            selected_color = st.sidebar.color_picker(f"線條顏色 - {col}", default_color)
            color_map_per_line[col] = selected_color

        # ==== 畫線（含 PIT/TT & FIT）====
        for col in pit_cols:
            if show_line_map.get(col, True):
                ax1.plot(
                    df_pit_resampled.index[trim_mask],
                    df_pit_resampled[col][trim_mask],
                    label=col,
                    linewidth=line_w,
                    color=color_map_per_line[col],
                    alpha=line_alpha
                )

        # 額外畫 fit-311-vol
        if show_fit and fit_col in df_pit_resampled.columns:
            ax1.plot(
                df_pit_resampled.index[trim_mask],
                df_pit_resampled[fit_col][trim_mask],
                label="FIT-311-VOL / 沼氣總量",
                linewidth=line_w + 3,
                color="black",
                alpha=1.0,
                marker="o",
                markersize=10,
                markerfacecolor="black",
                markeredgecolor="black",
                markeredgewidth=1.5
            )


        # ==== X 軸設定 ==== (動態 DateFormatter)
        ax1.xaxis.set_major_locator(x_major_locator)

        if x_axis_interval in ["1天", "7天"]:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # 只顯示月-日
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # 顯示 月-日 時:分

        tick_length = font_size * 0.5
        tick_width = font_size * 0.05
    
        # X 軸
        ax1.tick_params(axis='x', labelsize=font_size + 4, length=tick_length, width=tick_width)
        plt.xticks(rotation=45)  # 不要再另外指定 fontsize，tick_params 已經設定好了


        # 準備 Y 軸範圍考慮的欄位
        y_axis_cols = pit_cols.copy()
        if show_fit and fit_col and fit_col in df_pit_resampled.columns:
            y_axis_cols.append(fit_col)

        # ==== Y 軸範圍 robust ====
        if pit_tt_y_axis_mode == "固定 0~1":
            ax1.set_ylim(0, 1)
        elif pit_tt_y_axis_mode == "自訂 min/max":
            ax1.set_ylim(y_min_custom, y_max_custom)
        else:
            df_valid_columns = df_pit_resampled[y_axis_cols].dropna(axis=1, how='all')
            if not df_valid_columns.empty:
                y_min = df_valid_columns.min().min()
                y_max = df_valid_columns.max().max()
                print(f"[DEBUG] y_min={y_min}, y_max={y_max}")

                if np.isfinite(y_min) and np.isfinite(y_max):
                    ax1.set_ylim(y_min * 0.95, y_max * 1.05)
                else:
                    print("[WARNING] y_min or y_max is not finite, skipping set_ylim()")
            else:
                print("[WARNING] All selected columns are empty after dropna, skipping set_ylim()")

        # ==== X 軸區間 ==== → 用完整 start_datetime ~ end_datetime
        ax1.set_xlim(start_datetime, end_datetime)


        # ==== 標題、X、Y 軸標籤 ====
        ax1.set_xlabel("時間", fontsize=font_size + 6, labelpad=10, fontweight="bold")
        ax1.set_ylabel(y_label, fontsize=font_size + 6, labelpad=10, fontweight="bold")
        ax1.set_title(plot_title, fontsize=font_size + 17, pad=70, fontweight="bold")

        # Y 軸
        ax1.tick_params(axis='y', labelsize=font_size + 4, length=tick_length, width=tick_width)

        # ==== 圖例 ====
        # 計算有幾條線
        num_lines = len(pit_cols)
        # 每行放 4 條
        ncol = min(num_lines, 4)
        # 算出有幾行 legend
        num_rows = int(np.ceil(num_lines / ncol))

        # 動態調整 legend y 位置 & 圖的 top
        # y_start 越大 → legend 靠上、圖區越大
        legend_y_start = 0.92 + 0.01 * (num_rows - 1)  # 每多一行多推一點上去
        top_adjust = 0.85 - 0.3 * (num_rows - 1)  # 主圖 top 往下收一點，避免擠到 legend

        # 加 legend
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y_start),
            ncol=ncol,
            fontsize=font_size + 4
        )
        # 調整主圖範圍，top 要動態
        plt.subplots_adjust(top=top_adjust)


        # ==== 設備啟停圖 ====
        if len(equipment_cols_full) > 0:
            ax2 = ax1.twinx()
            y_positions = np.arange(len(equipment_cols_full))

            for i, col in enumerate(equipment_cols_full):
                state_series = df_plot[col + "_running"].fillna(0).astype(int)
                change_idx = state_series.ne(state_series.shift()).cumsum()

                for grp_id, grp_df in state_series.groupby(change_idx):
                    grp_state = grp_df.iloc[0]
                    grp_start_time = grp_df.index[0]
                    grp_end_time = grp_df.index[-1] + pd.Timedelta(seconds=5)
                    grp_end_time = min(grp_end_time, end_datetime)
                    color = "green" if grp_state == 1 else "red"

                    alpha_running = 0.3  # 綠色 (設備啟動)
                    alpha_stopped = 0.3  # 紅色 (設備停止 → 淡一點)

                    color = "green" if grp_state == 1 else "red"
                    alpha_value = alpha_running if grp_state == 1 else alpha_stopped
                
                
                    ax2.axvspan(grp_start_time, grp_end_time,
                                    ymin=(i+0.1)/len(equipment_cols_full),
                                    ymax=(i+0.9)/len(equipment_cols_full),
                                    color=color, alpha=0.25)

            ax2.set_ylim(-0.5, len(equipment_cols_full)-0.5)
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels(equipment_cols_full, fontsize=font_size + 4)
            ax2.set_ylabel("設備運作狀態", fontsize=font_size + 6)

        # ==== 完成圖表繪製 ====
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)


with tabs[2]:
    st.title("📅 PIT/TT 日對日比對 (時間表示版)")

    if st.session_state.df_all is None or st.session_state.all_columns is None:
        st.warning("⚠️ 請先在【分析功能】頁查詢過一次資料，載入欄位定義。")
    else:
        # === Sidebar 設定 ===
        st.sidebar.title("⚙️ 日對日比對設定")

        # 選擇日期
        date_options = pd.date_range(end=pd.Timestamp.today(), periods=14).strftime("%Y-%m-%d").tolist()
        selected_dates = st.sidebar.multiselect(
            "選擇要比對的日期 (可多選)",
            options=date_options,
            default=[date_options[-1], date_options[-2]]
        )

        # 選擇 PIT/TT 欄位
        available_pit_tt_prefixes = sorted(list(set(
            [col.split(" / ")[0] for col in st.session_state.all_columns if col.startswith("pit-") or col.startswith("tt-")]
        )))
        pit_tt_selected = st.sidebar.selectbox("選擇 PIT / TT 欄位", available_pit_tt_prefixes)

        # Y軸區間
        y_axis_mode = st.sidebar.radio("Y 軸區間", ["Auto", "固定 0~1", "自訂 min/max"])
        y_min_custom = None
        y_max_custom = None
        if y_axis_mode == "自訂 min/max":
            y_min_custom = st.sidebar.number_input("自訂 Y 軸最小值", value=0.0)
            y_max_custom = st.sidebar.number_input("自訂 Y 軸最大值", value=1.0)

        # 取樣間隔 Resample
        sampling_interval_display = st.sidebar.selectbox(
            "取樣間隔 (Resample)",
            ["5秒", "10秒", "30秒", "1分鐘", "5分鐘", "15分鐘"],
            index=4
        )
        sampling_interval_map = {
            "5秒": "5s",
            "10秒": "10s",
            "30秒": "30s",
            "1分鐘": "1min",
            "5分鐘": "5min",
            "15分鐘": "15min",
        }
        sampling_interval = sampling_interval_map[sampling_interval_display]

        # 線條顏色 & 粗細 per 日期
        color_per_date = {}
        line_width_per_date = {}

        for date_str in selected_dates:
            default_color = "#1f77b4"
            color_picker = st.sidebar.color_picker(f"線條顏色 - {date_str}", default_color)
            color_per_date[date_str] = color_picker

            line_width = st.sidebar.slider(f"線條粗細 - {date_str}", 1, 10, 2)
            line_width_per_date[date_str] = line_width

        # 開始比對按鈕
        if st.button("🚀 開始比對") and len(selected_dates) > 0:
            import matplotlib.dates as mdates

            fig, ax = plt.subplots(figsize=(20, 10))

            for date_str in selected_dates:
                date_obj = pd.to_datetime(date_str).date()

                df_day, _ = fetch_csv_and_load_df(
                    start_date=date_obj,
                    start_time=pd.to_datetime("00:00").time(),
                    end_date=date_obj,
                    end_time=pd.to_datetime("23:59").time()
                )

                full_col = [col for col in st.session_state.all_columns if col.startswith(pit_tt_selected)][0]

                df_day_resampled = df_day[[full_col]].resample(sampling_interval).mean()
                df_day_resampled = df_day_resampled.asfreq(sampling_interval)
                df_day_resampled = df_day_resampled.dropna()

                # 將 index 轉成虛擬日期 + 時間 (for HH:MM X 軸)
                df_day_resampled["Time_dt"] = df_day_resampled.index.map(
                    lambda t: pd.Timestamp(year=2000, month=1, day=1, hour=t.hour, minute=t.minute, second=t.second)
                )

                df_day_resampled = df_day_resampled.sort_values("Time_dt")

                # 畫線
                ax.plot(
                    df_day_resampled["Time_dt"],
                    df_day_resampled[full_col],
                    label=f"{date_str}",
                    linewidth=line_width_per_date[date_str],
                    color=color_per_date[date_str]
                )

            # 繪圖設定
            ax.set_xlabel("時間 (HH:MM)", fontsize=16)
            ax.set_ylabel(full_col, fontsize=16)
            ax.set_title(f"同一天時間不同日期比對 - {pit_tt_selected} (取樣間隔：{sampling_interval_display})", fontsize=20)
            ax.legend(fontsize=14)
            ax.grid(True)

            # X 軸 formatter → HH:MM，locator 每1hr一格
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # 固定 X 軸範圍 00:00 ~ 23:59
            ax.set_xlim(pd.Timestamp("2000-01-01 00:00"), pd.Timestamp("2000-01-01 23:59"))

            # Y軸設定
            if y_axis_mode == "固定 0~1":
                ax.set_ylim(0, 1)
            elif y_axis_mode == "自訂 min/max":
                ax.set_ylim(y_min_custom, y_max_custom)
            else:
                pass

            st.pyplot(fig, use_container_width=True)

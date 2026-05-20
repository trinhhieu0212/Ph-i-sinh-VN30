import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ============================================================
# 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ============================================================
filename = 'data.csv'

if not os.path.exists(filename):
    raise FileNotFoundError(f"Không tìm thấy file {filename}!")

df = pd.read_csv(filename)
df.columns = [col.lower() for col in df.columns]

if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
elif 'date' in df.columns:
    df['time'] = pd.to_datetime(df['date'])

open_p = df['open'].values
close_p = df['close'].values
high_p  = df['high'].values
low_p   = df['low'].values
vol_p   = df['volume'].values
times   = df['time'].values

# ============================================================
# 2. TÍNH CHỈ BÁO RSI (14 GIAI ĐOẠN)
# ============================================================
rsi_period = 14
delta = df['close'].diff()

gain = (delta.where(delta > 0, 0)).copy()
loss = (-delta.where(delta < 0, 0)).copy()

avg_gain = gain.rolling(window=rsi_period).mean()
avg_loss = loss.rolling(window=rsi_period).mean()

for i in range(rsi_period, len(df)):
    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period

rs = avg_gain / (avg_loss + 1e-9)
rsi = 100 - (100 / (1 + rs))
rsi_values = rsi.values

# ============================================================
# 3. TÍNH ĐƯỜNG TRUNG BÌNH SMA 20 VÀ VOLUME TRUNG BÌNH 10 PHIÊN TRƯỚC
# ============================================================
sma_20 = df['close'].rolling(window=20).mean().values
vol_series = pd.Series(vol_p)
vol_sma_10_prev = vol_series.shift(1).rolling(window=10).mean().values

# ============================================================
# 4. QUÉT TÍN HIỆU THEO TRỤC RSI 50 + LỌC VOL ĐỘT BIẾN
# ============================================================
min_distance       = 15
last_added_idx     = -min_distance

confluence_indices = []
confluence_dirs    = []
signals_for_csv    = [] 

# Khởi tạo danh sách các nút bấm điều hướng góc trái
nav_buttons = []
nav_buttons.append(dict(
    label="--- RESET VIEW ---", 
    method="relayout", 
    args=[{
        "xaxis.range": [0, len(df)], 
        "yaxis.range": [min(low_p), max(high_p)]
    }]
))

for i in range(max(rsi_period, 20) + 1, len(df) - 1):
    if np.isnan(rsi_values[i]) or np.isnan(sma_20[i]) or np.isnan(vol_sma_10_prev[i]):
        continue

    curr_rsi = rsi_values[i]
    is_cross_up   = (close_p[i-1] <= sma_20[i-1] and close_p[i] > sma_20[i])
    is_cross_down = (close_p[i-1] >= sma_20[i-1] and close_p[i] < sma_20[i])
    is_vol_active = (vol_p[i] > vol_sma_10_prev[i])

    signal_data = None

    if is_cross_up and (curr_rsi >= 50.0) and is_vol_active:
        if i - last_added_idx >= min_distance:
            confluence_indices.append(i)
            confluence_dirs.append("UP")
            
            time_str = pd.Timestamp(times[i]).strftime('%H:%M %d/%m')
            signal_data = {
                'Nến thứ': i, 
                'Thời gian': pd.Timestamp(times[i]).strftime('%H:%M %d/%m/%Y'), 
                'Hướng lệnh': "LONG",
                'Giá Close': close_p[i], 
                'Volume': vol_p[i], 
                'RSI 14': round(curr_rsi, 1)
            }
            
            # Tạo hiệu ứng zoom thông minh (đồng bộ cả trục x của chart giá và chart rsi)
            view_start, view_end = max(0, i - 20), min(len(df), i + 40)
            nav_buttons.append(dict(
                label=f"Tín hiệu LONG | {time_str}", method="relayout",
                args=[{
                    "xaxis.range": [view_start, view_end],
                    "xaxis2.range": [view_start, view_end],
                    "yaxis.range": [min(low_p[view_start:view_end])*0.999, max(high_p[view_start:view_end])*1.001]
                }]
            ))
            last_added_idx = i

    elif is_cross_down and (curr_rsi <= 50.0) and is_vol_active:
        if i - last_added_idx >= min_distance:
            confluence_indices.append(i)
            confluence_dirs.append("DOWN")
            
            time_str = pd.Timestamp(times[i]).strftime('%H:%M %d/%m')
            signal_data = {
                'Nến thứ': i, 
                'Thời gian': pd.Timestamp(times[i]).strftime('%H:%M %d/%m/%Y'), 
                'Hướng lệnh': "SHORT",
                'Giá Close': close_p[i], 
                'Volume': vol_p[i], 
                'RSI 14': round(curr_rsi, 1)
            }
            
            # Tạo hiệu ứng zoom thông minh
            view_start, view_end = max(0, i - 20), min(len(df), i + 40)
            nav_buttons.append(dict(
                label=f"Tín hiệu SHORT | {time_str}", method="relayout",
                args=[{
                    "xaxis.range": [view_start, view_end],
                    "xaxis2.range": [view_start, view_end],
                    "yaxis.range": [min(low_p[view_start:view_end])*0.999, max(high_p[view_start:view_end])*1.001]
                }]
            ))
            last_added_idx = i

    if signal_data:
        signals_for_csv.append(signal_data)

# --- ĐOẠN CODE THỰC HIỆN XUẤT CSV TÊN 'KETQUA.CSV' ---
if signals_for_csv:
    df_export = pd.DataFrame(signals_for_csv)
    export_name = 'ketqua.csv'
    # Sử dụng utf-8-sig để khi mở bằng Excel không bị lỗi font Tiếng Việt
    df_export.to_csv(export_name, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu {len(signals_for_csv)} tín hiệu vào file: {export_name}")
else:
    print("❌ Không tìm thấy tín hiệu nào để xuất CSV.")

# ============================================================
# 5. THIẾT KẾ GIAO DIỆN CHIA CỘT (ĐỒ THỊ BÊN TRÁI - BẢNG BÊN PHẢI)
# ============================================================
fig = make_subplots(
    rows=2, cols=2,
    shared_xaxes=False,
    column_widths=[0.75, 0.25], 
    row_heights=[0.70, 0.30],
    vertical_spacing=0.05,
    horizontal_spacing=0.04,
    subplot_titles=('ĐỒ THỊ GIÁ & SMA 20', ' DANH SÁCH LỆNH', 'CHỈ BÁO RSI (14)'),
    specs=[[{"type": "xy"}, {"type": "domain", "rowspan": 2}], 
           [{"type": "xy"}, None]]
)

# --- CỘT 1 - HÀNG 1: ĐỒ THỊ GIÁ ---
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price', opacity=0.3, 
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1
)
fig.add_trace(go.Scattergl(x=df.index, y=sma_20, line=dict(color='#00E5FF', width=1.5), name='SMA 20'), row=1, col=1)

up_idx = [confluence_indices[k] for k in range(len(confluence_indices)) if confluence_dirs[k] == 'UP']
down_idx = [confluence_indices[k] for k in range(len(confluence_indices)) if confluence_dirs[k] == 'DOWN']

if up_idx:
    fig.add_trace(go.Scatter(x=up_idx, y=[low_p[i]*0.999 for i in up_idx], mode='markers',
                             marker=dict(symbol='triangle-up', size=13, color='#00FF88'), name='LONG Signal'), row=1, col=1)
if down_idx:
    fig.add_trace(go.Scatter(x=down_idx, y=[high_p[i]*1.001 for i in down_idx], mode='markers',
                             marker=dict(symbol='triangle-down', size=13, color='#FF4466'), name='SHORT Signal'), row=1, col=1)

# --- CỘT 1 - HÀNG 2: ĐỒ THỊ RSI ---
fig.add_trace(go.Scattergl(x=list(df.index), y=rsi_values.tolist(), line=dict(color='#E0E0E0', width=1.8), name='RSI'), row=2, col=1)
fig.add_hrect(y0=50, y1=100, fillcolor="rgba(0, 255, 136, 0.02)", line_width=0, row=2, col=1)
fig.add_hrect(y0=0, y1=50, fillcolor="rgba(255, 68, 102, 0.02)", line_width=0, row=2, col=1)
fig.add_hline(y=50, line=dict(color='#FFCC00', dash='solid', width=1.2), row=2, col=1)

# --- CỘT 2 - HÀNG 1 & 2 (ROWSPAN): BẢNG TÍN HIỆU ---
if signals_for_csv:
    def make_table_dict(filter_dir=None):
        filtered = [s for s in signals_for_csv if filter_dir is None or s['Hướng lệnh'] == filter_dir]
        
        colors = []
        for s in filtered:
            colors.append('rgba(0, 255, 136, 0.2)' if s['Hướng lệnh'] == "LONG" else 'rgba(255, 68, 102, 0.2)')

        return dict(
            header=dict(
                values=['<b>Thời Gian</b>', '<b>Lệnh</b>', '<b>Giá</b>', '<b>RSI</b>'],
                fill_color='#1f242d', align='center', font=dict(color='white', size=11)
            ),
            cells=dict(
                values=[
                    [s['Thời gian'] for s in filtered],
                    [f"<b>{s['Hướng lệnh']}</b>" for s in filtered],
                    [s['Giá Close'] for s in filtered],
                    [s['RSI 14'] for s in filtered]
                ],
                fill_color=[
                    ['#2a2e39' if i % 2 == 0 else '#232731' for i in range(len(filtered))],
                    colors,
                    ['#2a2e39' if i % 2 == 0 else '#232731' for i in range(len(filtered))],
                    ['#2a2e39' if i % 2 == 0 else '#232731' for i in range(len(filtered))]
                ],
                align='center', font=dict(color='#E0E0E0', size=11), height=22
            )
        )

    table_trace = go.Table(make_table_dict(None))
    fig.add_trace(table_trace, row=1, col=2)

    # ============================================================
    # KẾT HỢP CẢ 2 HỆ THỐNG MENU (NAVIGATE GÓC TRÁI + FILTER GÓC PHẢI)
    # ============================================================
    menus = [
        dict(
            buttons=nav_buttons, 
            direction="down", 
            showactive=True, 
            x=-0.08, 
            xanchor="left", 
            y=1.08, 
            bgcolor="#1e1e1e", 
            font=dict(color="white", size=10)
        ),
        dict(
            type="dropdown",
            direction="down",
            active=0,
            x=0.82,  
            y=1.08,
            showactive=True,
            bgcolor="#1e1e1e",
            font=dict(color="white", size=10),
            buttons=[
                dict(label="Lọc bảng: TẤT CẢ", method="restyle", args=[make_table_dict(None), [5]]),
                dict(label="Lọc bảng: LONG 🟢", method="restyle", args=[make_table_dict("LONG"), [5]]),
                dict(label="Lọc bảng: SHORT 🔴", method="restyle", args=[make_table_dict("SHORT"), [5]])
            ]
        )
    ]
    fig.update_layout(updatemenus=menus)

# ============================================================
# CẤU HÌNH LAYOUT VÀ HIỂN THỊ
# ============================================================
fig.update_layout(
    title=f'<b>RSI 50 + SMA 20 STRATEGY DASHBOARD</b> - Đã xuất {len(signals_for_csv)} tín hiệu ra CSV',
    template='plotly_dark',
    height=850, 
    margin=dict(l=120, r=30, t=100, b=40), 
    xaxis_rangeslider_visible=False,
    yaxis=dict(side='right'),
    yaxis2=dict(side='right', range=[15, 85]),
    legend=dict(orientation='h', y=-0.02, x=0.01)
)

fig.update_layout(xaxis1_matches='x2')
fig.show(config={'scrollZoom': True})
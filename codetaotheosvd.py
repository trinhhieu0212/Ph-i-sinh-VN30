import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

close_p, high_p, low_p, vol_p = df['close'].values, df['high'].values, df['low'].values, df['volume'].values
times = df['time'].values

# --- SVD CẢI TIẾN ---
window_size = 40
svd_line = np.full(len(df), np.nan)

for i in range(window_size, len(df)):
    window_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[i-window_size+1 : i+1].values
    mean = np.mean(window_data, axis=0)
    std = np.std(window_data, axis=0) + 1e-9
    scaled_win = (window_data - mean) / std
    U, S, Vt = np.linalg.svd(scaled_win, full_matrices=False)
    rank = 2
    reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    val = reconstructed[-1, 3] * std[3] + mean[3]
    svd_line[i] = val

ema_20 = df['close'].ewm(span=20, adjust=False).mean().values
tr = np.maximum(high_p[1:]-low_p[1:], np.maximum(abs(high_p[1:]-close_p[:-1]), abs(low_p[1:]-close_p[:-1])))
atr = np.zeros_like(close_p)
atr[14:] = pd.Series(tr).rolling(14).mean().values[13:]
vol_sma = pd.Series(vol_p).rolling(20).mean().values

# ============================================================
# 2. RÀ SOÁT CÁC ĐIỂM HỢP LƯU & LƯU TÍN HIỆU
# ============================================================
SQUEEZE_THRESHOLD = 0.15 
min_distance = 15
last_added_idx = -min_distance

confluence_indices = []
signals_for_csv = [] 
buttons = []
buttons.append(dict(label="--- RESET VIEW ---", method="relayout", 
                    args=[{"xaxis.range": [0, len(df)], "yaxis.range": [min(low_p), max(high_p)]}]))

for i in range(window_size + 5, len(df)):
    if np.isnan(svd_line[i]): continue
    
    line_dist = abs(svd_line[i] - ema_20[i])
    center_point = (svd_line[i] + ema_20[i]) / 2
    dist_candle = max(0, low_p[i] - center_point, center_point - high_p[i])
    
    is_squeezed = (line_dist < (atr[i] * SQUEEZE_THRESHOLD)) and (dist_candle < (atr[i] * SQUEEZE_THRESHOLD))
    is_vol_active = vol_p[i] > vol_sma[i] * 0.8
    slope = svd_line[i] - svd_line[i-3]
    has_trend = abs(slope) > (atr[i] * 0.05)

    if is_squeezed and is_vol_active and has_trend:
        if i - last_added_idx >= min_distance:
            confluence_indices.append(i)
            
            time_str = pd.Timestamp(times[i]).strftime('%H:%M %d/%m/%Y')
            
            # Lưu CSV: Bỏ cột Hướng
            signals_for_csv.append({
                'Nến thứ': i,
                'Thời gian': time_str,
                'Giá Close': close_p[i],
                'SVD': round(svd_line[i], 2),
                'EMA20': round(ema_20[i], 2),
                'Volume': vol_p[i],
                'ATR': round(atr[i], 2)
            })
            
            # Menu: Bỏ dán nhãn LONG/SHORT
            view_start, view_end = max(0, i - 20), min(len(df), i + 40)
            buttons.append(dict(label=f"Hợp lưu | {time_str}", method="relayout",
                args=[{"xaxis.range": [view_start, view_end],
                       "yaxis.range": [min(low_p[view_start:view_end])*0.999, max(high_p[view_start:view_end])*1.001]}]))
            last_added_idx = i

# --- THỰC HIỆN XUẤT CSV ---
if signals_for_csv:
    df_export = pd.DataFrame(signals_for_csv)
    export_name = 'ket_qua_tin_hieu.csv'
    df_export.to_csv(export_name, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu {len(signals_for_csv)} tín hiệu vào file: {export_name}")
else:
    print("❌ Không tìm thấy tín hiệu nào để xuất CSV.")

# ============================================================
# 3. HIỂN THỊ BIỂU ĐỒ
# ============================================================
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price', opacity=0.3))
fig.add_trace(go.Scattergl(x=df.index, y=svd_line, line=dict(color='#0066FF', width=2), name='SVD (Rolling)'))
fig.add_trace(go.Scattergl(x=df.index, y=ema_20, line=dict(color='#FFCC00', width=1, dash='dot'), name='EMA 20'))

fig.add_trace(go.Scatter(
    x=confluence_indices, y=[close_p[idx] for idx in confluence_indices],
    mode='markers+text', text=["  Hợp lưu" for _ in confluence_indices], textposition="top center",
    marker=dict(symbol='star', size=12, color='#FF00FF', line=dict(width=1, color='white')),
    name='Điểm hợp lưu hội tụ'
))

fig.update_layout(
    updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=-0.25, xanchor="left", y=1.1, bgcolor="#1e1e1e", font=dict(color="white", size=10))],
    title=f'<b>SVD & EMA CONFLUENCE SCANNER</b> - Đã xuất {len(signals_for_csv)} tín hiệu ra CSV',
    template='plotly_dark', height=850, margin=dict(l=280),
    xaxis=dict(rangeslider=dict(visible=False)), yaxis=dict(side='right')
)

fig.show(config={'scrollZoom': True})
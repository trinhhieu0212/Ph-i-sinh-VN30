"""
╔══════════════════════════════════════════════════════════════════╗
║       SVD CONFLUENCE — ML MODEL v3 (PRO)                         ║
║   Input : ketqua.csv  (điểm hợp lưu từ scanner)                  ║
║           data.csv    (OHLCV gốc)                                ║
║   Output: predictions_final.csv (Bảng dự báo + Thống kê Win/Loss)║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    print("⚠️  XGBoost chưa cài → dùng GradientBoosting. pip install xgboost")
    USE_XGB = False

# ════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ════════════════════════════════════════════════════════════════
SL_POINTS       = 7        # Stop Loss cố định (điểm)
TP_POINTS       = 14       # Take Profit = 2x SL
MAX_CANDLES_FWD = 100      # Số nến tối đa chờ TP/SL
TRAIN_RATIO     = 0.80     # 80% train / 20% predict

print("=" * 60)
print("  SVD CONFLUENCE — ML MODEL v3")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# 1. ĐỌC FILE
# ════════════════════════════════════════════════════════════════
for fname in ['ketqua.csv', 'data.csv']:
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Không tìm thấy {fname}!")

# --- ketqua.csv ---
kq = pd.read_csv('ketqua.csv', encoding='utf-8-sig')
kq.columns = [c.strip() for c in kq.columns]

col_map = {}
for c in kq.columns:
    cl = c.lower()
    if any(x in cl for x in ['nến', 'nen', 'index', 'idx', 'candle']):
        col_map[c] = 'candle_idx'
    elif any(x in cl for x in ['thời', 'time', 'date']):
        col_map[c] = 'time'
    elif any(x in cl for x in ['close', 'giá']):
        col_map[c] = 'close'
    elif 'svd' in cl:
        col_map[c] = 'svd'
    elif 'ema' in cl:
        col_map[c] = 'ema20'
    elif any(x in cl for x in ['volume', 'vol']):
        col_map[c] = 'volume'
    elif 'atr' in cl:
        col_map[c] = 'atr'
kq = kq.rename(columns=col_map)

# Kiểm tra cột bắt buộc
for col in ['candle_idx', 'close', 'atr']:
    if col not in kq.columns:
        raise ValueError(f"Thiếu cột '{col}' trong ketqua.csv!")

kq['candle_idx'] = kq['candle_idx'].astype(int)
kq = kq.sort_values('candle_idx').reset_index(drop=True)
print(f"✅ ketqua.csv : {len(kq)} điểm hợp lưu")

# --- data.csv ---
df = pd.read_csv('data.csv')
df.columns = [c.lower() for c in df.columns]
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
elif 'date' in df.columns:
    df['time'] = pd.to_datetime(df['date'])
df = df.sort_values('time').reset_index(drop=True)

high_all  = df['high'].values
low_all   = df['low'].values
close_all = df['close'].values
open_all  = df['open'].values
print(f"✅ data.csv   : {len(df)} nến")

# ════════════════════════════════════════════════════════════════
# 2. CHIA 80 / 20
# ════════════════════════════════════════════════════════════════
split     = max(5, int(len(kq) * TRAIN_RATIO))
kq_train  = kq.iloc[:split].reset_index(drop=True)
kq_pred   = kq.iloc[split:].reset_index(drop=True)

print(f"\n📊 Train : {len(kq_train)} điểm (80% đầu)")
print(f"   Predict: {len(kq_pred)} điểm (20% cuối)")

# ════════════════════════════════════════════════════════════════
# 3. GÁN NHÃN CHO 80% TRAIN
# ════════════════════════════════════════════════════════════════
def assign_label(candle_idx):
    if candle_idx >= len(close_all): return -1
    entry  = close_all[candle_idx]
    tp, sl = entry + TP_POINTS, entry - SL_POINTS
    end    = min(candle_idx + MAX_CANDLES_FWD + 1, len(close_all))
    for j in range(candle_idx + 1, end):
        if high_all[j] >= tp: return 1
        if low_all[j] <= sl: return 0
    return -1

labels = [assign_label(int(r['candle_idx'])) for _, r in kq_train.iterrows()]
kq_train['label'] = labels
kq_train_valid = kq_train[kq_train['label'] != -1].reset_index(drop=True)

# ════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
def build_features(row_df, i, source_df):
    row = source_df.iloc[i]
    idx = int(row['candle_idx'])
    c, a, svd, ema, vol = row['close'], max(row.get('atr', 1.0), 1e-9), row.get('svd', row['close']), row.get('ema20', row['close']), row.get('volume', 1)
    o = open_all[idx] if idx < len(open_all) else c
    h = high_all[idx] if idx < len(high_all) else c
    l = low_all[idx]  if idx < len(low_all)  else c
    
    return {
        'svd_vs_close': (svd - c) / a, 'ema_vs_close': (ema - c) / a,
        'body': (c - o) / a, 'candle_range': (h - l) / a,
        'vol_norm': np.log1p(vol) / 10.0, 'atr_rel': a / c if c > 0 else 0.01
    }

X_train = pd.DataFrame([build_features(None, i, kq_train_valid) for i in range(len(kq_train_valid))]).values
y_train = kq_train_valid['label'].values
X_pred = pd.DataFrame([build_features(None, i, kq_pred) for i in range(len(kq_pred))]).values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_pred_s = scaler.transform(X_pred)

# ════════════════════════════════════════════════════════════════
# 5. TRAIN MODEL
# ════════════════════════════════════════════════════════════════
rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42).fit(X_train_s, y_train)

if USE_XGB:
    boost = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, eval_metric='logloss', verbosity=0).fit(X_train_s, y_train)
else:
    boost = GradientBoostingClassifier(n_estimators=300, max_depth=4).fit(X_train_s, y_train)

# Ensemble
ensemble_proba = (rf.predict_proba(X_pred_s)[:, 1] + boost.predict_proba(X_pred_s)[:, 1]) / 2

# ════════════════════════════════════════════════════════════════
# 6. PREDICT 20% CUỐI & BACKTEST THỰC TẾ (WIN/LOSS)
# ════════════════════════════════════════════════════════════════
results = []
total_pnl = 0
wins = 0
finished = 0

for k in range(len(kq_pred)):
    row = kq_pred.iloc[k]
    prob = ensemble_proba[k]
    sig = "LONG" if prob >= 0.5 else "SHORT"
    conf = prob if sig == "LONG" else 1 - prob
    c, c_idx = row['close'], int(row['candle_idx'])
    
    sl = round(c - SL_POINTS, 2) if sig == "LONG" else round(c + SL_POINTS, 2)
    tp = round(c + TP_POINTS, 2) if sig == "LONG" else round(c - TP_POINTS, 2)

    # Kiểm tra thực tế trong data.csv
    outcome, pnl = "OPEN", 0
    end_scan = min(c_idx + MAX_CANDLES_FWD + 1, len(close_all))
    for j in range(c_idx + 1, end_scan):
        if sig == "LONG":
            if high_all[j] >= tp: outcome, pnl = "WIN", TP_POINTS; break
            if low_all[j] <= sl: outcome, pnl = "LOSS", -SL_POINTS; break
        else:
            if low_all[j] <= tp: outcome, pnl = "WIN", TP_POINTS; break
            if high_all[j] >= sl: outcome, pnl = "LOSS", -SL_POINTS; break
    
    if outcome != "OPEN":
        finished += 1
        if outcome == "WIN": wins += 1
        total_pnl += pnl

    results.append({
        'Thời gian': row.get('time', f"Nến {c_idx}"), 'Entry': round(c, 2),
        'Tín hiệu': sig, 'Conf (%)': round(conf * 100, 1),
        'SL': sl, 'TP': tp, 'Kết quả': outcome, 'PnL': pnl
    })

df_res = pd.DataFrame(results)
df_res.to_csv('predictions_final.csv', index=False, encoding='utf-8-sig')

# ════════════════════════════════════════════════════════════════
# 7. IN KẾT QUẢ & THỐNG KÊ (PHẦN BẠN CẦN)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print(f"{'#':<4} {'Thời gian':<20} {'Entry':>8} {'Tín hiệu':<8} {'Conf':>6} {'Kết quả':<8} {'PnL':>6} {'SL':>8} {'TP':>8}")
print("-" * 95)

for i, r in df_res.iterrows():
    icon = "🟢" if r['Kết quả'] == "WIN" else ("🔴" if r['Kết quả'] == "LOSS" else "⚪")
    print(f"{i+1:<4} {str(r['Thời gian'])[:20]:<20} {r['Entry']:>8.2f} {r['Tín hiệu']:<8} {r['Conf (%)']:>5.1f}% {icon} {r['Kết quả']:<6} {r['PnL']:>6.1f} {r['SL']:>8.2f} {r['TP']:>8.2f}")

wr = (wins / finished * 100) if finished > 0 else 0
print("=" * 95)
print(f"📊 THỐNG KÊ CHIẾN THUẬT (Tập 20%):")
print(f"   ➤ Tổng lệnh: {len(df_res)} | Lệnh đã đóng: {finished}")
print(f"   ➤ Tỉ lệ thắng (Win Rate): {wr:.2f}%")
print(f"   ➤ Tổng điểm Net Profit: {total_pnl:.1f} điểm")
print(f"✅ Đã lưu báo cáo: predictions_final.csv")
print("=" * 95)
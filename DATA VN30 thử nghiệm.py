import pandas as pd
import numpy as np
from vnstock3 import Vnstock
from sklearn.preprocessing import StandardScaler

# Khởi tạo
stock = Vnstock()

def get_expiry_dates(df):
    """Tính toán ngày đáo hạn phái sinh (Thứ 5 tuần thứ 3 của tháng)"""
    dates = pd.Series(df['time'].dt.date.unique())
    df_dates = pd.DataFrame({'date': dates})
    df_dates['year'] = pd.to_datetime(df_dates['date']).dt.year
    df_dates['month'] = pd.to_datetime(df_dates['date']).dt.month
    df_dates['dayofweek'] = pd.to_datetime(df_dates['date']).dt.dayofweek
    
    # Lọc các ngày Thứ 5
    thursdays = df_dates[df_dates['dayofweek'] == 3]
    # Lấy nến Thứ 5 xuất hiện lần thứ 3 trong tháng
    expiry_dates = thursdays.groupby(['year', 'month']).nth(2)['date'].tolist()
    return expiry_dates

try:
    print("🚀 Đang lấy dữ liệu VN30F1M...")
    df = stock.stock(symbol='VN30F1M', source='VCI').quote.history(
        start='2021-05-10', end='2026-05-10', interval='5m'
    )

    if df is not None and not df.empty:
        df['time'] = pd.to_datetime(df['time'])
        
        # 1. Lọc cơ bản & Loại bỏ trùng lặp (nếu có)
        df = df[df['close'] > 0].drop_duplicates(subset=['time']).copy()
        
        # 2. Lọc khung giờ giao dịch (8:45-11:30 và 13:00-14:45)
        def is_trading_time(t):
            return ((t >= pd.Timestamp("08:45:00").time()) & (t <= pd.Timestamp("11:30:00").time())) | \
                   ((t >= pd.Timestamp("13:00:00").time()) & (t <= pd.Timestamp("14:45:00").time()))
        
        df = df[df['time'].dt.time.apply(is_trading_time)].copy()

        # 3. Lọc Thứ 7, Chủ Nhật (DayOfWeek: 5, 6)
        df = df[df['time'].dt.dayofweek < 5].copy()

        # 4. Lọc Ngày đáo hạn phái sinh (Vùng nhiễu cực mạnh)
        expiry_dates = get_expiry_dates(df)
        df = df[~df['time'].dt.date.isin(expiry_dates)].copy()

        # 5. Sắp xếp và Xử lý Gaps (Lấp đầy nến thiếu)
        df = df.sort_values('time').ffill()

        # 6. Lọc Outlier (Chặn nến nhảy vọt > 1%)
        pct_change = df['close'].pct_change().abs()
        df = df[pct_change < 0.01].copy()

        # 7. Tính Log Returns & Chuẩn hóa (Z-Score)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()
        
        scaler = StandardScaler()
        df['scaled_log_ret'] = scaler.fit_transform(df[['log_ret']])
        
        # 8. Xuất kết quả
        print("-" * 30)
        print(f"✅ Xử lý XONG (Đã bỏ qua lọc Holiday)")
        print(f"📊 Số dòng còn lại: {len(df)}")
        print(f"📅 Số ngày đáo hạn đã loại: {len(expiry_dates)}")
        
        df.to_csv('data_vn30f1m_no_holiday.csv', index=False)
        print("💾 Đã lưu file: data_vn30f1m_no_holiday.csv")
        
    else:
        print("❌ Không có dữ liệu.")

except Exception as e:
    print(f"❌ Lỗi rồi: {e}")
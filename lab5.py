import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')

def bai1_supermarket():
    print("--- BÀI 1: DOANH THU SIÊU THỊ ---")
    df = pd.read_csv('ITA105_Lab_5_Supermarket.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    df['revenue'] = df['revenue'].interpolate(method='linear')
    
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    plt.figure(figsize=(12, 5))
    df.resample('ME')['revenue'].sum().plot(title='Tong doanh thu theo thang', marker='o')
    plt.savefig('b1_revenue_monthly.png')
    plt.close()
    
    plt.figure(figsize=(12, 5))
    df.resample('W')['revenue'].sum().plot(title='Tong doanh thu theo tuan', marker='o')
    plt.savefig('b1_revenue_weekly.png')
    plt.close()
    
    plt.figure(figsize=(12, 5))
    df['revenue'].plot(label='Daily Revenue', alpha=0.5)
    df['revenue'].rolling(window=30).mean().plot(label='30-Day Rolling Mean', color='red')
    plt.legend()
    plt.title('Trend doanh thu')
    plt.savefig('b1_revenue_trend.png')
    plt.close()
    
    decomposition = seasonal_decompose(df['revenue'], model='additive', period=30)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.savefig('b1_revenue_decomposition.png')
    plt.close()
    print("Hoan thanh Bai 1.")

def bai2_webtraffic():
    print("\n--- BÀI 2: LƯU LƯỢNG TRUY CẬP WEBSITE ---")
    df = pd.read_csv('ITA105_Lab_5_Web_traffic.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    df = df.resample('h').mean()
    df['visits'] = df['visits'].interpolate(method='linear')
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    plt.figure(figsize=(10, 5))
    df.groupby('hour')['visits'].mean().plot(kind='bar', color='skyblue')
    plt.title('Luu luong truy cap trung binh theo gio')
    plt.savefig('b2_traffic_hourly.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    df.groupby('day_of_week')['visits'].mean().plot(kind='bar', color='lightgreen')
    plt.title('Luu luong truy cap trung binh theo ngay trong tuan')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    plt.savefig('b2_traffic_daily.png')
    plt.close()
    print("Hoan thanh Bai 2.")

def bai3_stock():
    print("\n--- BÀI 3: GIÁ CỔ PHIẾU ---")
    df = pd.read_csv('ITA105_Lab_5_Stock.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    df = df.asfreq('B')
    df['close_price'] = df['close_price'].ffill()
    
    plt.figure(figsize=(12, 5))
    df['close_price'].plot(label='Close Price', alpha=0.5)
    df['close_price'].rolling(window=7).mean().plot(label='7-Day Rolling Mean', color='orange')
    df['close_price'].rolling(window=30).mean().plot(label='30-Day Rolling Mean', color='red')
    plt.title('Gia co phieu va Trend')
    plt.legend()
    plt.savefig('b3_stock_trend.png')
    plt.close()
    
    df['month'] = df.index.month
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='month', y='close_price', data=df)
    plt.title('Tinh mua vu cua gia co phieu theo thang')
    plt.savefig('b3_stock_seasonality.png')
    plt.close()
    print("Hoan thanh Bai 3.")

def bai4_production():
    print("\n--- BÀI 4: SẢN XUẤT CÔNG NGHIỆP ---")
    df = pd.read_csv('ITA105_Lab_5_Production.csv')
    df['week_start'] = pd.to_datetime(df['week_start'])
    df.set_index('week_start', inplace=True)
    
    df['production'] = df['production'].interpolate(method='linear')
    
    df['week'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    plt.figure(figsize=(12, 5))
    df['production'].plot(label='Weekly Production', alpha=0.5)
    df['production'].rolling(window=52).mean().plot(label='52-Week Rolling Mean (Trend)', color='red')
    plt.title('San xuat cong nghiep va Trend')
    plt.legend()
    plt.savefig('b4_production_trend.png')
    plt.close()
    
    plt.figure(figsize=(8, 5))
    df.groupby('quarter')['production'].mean().plot(kind='bar', color='coral')
    plt.title('San xuat trung binh theo quy')
    plt.savefig('b4_production_quarterly.png')
    plt.close()
    
    decomposition = seasonal_decompose(df['production'], model='additive', period=52)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.savefig('b4_production_decomposition.png')
    plt.close()
    print("Hoan thanh Bai 4.")

def main():
    bai1_supermarket()
    bai2_webtraffic()
    bai3_stock()
    bai4_production()
    print("\nHoan tat toan bo Lab 5! Da luu cac bieu do thanh file anh.")

if __name__ == "__main__":
    main()
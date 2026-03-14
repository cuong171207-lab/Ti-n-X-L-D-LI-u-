import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ITA105_Lab_1.csv')

print("=== BÀI 1: KHÁM PHÁ DỮ LIỆU ===")
print(f"Kích thước dữ liệu: {df.shape}")
print("\nThống kê mô tả (cột số):\n", df.describe())
print("\nGiá trị thiếu trong các cột:\n", df.isnull().sum())

print("\n=== BÀI 2: XỬ LÝ DỮ LIỆU THIẾU ===")
df_dropna = df.dropna()
print(f"Kích thước sau khi dùng dropna(): {df_dropna.shape}")


df['Price'] = df['Price'].fillna(df['Price'].median())
df['StockQuantity'] = df['StockQuantity'].fillna(df['StockQuantity'].median())
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])

print("\nSố lượng giá trị thiếu sau khi điền:\n", df.isnull().sum())

print("\n=== BÀI 3: XỬ LÝ DỮ LIỆU LỖI ===")
print(f"Số lượng Price < 0: {(df['Price'] < 0).sum()}")
print(f"Số lượng StockQuantity < 0: {(df['StockQuantity'] < 0).sum()}")

df['Price'] = df['Price'].abs()
df['StockQuantity'] = df['StockQuantity'].abs()

df['Rating'] = df['Rating'].clip(lower=1, upper=5)
print("Đã xử lý Rating về khoảng 1-5.")

print("\n=== BÀI 4: LÀM MƯỢT DỮ LIỆU NHIỄU ===")
df['Price_MA'] = df['Price'].rolling(window=5, min_periods=1).mean()


plt.figure(figsize=(12, 6))
plt.plot(df.index[:50], df['Price'][:50], label='Giá gốc (Original Price)', alpha=0.5, marker='o')
plt.plot(df.index[:50], df['Price_MA'][:50], label='Moving Average (window=5)', color='red', linewidth=2)
plt.title('So sánh Cột Price và Moving Average (50 mẫu đầu tiên)')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n=== BÀI 5: CHUẨN HÓA DỮ LIỆU ===")
df['Category'] = df['Category'].astype(str).str.lower()

df['Description'] = df['Description'].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

TIGIA_USD_VND = 25000
df['Price_VND'] = df['Price'] * TIGIA_USD_VND

print("Đã hoàn tất chuẩn hoá. Lưu ra file mới.")

df.to_csv('ITA105_Lab_1_Cleaned.csv', index=False)
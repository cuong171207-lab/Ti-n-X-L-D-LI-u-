import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_data(filename):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, filename)
    if not os.path.exists(file_path):
        print(f"\n[LỖI] Không tìm thấy file: '{filename}'")
        print(f"Vui lòng copy file vào thư mục: {base_path}")
        return None
    return pd.read_csv(file_path)

def bai_1_housing():
    print("\n--- BÀI 1: HOUSING DATA ---")
    df = load_data("ITA105_Lab_2_Housing.csv")
    if df is None: return
    print(f"Shape: {df.shape}")
    print("Missing values:\n", df.isnull().sum())

    print("Mô tả dữ liệu:\n", df.describe())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure()
    df[numeric_cols].boxplot()
    plt.title("Boxplot các biến Numeric - Housing")
    plt.show()

    plt.figure()
    sns.scatterplot(data=df, x="dien_tich", y="gia")
    plt.title("Scatterplot: Diện tích vs Giá")
    plt.show()

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = (
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).sum()

    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outliers_z = (z_scores > 3).sum()

    print(f"Số lượng ngoại lệ theo IQR:\n{outliers_iqr}")
    print(f"Số lượng ngoại lệ theo Z-score (|Z|>3):\n{outliers_z}")

    df["gia_log"] = np.log1p(df["gia"])
    plt.figure()
    sns.boxplot(x=df["gia_log"])
    plt.title("Boxplot Giá sau khi Log-transform")
    plt.show()


def bai_2_iot():
    print("\n--- BÀI 2: IOT DATA ---")
    df = load_data("ITA105_Lab_2_Iot.csv")
    if df is None: return
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    s1 = df[df["sensor_id"] == "S1"]
    plt.figure()
    s1["temperature"].plot()
    plt.title("Temperature over time (Sensor S1)")
    plt.show()

    rm = s1["temperature"].rolling(window=10).mean()
    rs = s1["temperature"].rolling(window=10).std()
    upper = rm + 3 * rs
    lower = rm - 3 * rs
    outliers_rolling = s1[(s1["temperature"] > upper) | (s1["temperature"] < lower)]
    print(f"Số ngoại lệ phát hiện bằng Rolling Mean (S1): {len(outliers_rolling)}")

    plt.figure()
    sns.scatterplot(data=df, x="temperature", y="pressure", alpha=0.5)
    plt.title("Temperature vs Pressure")
    plt.show()

    df["temp_clipped"] = df["temperature"].clip(
        lower=df["temperature"].quantile(0.01), upper=df["temperature"].quantile(0.99)
    )
    print("Đã xử lý ngoại lệ bằng phương pháp Clip.")


def bai_3_ecommerce():
    print("\n--- BÀI 3: E-COMMERCE DATA ---")
    df = load_data("ITA105_Lab_2_Ecommerce.csv")
    if df is None: return

    plt.figure()
    df[["price", "quantity", "rating"]].boxplot()
    plt.title("Boxplot E-commerce")
    plt.show()

    invalid_rating = df[df["rating"] > 5]
    zero_price = df[df["price"] <= 0]
    print(f"Số dòng Rating lỗi (>5): {len(invalid_rating)}")
    print(f"Số dòng Giá lỗi (<=0): {len(zero_price)}")

    df_clean = df[(df["rating"] <= 5) & (df["price"] > 0)].copy()
    df_clean["price_log"] = np.log1p(df_clean["price"])

    plt.figure()
    sns.scatterplot(data=df_clean, x="price_log", y="quantity", hue="category")
    plt.title("Price (Log) vs Quantity sau khi làm sạch")
    plt.show()


def bai_4_multivariate():
    print("\n--- BÀI 4: MULTIVARIATE OUTLIER ---")
    df = load_data("ITA105_Lab_2_Housing.csv")
    if df is None: return

    z = np.abs(stats.zscore(df[["dien_tich", "gia"]]))
    is_outlier = (z > 3).any(axis=1)

    plt.figure()
    sns.scatterplot(
        data=df,
        x="dien_tich",
        y="gia",
        hue=is_outlier,
        palette={True: "red", False: "blue"},
    )
    plt.title("Multivariate Outlier Detection (Area vs Price)")
    plt.legend(title="Is Outlier (Z>3)")
    plt.show()

    print("Thảo luận: Các điểm màu đỏ là ngoại lệ đa biến/đơn biến cực đoan.")


if __name__ == "__main__":
    bai_1_housing()
    bai_2_iot()
    bai_3_ecommerce()
    bai_4_multivariate()

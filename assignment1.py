import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def exploratory_data_analysis(df):
    print("--- 1.1 Tổng quan dữ liệu ---")
    print(df.info())

    print("\n--- 1.2 Thống kê mô tả (bao gồm Missing & Duplicate) ---")
    stats = df.describe(include="all")
    stats.loc["missing"] = df.isnull().sum()
    stats.loc["duplicate"] = df.duplicated().sum()
    print(stats)

    print("\n--- 1.3 Trực quan hóa dữ liệu số ---")

    if "price" in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(df["price"], kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title("Histogram: Phân phối giá")

        sns.boxplot(y=df["price"], ax=axes[1], color="lightgreen")
        axes[1].set_title("Boxplot: Phát hiện Outliers")

        sns.violinplot(y=df["price"], ax=axes[2], color="salmon")
        axes[2].set_title("Violin Plot: Mật độ phân phối")
        plt.show()

    print("\n--- 1.4 Phân tích Categorical (Vị trí/Khu vực) ---")
    if "location" in df.columns:
        plt.figure(figsize=(10, 6))
        df["location"].value_counts().plot(kind="bar", color="orange")
        plt.title("Số lượng tin đăng theo khu vực")
        plt.ylabel("Số lượng")
        plt.show()


def clean_proptech_data(df):
    df_cleaned = df.copy()

    # 2.1 Điền missing values (chỉ thực hiện nếu cột tồn tại)
    if "price" in df_cleaned.columns:
        df_cleaned["price"] = df_cleaned["price"].fillna(df_cleaned["price"].median())
    if "area" in df_cleaned.columns:
        df_cleaned["area"] = df_cleaned["area"].fillna(df_cleaned["area"].median())

    if "rooms" in df_cleaned.columns:
        rooms_mode = df_cleaned["rooms"].mode()
        if not rooms_mode.empty:
            df_cleaned["rooms"] = df_cleaned["rooms"].fillna(rooms_mode[0])

    # 2.2 Loại bỏ giá trị không hợp lệ
    if "price" in df_cleaned.columns and "area" in df_cleaned.columns:
        df_cleaned = df_cleaned[(df_cleaned["price"] > 0) & (df_cleaned["area"] > 0)]

    if "location" in df_cleaned.columns:
        mapping = {
            "Ha Noi": "Hà Nội",
            "Hanoi": "Hà Nội",
            "HCM": "TP.HCM",
            "Sai Gon": "TP.HCM",
        }
        df_cleaned["location"] = df_cleaned["location"].replace(mapping)

    before_count = len(df_cleaned)
    df_cleaned.drop_duplicates(
        subset=["price", "area", "location"], keep="first", inplace=True
    )
    after_count = len(df_cleaned)
    print(f"Đã loại bỏ {before_count - after_count} bản ghi trùng lặp.")

    return df_cleaned


if __name__ == "__main__":
    data = {
        "price": [2500, 3000, np.nan, 4500, 15000, 3200, 3000, -500],
        "area": [50, 60, 55, np.nan, 250, 62, 60, 45],
        "location": [
            "Hanoi",
            "HCM",
            "Hà Nội",
            "Sai Gon",
            "Da Nang",
            "Hanoi",
            "Hanoi",
            "TP.HCM",
        ],
        "rooms": [2, 3, 2, 4, 6, np.nan, 3, 2],
    }
    df_test = pd.DataFrame(data)

    print("=== CHƯƠNG TRÌNH PHÂN TÍCH DỮ LIỆU BẤT ĐỘNG SẢN ===")

    exploratory_data_analysis(df_test)

    print("\n--- 2. Bắt đầu làm sạch dữ liệu ---")
    df_final = clean_proptech_data(df_test)

    print("\n--- Kết quả sau khi làm sạch ---")
    print(df_final)
    print("\nThống kê sau khi làm sạch:")
    print(df_final.describe())

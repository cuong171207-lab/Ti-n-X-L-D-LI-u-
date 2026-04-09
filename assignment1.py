import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import difflib

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



# --- PHẦN 3: XỬ LÝ OUTLIERS & SKEW ---
def handle_outliers_and_skew(df, column):
    print(f"\n--- 3. Xử lý Outliers cho cột: {column} ---")
    
    # Tính toán IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Chiến lược: Capping (giới hạn giá trị thay vì xóa bỏ để giữ data)
    outliers_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
    print(f"Phát hiện {outliers_count} outliers. Thực hiện Capping...")
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# --- PHẦN 4: CHUẨN HÓA SỐ & BIẾN ĐỔI CATEGORICAL ---
def transform_features(df):
    print("\n--- 4. Chuẩn hóa & Biến đổi dữ liệu ---")
    df_transformed = df.copy()
    
    # 4.1 Scaling Numerical (Sử dụng Min-Max cho Price)
    scaler = MinMaxScaler()
    if "price" in df_transformed.columns:
        df_transformed["price_scaled"] = scaler.fit_transform(df_transformed[["price"]])
        print("- Đã tạo cột 'price_scaled' bằng Min-Max Scaling.")
    
    # 4.2 Encoding Categorical (Sử dụng Label Encoding cho Location)
    if "location" in df_transformed.columns:
        le = LabelEncoder()
        df_transformed["location_encoded"] = le.fit_transform(df_transformed["location"])
        print("- Đã tạo cột 'location_encoded' bằng Label Encoding.")
        
    return df_transformed

# --- PHẦN 5: PHÁT HIỆN TRÙNG LẶP DỰA TRÊN TEXT SIMILARITY ---
def detect_text_duplicates(df, text_column):
    print(f"\n--- 5. Phát hiện trùng lặp dựa trên độ tương đồng: {text_column} ---")
    
    # Giả lập một cột mô tả nếu chưa có để demo
    if text_column not in df.columns:
        df[text_column] = [
            "Căn hộ cao cấp quận 1, gần chợ",
            "Căn hộ cao cấp Q1, gần chợ Bến Thành", # Tương đồng cao
            "Nhà phố giá rẻ",
            "Nhà phố giá rẻ, chính chủ", # Tương đồng cao
            "Đất nền ven đô",
            "Căn hộ cao cấp quận 1, gần chợ" # Trùng 100%
        ]
    
    descriptions = df[text_column].tolist()
    to_merge = []

    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            # Tính độ tương đồng từ 0 đến 1
            similarity = difflib.SequenceMatcher(None, descriptions[i], descriptions[j]).ratio()
            if similarity > 0.7: # Ngưỡng 70% là tương đồng cao
                print(f"Gợi ý merge: \n  - '{descriptions[i]}' \n  - '{descriptions[j]}' \n  (Độ tương đồng: {similarity:.2f})")
                to_merge.append(j)
                
    # Loại bỏ các bản ghi bị trùng lặp cao (keep first)
    df_final = df.drop(df.index[list(set(to_merge))]).reset_index(drop=True)
    print(f"Đã xử lý xong. Còn lại {len(df_final)} bản ghi.")
    return df_final

# --- CẬP NHẬT HÀM MAIN ---
if __name__ == "__main__":
    # Giả lập dữ liệu như trong code của bạn
    data = {
        "price": [2500, 3000, 2800, 4500, 15000, 3200, 3000, 2200], # 15000 là outlier
        "area": [50, 60, 55, 70, 250, 62, 60, 45],
        "location": ["Hà Nội", "TP.HCM", "Hà Nội", "TP.HCM", "Đà Nẵng", "Hà Nội", "Hà Nội", "TP.HCM"],
        "rooms": [2, 3, 2, 4, 6, 3, 3, 2],
    }
    df_test = pd.DataFrame(data)

    print("=== CHƯƠNG TRÌNH PHÂN TÍCH DỮ LIỆU NÂNG CAO ===")

    # 1 & 2: Đã có trong code của bạn (coi như df_cleaned đã xong)
    df_cleaned = df_test 

    # 3. Xử lý Outliers
    df_no_outlier = handle_outliers_and_skew(df_cleaned, "price")

    # 4. Chuẩn hóa & Encoding
    df_transformed = transform_features(df_no_outlier)

    # 5. Xử lý trùng lặp văn bản
    # Giả định cột mô tả nhà là 'description'
    df_final_result = detect_text_duplicates(df_transformed, "description")

    print("\n--- BẢNG DỮ LIỆU CUỐI CÙNG ---")
    print(df_final_result[["price", "price_scaled", "location", "location_encoded", "description"]].head())
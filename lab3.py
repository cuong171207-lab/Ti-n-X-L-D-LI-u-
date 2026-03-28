import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

base_path = os.path.dirname(os.path.abspath(__file__))
files = {
    "Bài 1": os.path.join(base_path, "ITA105_Lab_3_Sports.csv"),
    "Bài 2": os.path.join(base_path, "ITA105_Lab_3_Health.csv"),
    "Bài 3": os.path.join(base_path, "ITA105_Lab_3_Finance.csv"),
    "Bài 4": os.path.join(base_path, "ITA105_Lab_3_Gaming.csv"),
}

def process_lab_3(file_path, title, is_finance=False, is_health=False):
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    if not os.path.exists(file_path):
        print(f"[LỖI] Không tìm thấy file tại: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    print("- Kiểm tra missing values:\n", df.isnull().sum())
    print("- Thống kê mô tả:\n", df.describe())

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f"Phân phối dữ liệu gốc - {title}", fontsize=16)

    cols = df.columns[:4] 
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[0, i // 2 if len(cols) > 2 else i])
        sns.boxplot(x=df[col], ax=axes[1, i // 2 if len(cols) > 2 else i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    mm_scaler = MinMaxScaler()
    df_minmax = pd.DataFrame(mm_scaler.fit_transform(df), columns=df.columns)

    z_scaler = StandardScaler()
    df_zscore = pd.DataFrame(z_scaler.fit_transform(df), columns=df.columns)

    target_col = df.columns[0]
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(df[target_col], kde=True, color="blue")
    plt.title(f"Gốc: {target_col}")

    plt.subplot(1, 3, 2)
    sns.histplot(df_minmax[target_col], kde=True, color="green")
    plt.title(f"Min-Max: {target_col}")

    plt.subplot(1, 3, 3)
    sns.histplot(df_zscore[target_col], kde=True, color="red")
    plt.title(f"Z-Score: {target_col}")

    plt.suptitle(f"So sánh phân phối - {title}", fontsize=16)
    plt.show()

    if is_finance:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x="doanh_thu_musd", y="loi_nhuan_musd")
        plt.title("Gốc: Doanh thu vs Lợi nhuận")

        plt.subplot(1, 2, 2)
        sns.scatterplot(data=df_zscore, x="doanh_thu_musd", y="loi_nhuan_musd")
        plt.title("Z-Score: Doanh thu vs Lợi nhuận")
        plt.show()


process_lab_3(files["Bài 1"], "Bài 1: Thông số vận động viên")

process_lab_3(files["Bài 2"], "Bài 2: Chỉ số bệnh nhân", is_health=True)
print("\nNHẬN XÉT BÀI 2:")
print(
    "- Các biến có ngoại lệ cực đoan (như BMI hoặc Huyết áp) khiến Min-Max bị dồn dữ liệu."
)
print(
    "- Z-Score phù hợp hơn vì nó giữ được hình dạng phân phối và không bị bó hẹp bởi các điểm cực trị."
)

process_lab_3(files["Bài 3"], "Bài 3: Chỉ số công ty", is_finance=True)
print("\nTHẢO LUẬN BÀI 3:")
print(
    "- Với dữ liệu tài chính có ngoại lệ lớn (Big Tech), Min-Max không phù hợp vì đa số công ty sẽ bị ép về gần mức 0."
)
print(
    "- Nên chọn Z-Score cho Linear Regression và KNN để đảm bảo các biến có tầm ảnh hưởng ngang nhau."
)

process_lab_3(files["Bài 4"], "Bài 4: Người chơi trực tuyến")
print("\nTHẢO LUẬN BÀI 4:")
print("- Với game thủ 'cày cuốc' (ngoại lệ), Z-Score ổn định hơn.")
print(
    "- Chọn Z-Score cho Clustering/KNN để tránh việc các biến có scale lớn (như điểm tích lũy) lấn át các biến khác."
)

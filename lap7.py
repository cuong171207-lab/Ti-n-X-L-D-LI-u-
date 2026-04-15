import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def bai1_kham_pha(df):
    print("--- BÀI 1: KHÁM PHÁ PHÂN PHỐI DỮ LIỆU ---")
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    skew_vals = df[num_cols].apply(skew).sort_values(ascending=False, key=abs)
    print("Top cac cot co do lech (skewness) cao nhat:\n", skew_vals)
    
    top3_cols = skew_vals.head(3).index
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, col in enumerate(top3_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'{col} (Skew: {skew_vals[col]:.2f})')
        
    plt.tight_layout()
    plt.savefig('b1_top3_skewed_columns.png')
    plt.close()
    
    print("\n[PHÂN TÍCH BÀI 1]")
    print("- Dữ liệu bị lệch (skewed) thường do sự xuất hiện của các giá trị ngoại lệ (outliers) quá lớn hoặc quá nhỏ.")
    print("- Skewness ảnh hưởng xấu đến các mô hình tuyến tính (Linear Regression) vì nó làm vi phạm giả định phân phối chuẩn, khiến mô hình nhạy cảm với nhiễu.")
    print("- Đề xuất: Cần dùng Log Transform, Box-Cox hoặc Power Transform để kéo phân phối về hình quả chuông (Normal Distribution).")

def bai2_bien_doi(df):
    print("\n--- BÀI 2: BIẾN ĐỔI DỮ LIỆU NÂNG CAO ---")
    col_pos1 = 'SalePrice'
    col_pos2 = 'LotArea'
    col_neg = 'NegSkewIncome'
    
    df_trans = df.copy()
    
    df_trans[f'{col_pos1}_log'] = np.log1p(df_trans[col_pos1])
    
    df_trans[f'{col_pos2}_boxcox'], lmbda = boxcox(df_trans[col_pos2] + 1)
    
    pt = PowerTransformer(method='yeo-johnson')
    df_trans[f'{col_neg}_yeo'] = pt.fit_transform(df_trans[[col_neg]])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    sns.histplot(df[col_pos1], kde=True, ax=axes[0, 0]).set_title(f'Goc: {col_pos1}')
    sns.histplot(df_trans[f'{col_pos1}_log'], kde=True, ax=axes[1, 0], color='orange').set_title(f'Log: {col_pos1}')
    
    sns.histplot(df[col_pos2], kde=True, ax=axes[0, 1]).set_title(f'Goc: {col_pos2}')
    sns.histplot(df_trans[f'{col_pos2}_boxcox'], kde=True, ax=axes[1, 1], color='green').set_title(f'Box-Cox: {col_pos2}')
    
    sns.histplot(df[col_neg], kde=True, ax=axes[0, 2]).set_title(f'Goc: {col_neg}')
    sns.histplot(df_trans[f'{col_neg}_yeo'], kde=True, ax=axes[1, 2], color='red').set_title(f'Yeo-Johnson: {col_neg}')
    
    plt.tight_layout()
    plt.savefig('b2_transformations_compare.png')
    plt.close()
    
    print("Bang so sanh Skewness truoc va sau bien doi:")
    print(f"1. {col_pos1}: Goc = {skew(df[col_pos1]):.2f} | Sau Log = {skew(df_trans[f'{col_pos1}_log']):.2f}")
    print(f"2. {col_pos2}: Goc = {skew(df[col_pos2]):.2f} | Sau BoxCox = {skew(df_trans[f'{col_pos2}_boxcox']):.2f}")
    print(f"3. {col_neg}: Goc = {skew(df[col_neg]):.2f} | Sau Yeo-Johnson = {skew(df_trans[f'{col_neg}_yeo']):.2f}")
    
    print("\n[PHÂN TÍCH BÀI 2]")
    print("- Yeo-Johnson là lựa chọn duy nhất và tốt nhất cho cột có giá trị âm vì Log và Box-Cox sẽ báo lỗi (chỉ nhận giá trị > 0).")
    print(f"- Ý nghĩa của tham số Lambda trong Box-Cox (tim duoc {lmbda:.2f}): Là số mũ tối ưu nhất mà thuật toán tự dò tìm để biến đổi dữ liệu sao cho giống phân phối chuẩn nhất.")

def bai3_mo_hinh(df):
    print("\n--- BÀI 3: ỨNG DỤNG VÀO MÔ HÌNH HÓA ---")
    features = ['LotArea', 'HouseAge', 'Rooms', 'MixedFeature']
    X = df[features]
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_A = LinearRegression()
    model_A.fit(X_train, y_train)
    pred_A = model_A.predict(X_test)
    rmse_A = np.sqrt(mean_squared_error(y_test, pred_A))
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    model_B = LinearRegression()
    model_B.fit(X_train, y_train_log)
    pred_B_log = model_B.predict(X_test)
    pred_B = np.expm1(pred_B_log)
    rmse_B = np.sqrt(mean_squared_error(y_test, pred_B))
    
    pt = PowerTransformer()
    X_train_pt = pt.fit_transform(X_train)
    X_test_pt = pt.transform(X_test)
    model_C = LinearRegression()
    model_C.fit(X_train_pt, y_train)
    pred_C = model_C.predict(X_test_pt)
    rmse_C = np.sqrt(mean_squared_error(y_test, pred_C))
    
    print(f"Version A (Du lieu goc)        - RMSE: {rmse_A:.2f}")
    print(f"Version B (Log muc tieu y)     - RMSE: {rmse_B:.2f}")
    print(f"Version C (Power Transform X)  - RMSE: {rmse_C:.2f}")
    
    print("\n[PHÂN TÍCH BÀI 3]")
    print("- Transform giúp gom cụm dữ liệu ngoại lệ, làm giảm phương sai sai số, từ đó giúp mô hình học được xu hướng chung tốt hơn và cải thiện RMSE.")
    print("- Lưu ý quan trọng: Khi dùng Log cho biến mục tiêu (y), dự đoán sinh ra cũng nằm ở hệ Log. Phải dùng np.expm1() để dịch ngược về giá trị thực tế trước khi tính RMSE.")

def bai4_nghiep_vu(df):
    print("\n--- BÀI 4: ỨNG DỤNG NGHIỆP VỤ THỰC TẾ ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.scatterplot(x='LotArea', y='SalePrice', data=df, ax=axes[0], alpha=0.6)
    axes[0].set_title('Version A: Raw Data (Dữ liệu gốc)')
    
    df_trans = df.copy()
    df_trans['LotArea_log'] = np.log1p(df_trans['LotArea'])
    df_trans['SalePrice_log'] = np.log1p(df_trans['SalePrice'])
    
    sns.scatterplot(x='LotArea_log', y='SalePrice_log', data=df_trans, ax=axes[1], color='green', alpha=0.6)
    axes[1].set_title('Version B: Log Transformed Data')
    
    plt.tight_layout()
    plt.savefig('b4_insight_scatter.png')
    plt.close()
    
    print("\n[INSIGHT CHO NGƯỜI KHÔNG CHUYÊN]")
    print("- Tại sao cần biến đổi? Nếu nhìn vào biểu đồ Raw Data, các bất động sản siêu lớn/siêu đắt (outliers) kéo giãn trục tọa độ, khiến hàng ngàn giao dịch bình thường bị dồn cục lại thành một góc mù mờ.")
    print("- Hiệu quả của Biểu đồ Transform: Khi áp dụng hệ Log, biểu đồ như được bóp lại theo tỷ lệ phần trăm thay vì giá trị tuyệt đối. Mối quan hệ giữa Diện tích (LotArea) và Giá nhà (SalePrice) hiện ra rõ ràng hơn thành một đường chéo tuyến tính.")
    print("- Lợi ích thị trường: Giúp nhà phân tích nhận diện được quy luật tăng giá ổn định của số đông thị trường, thay vì bị đánh lừa bởi một vài giao dịch đột biến.")

def main():
    df = pd.read_csv('ITA105_Lab_7.csv')
    bai1_kham_pha(df)
    bai2_bien_doi(df)
    bai3_mo_hinh(df)
    bai4_nghiep_vu(df)
    print("\nHoan tat toan bo Lab 7! Đã luu 4 bieu do thanh file anh.")

if __name__ == "__main__":
    main()
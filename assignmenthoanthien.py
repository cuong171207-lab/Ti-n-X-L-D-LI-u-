import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


def explore_and_clean_data(df):
    print("Thống kê:\n", df.describe())
    print("Giá trị thiếu:\n", df.isnull().sum())

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df["price"], kde=True)
    plt.title("Histogram Price")
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df["price"])
    plt.title("Boxplot Price")
    plt.subplot(1, 3, 3)
    sns.violinplot(y=df["area"])
    plt.title("Violin Area")
    plt.show()

    df = df.drop_duplicates()
    df = df[(df["price"] > 0) & (df["rooms"] > 0)]
    return df


def feature_engineering(df):
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    df["price_per_m2"] = df["price"] / df["area"]
    df["is_luxury"] = (
        df["description"]
        .str.contains("sang trọng|cao cấp|luxury", case=False)
        .astype(int)
    )

    df["area_rooms_interaction"] = df["area"] * df["rooms"]

    return df


def build_and_train_pipeline(df):
    X = df.drop(["price", "date"], axis=1)
    y = np.log1p(df["price"])

    num_features = ["area", "rooms", "month", "quarter", "area_rooms_interaction"]
    cat_features = ["location", "status"]
    text_features = "description"

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
            ("text", TfidfVectorizer(max_features=50), text_features),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    return model_pipeline


def detect_text_duplicates(df):
    tfidf = TfidfVectorizer().fit_transform(df["description"])
    cosine_sim = cosine_similarity(tfidf)
    return cosine_sim

if __name__ == "__main__":
    # Tạo dữ liệu giả lập để chạy thử
    data = {
        'price': [2000, 3500, 5000, 2200, 4800, 3000],
        'area': [50, 70, 100, 55, 95, 65],
        'rooms': [2, 3, 4, 2, 4, 3],
        'location': ['Hà Nội', 'TP.HCM', 'Hà Nội', 'Đà Nẵng', 'TP.HCM', 'Hà Nội'],
        'status': ['Mới', 'Cũ', 'Mới', 'Cũ', 'Mới', 'Mới'],
        'description': [
            'Căn hộ sang trọng view hồ', 'Nhà cấp 4 bình thường', 
            'Luxury apartment cao cấp', 'Nhà nhỏ gọn tiện nghi',
            'Biệt thự cao cấp', 'Chung cư hiện đại'
        ],
        'date': pd.date_range(start='2023-01-01', periods=6)
    }
    df_test = pd.DataFrame(data)
    
    print("Đang xử lý dữ liệu...")
    df_cleaned = explore_and_clean_data(df_test)
    df_features = feature_engineering(df_cleaned)
    model = build_and_train_pipeline(df_features)
    print("Hoàn tất chạy thử nghiệm!")

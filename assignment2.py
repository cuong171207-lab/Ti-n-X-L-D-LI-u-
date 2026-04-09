import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
 
def generate_sample_data(n=200, seed=42):
    """Tạo dataset mẫu đủ phong phú cho Giai đoạn 2."""
    np.random.seed(seed)
    locations = ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ']
    conditions = ['Mới', 'Tốt', 'Trung bình', 'Cần sửa']
    descriptions = [
        "Căn hộ luxury cao cấp, view đẹp, tiện ích đầy đủ",
        "Nhà phố cozy ấm cúng, gần chợ, yên tĩnh",
        "Biệt thự rộng rãi, hồ bơi riêng, an ninh tốt",
        "Chung cư bình dân, giao thông thuận tiện",
        "Nhà cấp 4, diện tích vừa, giá hợp lý",
        "Penthouse luxury tầng cao, view toàn thành phố",
        "Căn hộ studio nhỏ gọn, cozy, tiết kiệm chi phí",
        "Nhà liền kề, sân vườn, không gian sống thoải mái",
    ]
 
    df = pd.DataFrame({
        'price':       np.random.lognormal(mean=8.5, sigma=0.8, size=n) * 100,
        'area':        np.random.lognormal(mean=4.2, sigma=0.6, size=n),
        'rooms':       np.random.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.05, 0.25, 0.35, 0.25, 0.08, 0.02]),
        'location':    np.random.choice(locations, size=n),
        'condition':   np.random.choice(conditions, size=n),
        'description': np.random.choice(descriptions, size=n),
        'build_date':  pd.to_datetime('2010-01-01') + pd.to_timedelta(np.random.randint(0, 4000, n), unit='D'),
        'sell_date':   pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 700, n), unit='D'),
    })
 
    for col in ['price', 'area', 'rooms']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
 
    return df
 
 
def feature_engineering(df):
    """Tạo features nâng cao từ ngày, text, số."""
    df = df.copy()
 
    df['sell_month']   = df['sell_date'].dt.month
    df['sell_quarter'] = df['sell_date'].dt.quarter
    df['sell_season']  = df['sell_month'].map(
        {12:4, 1:4, 2:4, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}
    )
    df['days_to_sell'] = (df['sell_date'] - df['build_date']).dt.days
    df['house_age']    = (pd.Timestamp('today') - df['build_date']).dt.days / 365
 
    df['desc_word_count'] = df['description'].fillna('').str.split().str.len()
    df['is_luxury']       = df['description'].fillna('').str.lower()\
.str.contains('luxury|penthouse|biệt thự').astype(int)
    df['is_cozy']         = df['description'].fillna('').str.lower()\
                              .str.contains('cozy|ấm cúng|nhỏ gọn').astype(int)
 
    df['price_per_m2']  = df['price'] / df['area'].replace(0, np.nan)
    df['luxury_score']  = df['is_luxury'] * 2 + (df['rooms'] >= 4).astype(int)
    df['log_price']     = np.log1p(df['price'])
    df['log_price_index'] = (df['log_price'] - df['log_price'].mean()) / df['log_price'].std()
 
    df['area_x_rooms']  = df['area'] * df['rooms']
 
    return df
 
 
def build_pipeline():
    """Pipeline sklearn tái sử dụng: numerical + categorical."""
    numerical_features = ['area', 'rooms', 'sell_month', 'sell_quarter',
                          'days_to_sell', 'house_age', 'desc_word_count',
                          'is_luxury', 'is_cozy', 'area_x_rooms']
    categorical_features = ['location', 'condition']
 
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  PowerTransformer(method='yeo-johnson'))
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
 
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features),
    ])
    return preprocessor, numerical_features, categorical_features
 
 
def test_pipeline_on_new_data(preprocessor, numerical_features, categorical_features):
    """Kiểm thử pipeline với dataset mới (unseen categories)."""
    new_data = pd.DataFrame({
        'area': [80, None], 'rooms': [3, 4],
        'sell_month': [6, 11], 'sell_quarter': [2, 4],
        'days_to_sell': [400, 800], 'house_age': [5, 12],
        'desc_word_count': [8, 6], 'is_luxury': [0, 1],
        'is_cozy': [1, 0], 'area_x_rooms': [240, None],
        'location': ['Hà Nội', 'Vũng Tàu'],
        'condition': ['Mới', 'Unknown'],
    })
    try:
        result = preprocessor.transform(new_data)
        print(f"✅ Pipeline test PASSED — shape: {result.shape}")
    except Exception as e:
        print(f"❌ Pipeline test FAILED: {e}")
 
 
def train_and_compare_models(X_train, X_test, y_train, y_test, y_log_train, y_log_test):
    """Huấn luyện và so sánh 4 mô hình, bao gồm log-transform target."""
    models = {
        'Linear Regression':   LinearRegression(),
        'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
 
    results = []
 
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
    results.append({'Model': name, 'Target': 'raw', 'RMSE': round(rmse, 2),
                         'R2': round(r2, 4), 'MAE': round(mae, 2)})
    model.fit(X_train, y_log_train)
    y_pred_log = model.predict(X_test)
    y_pred_exp = np.expm1(y_pred_log)
    rmse_l = np.sqrt(mean_squared_error(y_test, y_pred_exp))
    r2_l   = r2_score(y_test, y_pred_exp)
    mae_l  = mean_absolute_error(y_test, y_pred_exp)
    results.append({'Model': name, 'Target': 'log', 'RMSE': round(rmse_l, 2),
                         'R2': round(r2_l, 4), 'MAE': round(mae_l, 2)})
 
    results_df = pd.DataFrame(results).sort_values('RMSE')
    print("\n📊 Bảng so sánh mô hình:")
    print(results_df.to_string(index=False))
    return results_df
 
 
def analyze_kpi(df):
    """Tạo và in KPI theo khu vực."""
    print("\n📍 Giá trung bình theo khu vực:")
    kpi = df.groupby('location').agg(
        avg_price      = ('price', 'mean'),
        avg_price_m2   = ('price_per_m2', 'mean'),
        avg_luxury     = ('luxury_score', 'mean'),
        count          = ('price', 'count'),
    ).round(2)
    print(kpi)
 
    top5 = df[df['price'] >= df['price'].quantile(0.95)]
    print(f"\n🏆 Top 5% giá cao: {len(top5)} bất động sản")
    print(top5[['location', 'price', 'area', 'price_per_m2', 'luxury_score']].head(10))
 
    q33, q66 = df['price'].quantile(0.33), df['price'].quantile(0.66)
    df['segment'] = pd.cut(df['price'], bins=[-np.inf, q33, q66, np.inf],
                           labels=['Bình dân', 'Trung cấp', 'Cao cấp'])
    print("\n👥 Phân khúc khách hàng:")
    print(df['segment'].value_counts())
    return df
 
 
def dashboard(df, results_df):
    """Vẽ dashboard gồm 6 biểu đồ."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PropTech Dashboard – Giai đoạn 2', fontsize=16, fontweight='bold')
 
    sns.histplot(df['price'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Raw Price Distribution')
 
    sns.histplot(df['log_price'], kde=True, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Log-Price Distribution (sau transform)')
 
    df.groupby('location')['price_per_m2'].mean().sort_values().plot(
        kind='barh', ax=axes[0, 2], color='mediumseagreen')
    axes[0, 2].set_title('Giá trung bình / m² theo khu vực')
 
    df['luxury_score'].value_counts().sort_index().plot(
        kind='bar', ax=axes[1, 0], color='gold')
    axes[1, 0].set_title('Phân phối Luxury Score')
 
    pivot = results_df.pivot(index='Model', columns='Target', values='RMSE')
    pivot.plot(kind='bar', ax=axes[1, 1], colormap='Set2')
    axes[1, 1].set_title('RMSE: Raw vs Log Target')
    axes[1, 1].tick_params(axis='x', rotation=30)
 
    locations = df['location'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(locations)))
    for loc, c in zip(locations, colors):
        sub = df[df['location'] == loc]
    axes[1, 2].scatter(sub['area'], sub['price'], alpha=0.5, label=loc, color=c, s=20)
    axes[1, 2].set_xlabel('Diện tích (m²)')
    axes[1, 2].set_ylabel('Giá (triệu VNĐ)')
    axes[1, 2].set_title('Diện tích vs Giá theo khu vực')
    axes[1, 2].legend(fontsize=7)
 
    plt.tight_layout()
    plt.savefig('dashboard_phase2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Dashboard đã lưu: dashboard_phase2.png")
 
 
if __name__ == "__main__":
    print("=" * 60)
    print("  ITA105 – GIAI ĐOẠN 2: Pipeline & Mô hình dự báo")
    print("=" * 60)
 
    df_raw = generate_sample_data(n=300)
    print(f"Dataset gốc: {df_raw.shape}")
 
    df = feature_engineering(df_raw)
    df.dropna(subset=['price'], inplace=True)
    print(f"Sau feature engineering: {df.shape}")
 
    preprocessor, num_feats, cat_feats = build_pipeline()
    all_features = num_feats + cat_feats
    X = df[all_features].copy()
    y = df['price']
    y_log = df['log_price']
 
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    print(f"\n✅ Pipeline fit xong — X shape sau transform: {X_transformed.shape}")
    test_pipeline_on_new_data(preprocessor, num_feats, cat_feats)
 
    X_tr, X_te, y_tr, y_te, yl_tr, yl_te = train_test_split(
        X_transformed, y, y_log, test_size=0.3, random_state=42)
    results_df = train_and_compare_models(X_tr, X_te, y_tr, y_te, yl_tr, yl_te)
 
    df = analyze_kpi(df)
 
    dashboard(df, results_df)
 
    print("\n✅ Giai đoạn 2 hoàn thành!")
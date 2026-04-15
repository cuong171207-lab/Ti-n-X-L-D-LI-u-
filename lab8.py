import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings

warnings.filterwarnings('ignore')

class OutlierCapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lower_ = np.nanpercentile(X, 5, axis=0)
        self.upper_ = np.nanpercentile(X, 95, axis=0)
        return self
        
    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)

class DateExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_date = pd.to_datetime(X.iloc[:, 0], errors='coerce')
        return np.c_[X_date.dt.month.fillna(0), X_date.dt.year.fillna(0)]

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X.iloc[:, 0].fillna('').astype(str).str.lower()

def bai1_pipeline(df):
    print("--- BÀI 1: XÂY DỰNG PIPELINE TỔNG QUÁT ---")
    num_cols = ['LotArea', 'Rooms', 'HasGarage', 'NoiseFeature']
    cat_cols = ['Neighborhood', 'Condition']
    text_col = 'Description'
    date_col = ['SaleDate']

    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('capper', OutlierCapper()),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer())
    ])

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    text_pipe = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=20, stop_words='english'))
    ])

    date_pipe = Pipeline([
        ('extractor', DateExtractor()),
        ('impute', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
        ('text', text_pipe, [text_col]),
        ('date', date_pipe, date_col)
    ], remainder='drop')

    X_demo = df.head(10)
    X_trans = preprocessor.fit_transform(X_demo)
    
    print(f"Shape sau transform 10 dong (Smoke Test): {X_trans.shape}")
    print("Pipeline hoat dong on dinh!")
    
    return preprocessor, num_cols, cat_cols, text_col, date_col

def bai2_kiem_thu(df, preprocessor):
    print("\n--- BÀI 2: KIỂM THỬ PIPELINE ---")
    preprocessor.fit(df)
    
    df_missing = df.copy()
    df_missing.loc[0:10, 'LotArea'] = np.nan
    df_missing.loc[0:10, 'Neighborhood'] = np.nan
    
    df_skewed = df.copy()
    df_skewed.loc[0, 'LotArea'] = 999999999
    
    df_unseen = df.copy()
    df_unseen.loc[0, 'Neighborhood'] = 'Z_Khu_Vuc_Moi_Tinh'
    
    df_wrong = df.copy()
    df_wrong['Rooms'] = df_wrong['Rooms'].astype(str)
    df_wrong.loc[0, 'Rooms'] = 'Ba phong'

    tests = {
        'Du lieu day du': df.copy(), 
        'Du lieu Missing': df_missing, 
        'Du lieu Skewed/Outlier': df_skewed, 
        'Du lieu Unseen Category': df_unseen
    }
    
    for name, data in tests.items():
        try:
            res = preprocessor.transform(data)
            print(f"Test '{name}': THANH CONG. Shape output: {res.shape}")
        except Exception as e:
            print(f"Test '{name}': LOI - {e}")
            
    try:
        data_wrong = df_wrong.copy()
        data_wrong['Rooms'] = pd.to_numeric(data_wrong['Rooms'], errors='coerce')
        res = preprocessor.transform(data_wrong)
        print("Test 'Sai dinh dang': THANH CONG (Sau khi ep kieu pd.to_numeric).")
    except Exception as e:
        print(f"Test 'Sai dinh dang': LOI - {e}")

    orig_lotarea = df['LotArea']
    trans_lotarea = preprocessor.transform(df)[:, 0] 
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(orig_lotarea, kde=True, ax=axes[0]).set_title('LotArea Goc (Truoc Pipeline)')
    sns.histplot(trans_lotarea, kde=True, ax=axes[1]).set_title('LotArea Sau Pipeline (Capped & Scaled)')
    plt.savefig('b2_pipeline_distribution.png')
    plt.close()
    
    print("\nBao cao kiem dinh:")
    print("- Outlier Capper chay tot, chan tran/day giup scale khong bi hu.")
    print("- OneHotEncoder xu ly category la an toan bang cach bo qua (ignore matrix = 0).")
    print("- Da luu bieu do so sanh phan phoi vao file 'b2_pipeline_distribution.png'.")

def bai3_mo_hinh(df, preprocessor, num_cols, cat_cols, text_col, date_col):
    print("\n--- BÀI 3: TÍCH HỢP MÔ HÌNH DỰ BÁO ---")
    X = df[num_cols + cat_cols + [text_col] + date_col]
    y = df['SalePrice']

    pipe_lr = Pipeline([('prep', preprocessor), ('model', LinearRegression())])
    pipe_rf = Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(random_state=42))])

    cv_lr = cross_val_score(pipe_lr, X, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rf = cross_val_score(pipe_rf, X, y, cv=5, scoring='neg_root_mean_squared_error')

    print(f"Linear Regression (5-fold) RMSE: {-cv_lr.mean():.2f}")
    print(f"Random Forest (5-fold) RMSE: {-cv_rf.mean():.2f}")
    
    print("\nDanh gia loi ich cua Pipeline:")
    print("- Pipeline giup tu dong hoa quy trinh, han che hoan toan rui ro 'Data Leakage' (ro ri du lieu) khi cross-validation.")
    print("- Transform va Fit luon tach biet giua tap Train/Test trong tung fold, giup ket qua CV that su chuan xac.")
    
    pipe_rf.fit(X, y)
    return pipe_rf

def bai4_trien_khai(model, num_cols, cat_cols, text_col, date_col):
    print("\n--- BÀI 4: TRIỂN KHAI PIPELINE ---")
    joblib.dump(model, 'house_price_pipeline.pkl')
    print("Da dong goi toan bo Pipeline (Transform + Model) vao file: 'house_price_pipeline.pkl'")
    
    loaded_model = joblib.load('house_price_pipeline.pkl')
    
    new_data = pd.DataFrame({
        'LotArea': [8500],
        'Rooms': [4],
        'HasGarage': [1],
        'NoiseFeature': [0.2],
        'Neighborhood': ['B'],
        'Condition': ['Excellent'],
        'Description': ['modern luxury house quiet garden'],
        'SaleDate': ['2026-04-14']
    })
    
    pred = loaded_model.predict(new_data)
    print("\nKet qua du doan tu mo hinh da load cho nha moi nhap:")
    print(f"-> Gia nha du doan: ${pred[0]:,.2f}")

def main():
    df = pd.read_csv('ITA105_Lab_8.csv')
    preprocessor, num_cols, cat_cols, text_col, date_col = bai1_pipeline(df)
    bai2_kiem_thu(df, preprocessor)
    final_model = bai3_mo_hinh(df, preprocessor, num_cols, cat_cols, text_col, date_col)
    bai4_trien_khai(final_model, num_cols, cat_cols, text_col, date_col)
    print("\nHoan tat toan bo Lab 8!")

if __name__ == "__main__":
    main()
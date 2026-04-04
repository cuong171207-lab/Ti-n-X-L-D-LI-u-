import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
 
STOP_WORDS_VI = {
    "và", "của", "là", "có", "trong", "không", "được", "với", "các",
    "cho", "một", "những", "này", "đó", "nhưng", "thì", "về", "hay",
    "cũng", "rất", "vì", "nên", "đã", "từ", "khi", "thế", "như",
    "để", "bị", "tôi", "bạn", "họ", "chúng", "mình", "anh", "chị",
    "ông", "bà", "đây", "còn", "hơn", "lại", "đến", "ra", "lên",
    "xuống", "theo", "trên", "dưới", "trước", "sau", "ngoài", "trong",
    "qua", "tới", "lúc", "vậy", "thật", "quá", "khá", "mà", "thì",
}
 
 
def tien_xu_ly_van_ban(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS_VI and len(t) > 1]
    return tokens
 
 
def hien_thi_tuong_tu(model, tu, topn=5):
    print(f"\n  5 từ gần nghĩa với '{tu}':")
    try:
        ket_qua = model.wv.most_similar(tu, topn=topn)
        for i, (word, score) in enumerate(ket_qua, 1):
            print(f"    {i}. {word:20s}  (similarity = {score:.4f})")
    except KeyError:
        print(f"    [!] Từ '{tu}' không có trong vocabulary.")
        print(f"        Vocabulary: {list(model.wv.index_to_key[:20])} ...")
 
 
def huan_luyen_word2vec(sentences):
    return Word2Vec(sentences=sentences, vector_size=100, window=5,
                    min_count=1, workers=4, epochs=50, sg=1)
 
 
def tao_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix
 
 
def in_tfidf_top(vectorizer, matrix, n=10):
    mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
    terms = vectorizer.get_feature_names_out()
    top_idx = mean_scores.argsort()[-n:][::-1]
    print(f"\n  Top {n} từ có TF-IDF trung bình cao nhất:")
    for idx in top_idx:
        print(f"    {terms[idx]:20s}  {mean_scores[idx]:.4f}")
    print(f"\n  Kích thước TF-IDF matrix: {matrix.shape}")
 
 
def bai1_hotel():
    print("\n" + "=" * 70)
    print("  BÀI 1: REVIEW KHÁCH SẠN")
    print("=" * 70)
 
    df = pd.read_csv("ITA105_Lab_4_Hotel_reviews.csv")
    print("\n--- Thông tin dataset ---")
    print(df.head())
    print(f"\nSố dòng: {len(df)}, Cột: {list(df.columns)}")
 
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    df.dropna(inplace=True)
 
    print("\n--- Encoding ---")
    le = LabelEncoder()
    df["hotel_name_enc"] = le.fit_transform(df["hotel_name"])
    print(f"  Label Encoding 'hotel_name': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    ohe = pd.get_dummies(df["customer_type"], prefix="customer_type")
    df = pd.concat([df, ohe], axis=1)
    print(f"  One-Hot Encoding 'customer_type': {list(ohe.columns)}")
 
    df["tokens"] = df["review_text"].apply(tien_xu_ly_van_ban)
    df["text_processed"] = df["tokens"].apply(lambda x: " ".join(x))
    print("\n--- Ví dụ tiền xử lý ---")
    for i in range(3):
        print(f"  Gốc : {df['review_text'].iloc[i]}")
        print(f"  Xử lý: {df['text_processed'].iloc[i]}\n")
 
    print("\n--- TF-IDF Matrix ---")
    vectorizer, tfidf_matrix = tao_tfidf(df["text_processed"].tolist())
    in_tfidf_top(vectorizer, tfidf_matrix)
 
    print("\n--- Word2Vec ---")
    model = huan_luyen_word2vec(df["tokens"].tolist())
    print(f"  Vocabulary size: {len(model.wv)}")
    hien_thi_tuong_tu(model, "sạch")
 
    print("""
--- Khi nào dùng TF-IDF, khi nào dùng Word2Vec? ---
  TF-IDF:
    + Phù hợp khi cần nhận dạng từ khóa quan trọng (phân loại review).
    + Đơn giản, nhanh, diễn giải được kết quả.
    - Không nắm bắt được ngữ nghĩa / mối quan hệ giữa các từ.
 
  Word2Vec:
    + Học được ngữ nghĩa và quan hệ giữa các từ (sạch ~ thoáng ~ mát).
    + Xử lý tốt từ đồng nghĩa, từ liên quan.
    - Cần corpus lớn để học tốt.
 
  → Dùng TF-IDF khi cần tốc độ và khả năng diễn giải.
  → Dùng Word2Vec khi cần hiểu ngữ nghĩa sâu hơn.
    """)
 
    return df, model, vectorizer, tfidf_matrix
 
 
def bai2_match():
    print("\n" + "=" * 70)
    print("  BÀI 2: BÌNH LUẬN TRẬN ĐẤU")
    print("=" * 70)
 
    df = pd.read_csv("ITA105_Lab_4_Match_comments.csv")
    print("\n--- Thông tin dataset ---")
    print(df.head())
    print(f"\nSố dòng: {len(df)}, Cột: {list(df.columns)}")
 
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    df.dropna(inplace=True)
 
    print("\n--- Encoding ---")
    le = LabelEncoder()
    df["team_enc"] = le.fit_transform(df["team"])
    print(f"  Label Encoding 'team': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    ohe = pd.get_dummies(df["author"], prefix="author")
    df = pd.concat([df, ohe], axis=1)
    print(f"  One-Hot Encoding 'author': tạo {len(ohe.columns)} cột")
 
    df["tokens"] = df["comment_text"].apply(tien_xu_ly_van_ban)
    df["text_processed"] = df["tokens"].apply(lambda x: " ".join(x))
    print("\n--- Ví dụ tiền xử lý ---")
    for i in range(3):
        print(f"  Gốc : {df['comment_text'].iloc[i]}")
        print(f"  Xử lý: {df['text_processed'].iloc[i]}\n")
 
    print("\n--- TF-IDF Matrix ---")
    vectorizer, tfidf_matrix = tao_tfidf(df["text_processed"].tolist())
    in_tfidf_top(vectorizer, tfidf_matrix)
 
    print("\n--- Word2Vec ---")
    model = huan_luyen_word2vec(df["tokens"].tolist())
    print(f"  Vocabulary size: {len(model.wv)}")
    hien_thi_tuong_tu(model, "xuất")
    hien_thi_tuong_tu(model, "sắc")
    print("""
--- So sánh TF-IDF vs Word2Vec cho bình luận trận đấu ---
  TF-IDF:
    + Tốt khi cần tìm từ đặc trưng (tên cầu thủ, đội bóng).
    + Phù hợp phân loại bình luận tích cực / tiêu cực theo từ khóa.
    - Không hiểu được "xuất sắc" và "tuyệt vời" là cùng nghĩa.
 
  Word2Vec:
    + Hiểu được ngữ cảnh: "xuất sắc" ~ "tuyệt vời" ~ "đỉnh cao".
    + Biểu diễn ý nghĩa tốt hơn cho các tác vụ NLP nâng cao.
    - Cần dữ liệu đủ lớn để embedding có ý nghĩa.
 
  → Word2Vec cho kết quả tốt hơn trong phân tích cảm xúc bình luận.
    """)
 
    return df, model, vectorizer, tfidf_matrix
 
 
def bai3_player():
    print("\n" + "=" * 70)
    print("  BÀI 3: FEEDBACK NGƯỜI CHƠI GAME")
    print("=" * 70)
 
    df = pd.read_csv("ITA105_Lab_4_Player_feedback.csv")
    print("\n--- Thông tin dataset ---")
    print(df.head())
    print(f"\nSố dòng: {len(df)}, Cột: {list(df.columns)}")
 
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    df.dropna(inplace=True)
 
    print("\n--- Encoding ---")
    le = LabelEncoder()
    df["player_type_enc"] = le.fit_transform(df["player_type"])
    print(f"  Label Encoding 'player_type': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    ohe = pd.get_dummies(df["device"], prefix="device")
    df = pd.concat([df, ohe], axis=1)
    print(f"  One-Hot Encoding 'device': {list(ohe.columns)}")
 
    df["tokens"] = df["feedback_text"].apply(tien_xu_ly_van_ban)
    df["text_processed"] = df["tokens"].apply(lambda x: " ".join(x))
    print("\n--- Ví dụ tiền xử lý ---")
    for i in range(3):
        print(f"  Gốc : {df['feedback_text'].iloc[i]}")
        print(f"  Xử lý: {df['text_processed'].iloc[i]}\n")
 
    print("\n--- TF-IDF Matrix ---")
    vectorizer, tfidf_matrix = tao_tfidf(df["text_processed"].tolist())
    in_tfidf_top(vectorizer, tfidf_matrix)
 
    print("\n--- Word2Vec ---")
    model = huan_luyen_word2vec(df["tokens"].tolist())
    print(f"  Vocabulary size: {len(model.wv)}")
    hien_thi_tuong_tu(model, "đẹp")
 
    print("""
--- Nên dùng TF-IDF hay Word2Vec để phân loại sentiment người chơi? ---
  → Khuyến nghị: Word2Vec (kết hợp với mô hình phân loại ML)
 
  Lý do:
    1. Feedback ngắn, đa dạng từ ngữ → TF-IDF dễ bỏ sót từ đồng nghĩa.
    2. Word2Vec capture được "đẹp" ≈ "mượt mà" ≈ "ấn tượng".
    3. Vector trung bình câu từ Word2Vec làm đặc trưng cho SVM / Random Forest.
 
  Quy trình: Feedback → Tiền xử lý → Word2Vec → Vector trung bình
             → Classifier → Sentiment (positive/negative)
    """)
 
    return df, model, vectorizer, tfidf_matrix
 
 
def bai4_album():
    print("\n" + "=" * 70)
    print("  BÀI 4: REVIEW ALBUM NHẠC")
    print("=" * 70)
 
    df = pd.read_csv("ITA105_Lab_4_Album_reviews.csv")
    print("\n--- Thông tin dataset ---")
    print(df.head())
    print(f"\nSố dòng: {len(df)}, Cột: {list(df.columns)}")
 
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    df.dropna(inplace=True)
 
    print("\n--- Encoding ---")
    le = LabelEncoder()
    df["genre_enc"] = le.fit_transform(df["genre"])
    print(f"  Label Encoding 'genre': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    ohe = pd.get_dummies(df["platform"], prefix="platform")
    df = pd.concat([df, ohe], axis=1)
    print(f"  One-Hot Encoding 'platform': {list(ohe.columns)}")
 
    df["tokens"] = df["review_text"].apply(tien_xu_ly_van_ban)
    df["text_processed"] = df["tokens"].apply(lambda x: " ".join(x))
    print("\n--- Ví dụ tiền xử lý ---")
    for i in range(3):
        print(f"  Gốc : {df['review_text'].iloc[i]}")
        print(f"  Xử lý: {df['text_processed'].iloc[i]}\n")
 
    print("\n--- TF-IDF Matrix ---")
    vectorizer, tfidf_matrix = tao_tfidf(df["text_processed"].tolist())
    in_tfidf_top(vectorizer, tfidf_matrix)
 
    print("\n--- Word2Vec ---")
    model = huan_luyen_word2vec(df["tokens"].tolist())
    print(f"  Vocabulary size: {len(model.wv)}")
    hien_thi_tuong_tu(model, "sáng")
    hien_thi_tuong_tu(model, "tạo")
 
    print("""
--- So sánh TF-IDF và Word2Vec cho review album nhạc ---
  TF-IDF:
    + Phát hiện tốt từ khóa đặc trưng của từng thể loại nhạc.
    + Thích hợp tìm kiếm review theo chủ đề.
    - Không phân biệt được "sáng tạo" và "độc đáo" là cùng nghĩa.
 
  Word2Vec:
    + Nắm bắt quan hệ ngữ nghĩa: "sáng tạo" ~ "độc đáo" ~ "mới lạ".
    + Phù hợp phân tích cảm xúc và gợi ý album tương tự.
    - Corpus ~160 review ngắn nên embedding chưa thật sự tối ưu.
 
  Nhận xét:
    → TF-IDF phù hợp cho phân tích thống kê và phân loại đơn giản.
    → Word2Vec phù hợp cho sentiment analysis và recommendation.
    → Thực tế thường kết hợp cả hai để tận dụng ưu điểm của mỗi phương pháp.
    """)
 
    return df, model, vectorizer, tfidf_matrix
 
 
if __name__ == "__main__":
    df1, model1, vec1, mat1 = bai1_hotel()
    df2, model2, vec2, mat2 = bai2_match()
    df3, model3, vec3, mat3 = bai3_player()
    df4, model4, vec4, mat4 = bai4_album()
 
    print("\n" + "=" * 70)
    print("  HOÀN THÀNH TẤT CẢ 4 BÀI!")
    print("=" * 70)
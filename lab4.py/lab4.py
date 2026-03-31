import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data ={
    "Hoars": [1,2,3,4,5,6,7,8],
    "Score": [2,4,5,6,7,8,8.5,9]
}
df = pd.DataFrame(data)
print(df)
A = df[ ["Hoars"] ]
B = df[ ["Score"] ]
print(A)
print(B)
model = LinearRegression()
model.fit(A, B)
new_hours = pd.DataFrame([[8]], columns=["Hoars"])
predicted_score = model.predict(new_hours)
print("Điểm dự đoán cho 8 giờ học:", predicted_score)

new_data = pd.DataFrame([[4],[6],[9]], columns=["Hoars"])
predicted_scores = model.predict(new_data)
print("số điểm dự đoán:", predicted_scores)

# vẽ biểu đồ dữ liệu 
plt.scatter(A,B)
plt.plot(A, model.predict(A))
plt.xlabel("Số giờ học")
plt.ylabel("Điểm số")
plt.title("Biểu đồ điểm số dự đoán theo số giờ học")
plt.show()
from sklearn.metrics import r2_score
du_doan= model.predict(A)
r2 = r2_score(B, du_doan)
print("R² score:", r2)
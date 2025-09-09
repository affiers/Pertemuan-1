import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Load dataset
df = pd.read_csv("Admission_Predict.csv")  # download dari Kaggle

# 2. Cek data
print(df.head())
print(df.info())
print(df.describe())

# 3. Visualisasi hubungan CGPA vs Chance of Admit
sns.scatterplot(x="CGPA", y="Chance of Admit ", data=df)
plt.title("Hubungan IPK (CGPA) dengan Peluang Diterima")
plt.show()

# 4. Buat variabel target biner (diterima / tidak)
df["Admitted"] = (df["Chance of Admit "] >= 0.75).astype(int)

# 5. Pisahkan fitur & target
X = df.drop(columns=["Chance of Admit ", "Admitted", "Serial No."])
y = df["Admitted"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluasi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# 9. Feature importance (koefisien)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coef": model.coef_[0]
}).sort_values("Coef", ascending=False)

print(importance)
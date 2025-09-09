import pandas as pd

url = "https://raw.githubusercontent.com/username/repo/main/indonesian_student_performance.csv"
df = pd.read_csv(url)

print(df.head())

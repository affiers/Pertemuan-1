import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://news.detik.com/indeks"
res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(res.text, "html.parser")

articles = []
for item in soup.select("article a"):
    title = item.get_text(strip=True)
    link = item.get("href")
    if title and link:
        articles.append([title, link])

df = pd.DataFrame(articles, columns=["Judul", "Link"])
df.to_csv("berita_detik.csv", index=False, encoding="utf-8")
print(df.head())

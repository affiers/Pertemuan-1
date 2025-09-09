import seaborn as sns
import matplotlib.pyplot as plt

#Data contoh
data = sns.load_dataset("tips")

#membuat scatter plot dengan seaborn
sns.scatterplot(x="total_bill", y="tip", data=data)
plt.title("Hubungan antara Total Bill dan Tip" )
plt.show()
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = r"C:\Users\aamir\Desktop\YC\P\smart-diabetes-detector\src\infrastructure\data\dataset-diabete-68e2810ab0d7e949117525.csv"
data = pnd.read_csv(path)

# print(data.describe())

statisticDataFrame =  pnd.DataFrame({
    "meaddin" : data.median()
})

# missing values
print(data.isnull().sum())

# duplicated values
print(data.duplicated().sum())

# infos
print(data.info())

# data types
print(data.dtypes)

data_columns = num_cols = data.select_dtypes(include=[np.number]).columns
for colu in data_columns:
    plt.figure(figsize=(15, 10))
    data[colu].hist(bins=20, figsize=(15,10), color='#3498db', edgecolor='black')
    plt.suptitle("Distribution des variables numeriques", fontsize=16)
    plt.show()

corr = data.corr()


plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.suptitle("Matrice de correlation", fontsize=16)
plt.show()



sns.pairplot(data,diag_kind="kde",hue="Pregnancies")
plt.suptitle("Relations entre variables", y=1.02)
plt.show()







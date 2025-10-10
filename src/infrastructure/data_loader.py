from sklearn.impute import KNNImputer
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

path = r"C:\Users\aamir\Desktop\YC\P\smart-diabetes-detector\src\infrastructure\data\dataset-diabete-68e2810ab0d7e949117525.csv"
data = pnd.read_csv(path)

statisticDataFrame = pnd.DataFrame({"median": data.median()})
print(data.isnull().sum())
print(data.duplicated().sum())
print(data.info())
print(data.dtypes)

data_columns = data.select_dtypes(include=[np.number]).columns
for col in data_columns:
    plt.figure(figsize=(15, 10))
    data[col].hist(bins=20, color='#3498db', edgecolor='black')
    plt.suptitle("Distribution des variables numeriques", fontsize=16)
    plt.show()

corr = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.suptitle("Matrice de correlation", fontsize=16)
plt.show()

sns.pairplot(data, diag_kind="kde", hue="Pregnancies")
plt.suptitle("Relations entre variables", y=1.02)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.title("Detection visuelle des valeurs aberrantes (Boxplot)")
plt.show()

cols_to_replace = ['Glucose', 'BloodPressure', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=2)
data_imputed = pnd.DataFrame(imputer.fit_transform(data), columns=data.columns)

print("\nValeurs manquantes après imputation KNN :")
print(data_imputed.isnull().sum())

cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
        "DiabetesPedigreeFunction", "Age"]

z_scores = np.abs(zscore(data_imputed[cols]))
outliers = (z_scores > 3)
print(f"\nNombre total de valeurs aberrantes détectées (Z-Score) : {np.sum(outliers)}")

cleanData = data_imputed[(z_scores < 3).all(axis=1)]
print(f"Forme après suppression des outliers : {cleanData.shape}")

scaler = StandardScaler()
num_cols = cleanData.select_dtypes(include=[np.number]).columns
data_scaled = cleanData.copy()
data_scaled[num_cols] = scaler.fit_transform(cleanData[num_cols])

corr_MAT = data_scaled.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_MAT, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation après nettoyage")
plt.show()


for col in cols:
    plt.figure(figsize=(8, 5))
    plt.hist(cleanData[col], bins=20, color='#2ecc71', edgecolor='black')
    plt.title(f"Distribution de {col} après suppression des outliers")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=cleanData[cols])
plt.title("Boxplots après suppression des outliers")
plt.show()

corr_clean = cleanData.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_clean, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation après suppression des outliers")
plt.show()

sns.pairplot(cleanData[cols], diag_kind="kde")
plt.suptitle("Relations entre variables après suppression des outliers", y=1.02)
plt.show()

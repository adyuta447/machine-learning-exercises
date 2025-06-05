import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

test = pd.read_csv("test.csv")
test.head()
test.info()


train = pd.read_csv("train.csv")
train.head()
train.describe(include='all')

missing_values = train.isnull().sum()
missing_values[missing_values > 0]

less = missing_values[missing_values > 1000].index
over = missing_values[missing_values >= 1000].index

# Mengisi nilai hilang pake median
numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

kategorical_features = train[less].select_dtypes(include=['object']).columns

for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])


df = train.drop(columns=over)

missing_values = df.isnull().sum()
missing_values[missing_values > 0]


for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()

# identifikasi outliers menggunakan IQR
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1

# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df_filtered_numeric = df.loc[condition, numeric_features]
 
# Kolom kategorikal
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)

from sklearn.preprocessing import StandardScaler
# Standarisasi fitur numerik 
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train[numeric_features[3]], kde=True)
plt.title("Histogram Sebelum Standardisasi")

# sample = pd.read_csv("sample_submission.csv")
# train.head()
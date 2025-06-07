import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
train.head()
train.describe(include='all')

missing_values = train.isnull().sum()
missing_values[missing_values > 0]

less = missing_values[missing_values > 1000].index
over = missing_values[missing_values >= 1000].index

# Mengisi nilai hilang pake median
numeric_features = train[less].select_dtypes(include=['number']).columns
kategorical_features = train[less].select_dtypes(include=['object']).columns

train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())
 
for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])


df = train.drop(columns=over)

numeric_features = df.select_dtypes(include=['number']).columns

# Mengaatasi Outliers 
# for feature in numeric_features: 
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.show()

# Identifikasi outliers dengan IQR

Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)

IQR = Q3 - Q1

condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)

# Filter baris bebas outlier
df_filtered_numeric = df.loc[condition, numeric_features]

# Gabungkan dengan fitur kategorikal
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)

# Standarisasi fitur numerik
if not numeric_features.empty:
    numeric_features = df.select_dtypes(include=['number']).columns
    print("Kolom numerik untuk distandarisasi:", numeric_features)

    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Visualisasi
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(train[numeric_features[3]], kde=True)
    plt.title("Histogram Sebelum Standarisasi")

    plt.subplot(1, 2, 2)
    sns.histplot(df[numeric_features[3]], kde=True)
    plt.title("Histogram Setelah Standarisasi")
else:
    print("Tidak ada fitur numerik yang tersedia untuk distandarisasi")

# Identifikasi baris duplikat
duplicates = df.duplicated()

print("Baris duplikat:")
print(df[duplicates])

# Menghapus baris duplikat
df = df.drop_duplicates()

print("DataFrame setelah menghapus duplikat:")
print(df)

# Ordinal Encoding

category_features = df.select_dtypes(include=['object']).columns
df[category_features]

# One Hot Encoding
df_one_hot = pd.get_dummies(df, columns=category_features)
df_one_hot

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in category_features: 
    df_lencoder[col] = label_encoder.fit_transform(df[col])

df_lencoder

df_lencoder.head()

# Menghitung jumlah dan persentase missing
missing_values = df_lencoder.isnull().sum()
missing_percentage = (missing_values / len(df_lencoder)) * 100

missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
}).sort_values(by='Missing Values', ascending=False)

missing_data[missing_data['Missing Values'] > 0]

# sample = pd.read_csv("sample_submission.csv")
# train.head()
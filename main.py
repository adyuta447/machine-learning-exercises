import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn

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

# Menghitung jumlah variabel
num_vars = df_lencoder.shape[1]

# Menentukan jumlah baris dan kolom untuk frid
n_cols = 4
n_rows = -(-num_vars // n_cols)

# Membuat subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

# Flatten exes array
axes = axes.flatten()

# Plot setiap variabel
for i, column in enumerate(df_lencoder.columns):
    df_lencoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Menghapus subplot yang tak terpakai
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Menyesuaikan layout
# plt.tight_layout()
# plt.show()

# Visualisasi distribusi data untuk beberapa kolom
columns_to_plot = ['OverallQual', 'YearBuilt', 'LotArea', 'SaleType', 'SaleCondition']
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_lencoder[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()


# Visualisasi korelasi antar variabel numerik
plt.figure(figsize=(12, 10))
correlation_matrix = df_lencoder.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Menghitung korelasi antara variabel target dan semua variabel lain
target_corr = df_lencoder.corr()['SalePrice']

target_corr_sorted = target_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
target_corr_sorted.plot(kind='bar')
plt.title(f'Correlation with SalePrice')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
plt.show()

# Memisahkan fitur (X) dan target (Y)
X = df_lencoder.drop(columns=['SalePrice'])
y = df_lencoder['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Menghitung panjang/jumlah data
print("Jumlah data: ", len(X))

# Menghitung panjang/jumlah data pada x_test
print("Jumlah data latih: ", len(x_train))

# Menghitung panjang/jumlah data pada x_test
print("Jumlah data test: ", len(x_test))

# Melatih model 1 Angle Regression
lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)

# Melatih model 2 Linear Regression
LR = LinearRegression().fit(x_train, y_train)

# Melatih model 3 Gradient Boosting Regressor
GBR = GradientBoostingRegressor(random_state=184)
GBR.fit(x_train, y_train)

# Evaluasi model LARS
pred_lars = lars.predict(x_test)
mae_lars = mean_absolute_error(y_test, pred_lars)
mse_lars = mean_squared_error(y_test, pred_lars)
r2_lars = r2_score(y_test, pred_lars)

data = {
    'MAE': [mae_lars],
    'MSE': [mse_lars],
    'R2': [r2_lars]
}

# Konversi dictionary
df_results = pd.DataFrame(data, index=['Lars'])
df_results

# sample = pd.read_csv("sample_submission.csv")
# train.head()
# student_dropout_etl.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# -------------------------------
# 1️⃣ Carga de datos
# -------------------------------
file_path = "student_dropout.csv"  # Asegúrate de que esté en la misma carpeta
df = pd.read_csv(file_path)

print("Datos originales:")
print(df.info())
print(df.head())

# -------------------------------
# 2️⃣ Limpieza de datos
# -------------------------------

# 2a. Valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Rellenar valores faltantes numéricos con la media
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Rellenar valores faltantes categóricos con el modo
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 2b. Tipos de datos
print("\nTipos de datos después de limpieza:")
print(df.dtypes)

# 2c. Detección de valores atípicos (outliers) usando IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in num_cols:
    df = remove_outliers(df, col)

print("\nDatos después de eliminar outliers:")
print(df.describe())

# 2d. Codificación de variables categóricas
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# -------------------------------
# 3️⃣ Guardar dataset limpio
# -------------------------------
clean_file = "student_dropout_clean.csv"
df_encoded.to_csv(clean_file, index=False)
print(f"\nCSV limpio guardado como {clean_file}")

# -------------------------------
# 4️⃣ División en train/validation/test
# -------------------------------
train, temp = train_test_split(df_encoded, test_size=0.3, random_state=42)
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

# Crear carpeta para datasets si no existe
os.makedirs("datasets", exist_ok=True)

train.to_csv("datasets/train.csv", index=False)
validation.to_csv("datasets/validation.csv", index=False)
test.to_csv("datasets/test.csv", index=False)

print("\nDatasets generados:")
print("train.csv:", train.shape)
print("validation.csv:", validation.shape)
print("test.csv:", test.shape)

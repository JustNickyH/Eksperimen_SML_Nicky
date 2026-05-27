import pandas as pd
import os
import pandas as pd
import os

def load_data(file_path):
    print("Membaca data raw Titanic...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Melakukan preprocessing...")
    # 1. Hapus kolom tidak relevan
    df_clean = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # 2. Tangani missing values
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
    
    # 3. Hapus duplikat
    df_clean = df_clean.drop_duplicates()
    
    # 4. Encoding
    df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
    df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)
    
    # Konversi hasil get_dummies (True/False) ke angka (1/0) biar aman saat training model
    for col in df_clean.columns:
        if df_clean[col].dtype == bool:
            df_clean[col] = df_clean[col].astype(int)
            
    return df_clean

def save_data(df, output_path):
    # Pastikan foldernya ada sebelum nyimpen file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Menyimpan data bersih ke {output_path}...")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    RAW_DATA_PATH = "dataset_raw/data_mentah.csv"
    CLEAN_DATA_PATH = "preprocessing/dataset_preprocessing/data_bersih.csv"
    
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    save_data(df_clean, CLEAN_DATA_PATH)
    print("Otomatisasi preprocessing selesai!")
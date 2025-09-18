import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    Класс для предобработки данных автомобилей.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_columns = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']
        self.categorical_columns = ['Manufacturer', 'Model', 'Fuel type']
        self.feature_columns = []  # будем хранить имена финальных признаков

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных: удаление дубликатов и пропусков"""
        initial_len = len(df)
        
        # Удаляем дубликаты
        df = df.drop_duplicates()
        
        # Удаляем строки с пропусками
        df = df.dropna()
        
        cleaned_len = len(df)
        print(f"Удалено {initial_len - cleaned_len} строк (дубликаты + пропуски)")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание новых признаков"""
        current_year = datetime.datetime.now().year
        df['Car Age'] = current_year - df['Year of manufacture']
        df['Mileage per Year'] = df['Mileage'] / df['Car Age'].clip(lower=1)  # избегаем деления на 0
        
        # Добавляем в список числовых столбцов
        self.numeric_columns.extend(['Car Age', 'Mileage per Year'])
        
        print("Созданы новые признаки: Car Age, Mileage per Year")
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование категориальных признаков"""
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                # Для новых данных (не для обучения)
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        print("Категориальные признаки закодированы")
        return df_encoded

    def scale_numeric(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Масштабирование числовых признаков"""
        df_scaled = df.copy()
        
        if fit:
            df_scaled[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
        else:
            df_scaled[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        
        print("Числовые признаки масштабированы")
        return df_scaled

    def get_feature_names(self) -> list:
        """Возвращает список имен финальных признаков"""
        return self.categorical_columns + self.numeric_columns

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Полный пайплайн предобработки"""
        print("Начинаем предобработку...")
        
        df_clean = self.clean_data(df)
        df_with_features = self.create_features(df_clean)
        df_encoded = self.encode_categorical(df_with_features)
        df_final = self.scale_numeric(df_encoded, fit=fit)
        
        self.feature_columns = self.get_feature_names()
        print("Предобработка завершена!")
        return df_final

    def split_data(self, df: pd.DataFrame, target_column: str = 'Price', test_size: float = 0.2):
        """Разделение на train/test"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Train: {len(X_train)} строк, Test: {len(X_test)} строк")
        return X_train, X_test, y_train, y_test
    
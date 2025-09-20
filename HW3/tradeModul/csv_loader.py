import os
import sys
import pandas as pd
from typing import Optional

def load_data(filepath: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла в pandas DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"Ошибка: Файл '{filepath}' не найден.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath, nrows=limit)
        print(f"Данные успешно загружены из '{filepath}'. Форма: {df.shape}")
        return df
    except Exception as e:
        print(f"Ошибка при чтении файла '{filepath}': {e}")
        sys.exit(1)


def save_data(dataframe: pd.DataFrame, filepath: str) -> None:
    """
    Сохраняет DataFrame в CSV-файл.
    """
    try:
        dataframe.to_csv(filepath, index=False)
        print(f"Данные успешно сохранены в '{filepath}'")
    except Exception as e:
        print(f"Ошибка при сохранении файла '{filepath}': {e}")
        sys.exit(1)
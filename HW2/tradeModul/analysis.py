import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  

# Установим общий стиль для всех графиков
sns.set_style("whitegrid")


def show_basic_info(df: pd.DataFrame) -> None:
    """
    Выводит базовую информацию о DataFrame: форма, типы, пропуски, уникальные значения, дубликаты.
    """
    print("\n" + "="*60)
    print("ФОРМА И ТИПЫ ДАННЫХ")
    print("="*60)
    
    print(f"Форма данных (строк, столбцов): {df.shape}")
    print(f"Имена столбцов: {list(df.columns)}")
    print("Типы данных по столбцам:")
    print(df.dtypes)
    
    print("\n Подробная информация о данных:")
    df.info()
    
    print("\n Базовые статистики (только числовые столбцы):")
    print(df.describe())
    
    print("\n Пропущенные значения по столбцам:")
    print(df.isnull().sum())
    
    print("\n" + "="*60)
    print("УНИКАЛЬНЫЕ ЗНАЧЕНИЯ И ПРОПУСКИ ПО СТОЛБЦАМ")
    print("="*60)

    for col in df.columns:
        n_unique = df[col].nunique()
        n_nulls = df[col].isnull().sum()
        print(f"\n Столбец: '{col}'")
        print(f"  Уникальных значений: {n_unique}")
        print(f"  Пропусков (NaN): {n_nulls}")
        if n_unique <= 15:
            print(f"  Уникальные значения: {df[col].unique()}")
    
    print("\n" + "="*60)
    print("ДУБЛИКАТЫ")
    print("="*60)
    duplicates = df.duplicated().sum()
    print(f"Найдено дубликатов: {duplicates}")
    if duplicates > 0:
        print("Примеры дубликатов:")
        print(df[df.duplicated()].head(3))
    print("="*60)


def plot_correlation_matrix(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Строит и отображает тепловую карту корреляции для числовых столбцов.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        print("\n Недостаточно числовых столбцов для построения корреляционной матрицы.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title("Корреляционная матрица", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Корреляционная матрица сохранена: {save_path}")
    else:
        plt.show()


def plot_histograms(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Строит гистограммы для всех числовых столбцов.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("Нет числовых столбцов для построения гистограмм.")
        return

    numeric_df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.set_xlabel('Значение')
        ax.set_ylabel('Частота')

    plt.suptitle("Распределения числовых признаков", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Гистограммы сохранены: {save_path}")
    else:
        plt.show()


def plot_pairplot(df: pd.DataFrame, hue: str | None = None, save_path: str | None = None) -> None:
    """
    Строит pairplot (парные диаграммы рассеяния) с возможной группировкой по hue.
    """
    if hue and hue not in df.columns:
        print(f"Столбец '{hue}' не найден в DataFrame. Pairplot строится без группировки.")
        hue = None

    # Используем только числовые столбцы + hue (если он не числовой, seaborn сам обработает)
    cols_to_plot = list(df.select_dtypes(include=[np.number]).columns)
    if hue and hue not in cols_to_plot:
        cols_to_plot.append(hue)

    plt.figure(figsize=(12, 10))  # Размер влияет только если есть внешний subplot, иначе pairplot сам управляет
    sns.pairplot(df[cols_to_plot], hue=hue, diag_kind="hist")
    plt.suptitle("Парные диаграммы рассеяния", y=1.02, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pairplot сохранён: {save_path}")
    else:
        plt.show()


def plot_single_histogram(df: pd.DataFrame, column: str) -> None:
    """
    Строит гистограмму для одного столбца.
    """
    if column not in df.columns:
        print(f"Столбец '{column}' отсутствует в DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color='blue')
    plt.title(f"Распределение '{column}'", fontsize=14)
    plt.xlabel(column)
    plt.ylabel("Частота")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
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
    
def plot_dependency_graph(df: pd.DataFrame, columnX: str, columnY: str, xlabel: str = "X", ylabel: str = "Y") -> None:   
    # График
    xrange = df[columnX].max()
    plt.figure(figsize=(10, 6))
    plt.bar(df[columnX], df[columnY], color='steelblue', edgecolor='black')
    plt.title('Распределение '+columnX+' по '+columnY+'.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(0, xrange))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def evaluate_model(y_test, y_pred, test_dates=None, title_suffix="", show_plots=True):
    """
    Полная оценка модели регрессии.
    
    Параметры:
        y_test: истинные значения (np.array)
        y_pred: предсказанные значения (np.array)
        test_dates: даты для тестовых точек (опционально, для вывода в таблице)
        show_plots: показывать графики (True/False)
    """
    
    print("=" * 60)
    print(" ОЦЕНКА МОДЕЛИ")
    print("=" * 60)
    
    # 1. Базовые метрики
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print(f" MAE: {mae:.2f}")
    print(f" RMSE: {rmse:.2f}")
    print(f" MAPE: {mape:.2f}%")
    
    # 2. R²
    r2 = r2_score(y_test, y_pred)
    print(f" R²: {r2:.4f}")
    
    # 3. SMAPE
    def smape(y_true, y_pred):
        return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    smape_val = smape(y_test, y_pred)
    print(f" SMAPE: {smape_val:.2f}%")
    
    # 4. Directional Accuracy (если >1 точки)
    if len(y_test) > 1:
        actual_direction = np.sign(y_test[1:] - y_test[:-1])
        pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        print(f" Directional Accuracy: {direction_accuracy:.2f}%")
    else:
        print(" Directional Accuracy не рассчитана (требуется >1 тестовая точка)")
    
    # 5. Таблица с ошибками
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    pct_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    results_df = pd.DataFrame({
        'Datetime': test_dates if test_dates is not None else range(len(y_test)),
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': errors,
        'Abs Error': abs_errors,
        'Pct Error (%)': pct_errors
    })
    
    print("\n Подробный отчет по тестовым данным:")
    print(results_df.to_string(index=False))
    
    # Сохранение в CSV (опционально)
    results_df.to_csv('model_evaluation_report.csv', index=False)
    print(f"\n Отчёт сохранён в файл 'model_evaluation_report.csv'")
    
    # 6. График ошибок
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='red', label='Ошибка')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Ноль')
        plt.title("Ошибки модели (y_true - y_pred) " + title_suffix + "" )
        plt.xlabel("Тестовый пример")
        plt.ylabel("Ошибка")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # 7. Распределение ошибок
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        plt.title("Распределение ошибок модели " + title_suffix + "")
        plt.xlabel("Ошибка (y_true - y_pred)")
        plt.ylabel("Частота")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # 8. Статистика ошибок
    print("\n Статистика ошибок:")
    print(f"Средняя ошибка: {np.mean(errors):.2f}")
    print(f"Медианная ошибка: {np.median(errors):.2f}")
    print(f"Максимальная ошибка: {np.max(np.abs(errors)):.2f}")
    print(f"95% квантиль ошибок: {np.percentile(np.abs(errors), 95):.2f}")
    
    # Возвращаем словарь с метриками (для логирования или сравнения моделей)
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'SMAPE': smape_val,
        'Directional Accuracy': direction_accuracy if len(y_test) > 1 else None,
        'Mean Error': np.mean(errors),
        'Median Error': np.median(errors),
        'Max Abs Error': np.max(np.abs(errors)),
        '95th Percentile Error': np.percentile(np.abs(errors), 95)
    }
    
    return metrics


import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions_and_table(y_test, y_pred, test_dates=None, title_suffix="", show_table=True, show_plot=True):
    """
    Выводит таблицу прогнозов и строит график сравнения реальных и предсказанных значений.
    
    Параметры:
        y_test: истинные значения (np.array)
        y_pred: предсказанные значения (np.array)
        test_dates: даты/метки для тестовых точек (опционально)
        title_suffix: суффикс для заголовка графика
        show_table: показывать таблицу (True/False)
        show_plot: показывать график (True/False)
    """
    print("=" * 60)
    print(" Прогноз vs Факт")
    print("=" * 60)
    
    # Приводим test_dates к 1D, если нужно
    if test_dates is not None:
        if isinstance(test_dates, pd.Index):
            test_dates = test_dates.to_numpy()  # или .tolist()
        elif isinstance(test_dates, np.ndarray) and test_dates.ndim > 1:
            test_dates = test_dates.flatten()
        elif isinstance(test_dates, list):
            pass  # ok
        else:
            test_dates = None  # fallback
    
    # Создаем DataFrame
    results_df = pd.DataFrame({
        'Datetime': test_dates if test_dates is not None else range(len(y_test)),
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    # Вывод таблицы
    if show_table:
        print("\n Прогноз vs Факт:")
        print(results_df.to_string(index=False))
    
    # Построение графика
    if show_plot:
        plt.figure(figsize=(12, 6))
        
        # Если даты — строки (например, "2025-10-01 8ч"), используем их как метки
        x_labels = test_dates if test_dates is not None else range(len(y_test))
        
        plt.plot(x_labels, y_test, label='Фактическая цена закрытия', marker='o', linewidth=2, markersize=6)
        plt.plot(x_labels, y_pred, label='Прогнозная цена закрытия', marker='x', linewidth=2, markersize=6)
        
        plt.title(f'Сравнение фактической и прогнозной цены BTC {title_suffix}')
        plt.xlabel('Дата и время' if test_dates is not None else 'Тестовый пример')
        plt.ylabel('Цена закрытия (USD)')
        
        if test_dates is not None:
            plt.xticks(rotation=45, ha='right')  # поворот меток, если это даты
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return results_df  # Возвращаем DataFrame для дальнейшего использования
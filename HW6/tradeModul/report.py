from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.pyplot as plt

class PrintReport:
    """
    Класс для создания и визуализации результатов.
    """

    def __init__(self):
        self.label_encoders = {}

    @staticmethod
    def print_min_max_metric(y_true, y_pred):
        """
        Печатает минимальные и максимальные значения метрик precision, recall и f1-score.
        
        :param y_true: Истинные метки классов.
        :param y_pred: Предсказанные метки классов.
        :return: Кортеж с максимальной и минимальной метриками (max_metric, min_metric).
        """
        try:
            # Проверка на пустые данные
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("Входные данные (y_true или y_pred) пустые.")
            
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Извлечение метрик для каждого класса
            metrics = []
            for key in report:
                if key.isdigit():
                    metrics.extend([
                        report[key]['precision'],
                        report[key]['recall'],
                        report[key]['f1-score']
                    ])
            
            if not metrics:
                raise ValueError("Отчет не содержит данных о классах.")
            
            max_metric = max(metrics)
            min_metric = min(metrics)
            
            # Вывод результатов
            print("------------------------")
            print(f"Самая высокая метрика: {max_metric:.2f}")
            print(f"Самая низкая метрика: {min_metric:.2f}")
            print("------------------------")
            
            return max_metric, min_metric
        
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return None, None

    @staticmethod
    def save_test_results(filename, test_name, data_size, accuracy, max_metric, min_metric):
        """
        Сохраняет результаты теста в файл.
        :param filename: Имя файла для сохранения данных.
        :param test_name: Название теста.
        :param data_size: Количество данных (размер выборки).
        :param accuracy: Значение Accuracy.
        :param max_metric: Максимальная метрика.
        :param min_metric: Минимальная метрика.
        """
        max_metric = round(max_metric, 3)
        min_metric = round(min_metric, 3)
        
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Test Name", "Data Size", "Accuracy", "Max Metric", "Min Metric"])
            writer.writerow([test_name, data_size, accuracy, max_metric, min_metric])

    @staticmethod
    def plot_accuracy_vs_data_size(filename):
        """
        Строит график зависимости Accuracy от количества данных.
        :param filename: Имя файла с данными.
        """
        import pandas as pd

        # Чтение данных из файла
        if not os.path.isfile(filename):
            print(f"Файл {filename} не найден.")
            return

        data = pd.read_csv(filename)

        # Построение графика
        plt.figure(figsize=(20, 12))
        for test_name in data["Test Name"].unique():
            test_data = data[data["Test Name"] == test_name]
            plt.plot(test_data["Data Size"], test_data["Accuracy"], label=test_name, marker='o')

        plt.title("Зависимость Accuracy от количества данных")
        plt.xlabel("Количество данных")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
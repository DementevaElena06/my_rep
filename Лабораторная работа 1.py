import numpy as np
#Загрузка данных. Мы используем датасет Breast Cancer из библиотеки sklearn - код создает и обучает простую модель, которая учится различать два класса
#датасет предназначен для классификации опухолей молочной железы на два класса:
#Класс 0: Злокачественная опухоль (malignant).
#Класс 1: Доброкачественная опухоль (benign).
#Данные разделяются на обучающую и тестовую выборки, а также масштабируются для улучшения сходимости градиентного спуска
from sklearn.datasets import load_breast_cancer  #будем загружать датасет для тестирования однослойного персептрона с использованием градиентного спуска
from sklearn.model_selection import train_test_split #датасет будет разделен на два набора данных - обучающий (для тренировки) и тестовый для проверки точности его работы
from sklearn.preprocessing import StandardScaler #Масштабируем данные, чтобы они были в одном диапазоне, это помогает модели быстрее и точнее обучаться
#Класс однослойного персептрона
class OneLayerPerceptron:
    def __init__(self, learning_rate=0.01, n_iterations=50000):
      #Инициализация гиперпараметров
      #Инициализация весов и смещений случайными малыми значениями
      #Генератором случайных чисел всем синаптическим весам wi,j и
      #нейронным смещениям w0,j (i=0,…,n; j=1,…,k) присваиваются некоторые малые
      #случайные значения
        self.learning_rate = learning_rate #задан коэффициент скорости обучения
        self.n_iterations = n_iterations #задано количество итераций
        self.weights = None #веса модели
        self.bias = None  #смещение
    def activation_function(self, x):
    #Функция активации сигмоида - преобразует числа в диапазон от 0 до 1, чтобы предсказывать вероятность принадлежности к классу
    #Нужно, чтобы преобразовать линейную комбинацию входных данных в вероятность принадлежности к классу 1 (доброкачественная опухоль)
        return 1 / (1 + np.exp(-x))
    def predict(self, X):
      #Считаем предсказания модели, вычисляем взвешенную сумму входных сигналов:
      #Умножаем входные данные на веса и добавляем смещение
      #Применяем функцию активации, чтобы получить вероятности
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)
    def train(self, X, y):
        #Инициализация весов и смещения
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  #инициализация весов нулями
        self.bias = 0  #инициализация смещения нулем
        #Обучение с использованием градиентного спуска
        for _ in range(self.n_iterations):
            #Вычисление предсказаний модели
            model_output = self.predict(X)
            #Вычисление ошибки (градиентов) для обновления весов
            dw = (1 / n_samples) * np.dot(X.T, (model_output - y))  #градиент по весам
            db = (1 / n_samples) * np.sum(model_output - y)  #градиент по смещению
            #Обновление весов и смещения
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def binary_predict(self, X):
        #Преобразует вероятности в классы (0 или 1)
        #Если вероятность больше 0.5, то это класс 1 (доброкач.), иначе — класс 0 (злокач.)
        predicted_probs = self.predict(X)
        return [1 if prob > 0.5 else 0 for prob in predicted_probs]
#Загружаем данные
def load_data():
    #Загрузка обучающей выборки
    data = load_breast_cancer()  #Загрузка датасета Breast Cancer
    X = data.data  #Признаки (входные данные)
    y = data.target  #Метки классов (целевая переменная). 
    #Целевая переменная y в датасете принимает значения 0 или 1, где:
    #0 соответствует злокачественной опухоли
    #1 соответствует доброкачественной опухоли
    #разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #приведение к нормальному распределению
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  #обучение масштабирования на обучающей выборке
    X_test = scaler.transform(X_test)  #применение масштабирования к тестовой выборке
    return X_train, X_test, y_train, y_test
#Функция для вычисления точности модели
def calculate_accuracy(y_true, y_pred):
    #Подсчет доли правильных предсказаний
     return np.mean(y_true == y_pred)
#Применение
if __name__ == '__main__':
    #Шаг 1. Загрузка данных - данные делятся на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = load_data()
    #Шаг 2. Создание экземпляра персептрона с заданной скоростью обучения и кол-вом эпох
    ppn = OneLayerPerceptron(learning_rate=0.01, n_iterations=50000)
    #Шаг 3-8. Обучение персептрона на обучающей выборке
    ppn.train(X_train, y_train)
    #После обучения модель предсказывает метки для тестового набора данных, и мы оцениваем точность предсказаний.
    #Шаг 9. Подсчет предсказаний на тестовой выборке
    predictions = ppn.binary_predict(X_test)
    print("Предсказания:", predictions)
    #Вычисление точности модели
    accuracy = calculate_accuracy(y_test, predictions)
    print(f'Точность: {accuracy * 100:.2f}%')

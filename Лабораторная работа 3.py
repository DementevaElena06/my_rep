import tensorflow as tf 
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import RandomUniform
tf.compat.v1.enable_eager_execution()
#загружаем данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Изображения имеют размер 28x28 и, следовательно, является
#двухмерными. Поскольку наш персептрон способен считывать только
#одномерные данные, преобразуем их.
#Преобразуем изображения в одномерные массивы и нормализуем значения пикселей
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
#Преобразуем метки классов в one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#Разделение обучающей выборки на train (54k) и validation (6k)
x_train, x_val = x_train[:54000], x_train[54000:]
y_train, y_val = y_train[:54000], y_train[54000:]
#Опишем архитектуру сети:
#Реализация нейронной сети с архитектурой 2-4-2
#Создаем модель
model = Sequential()
#Первый слой: 4 нейрона, входной размер 784 (размер изображения 28x28)
#Инициализация весов из интервала [-0.3, 0.3]
model.add(Dense(4, input_dim=784, activation='sigmoid',
                kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)))
#второй слой: 2 нейрона (выходной слой)
model.add(Dense(2, activation='sigmoid',
                kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)))
#Компиляция модели
learning_rate = 0.35
optimizer = SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
#Обучение модели
#Поскольку MNIST имеет 10 классов, а архитектура 2-4-2 предполагает 2 выхода, мы будем использовать только первые два класса (цифры 0 и 1) для упрощения задачи
#Фильтруем данные для классов 0 и 1
train_filter = np.where((y_train[:, 0] == 1) | (y_train[:, 1] == 1))
val_filter = np.where((y_val[:, 0] == 1) | (y_val[:, 1] == 1))
test_filter = np.where((y_test[:, 0] == 1) | (y_test[:, 1] == 1))
x_train_filtered, y_train_filtered = x_train[train_filter], y_train[train_filter][:, :2]
x_val_filtered, y_val_filtered = x_val[val_filter], y_val[val_filter][:, :2]
x_test_filtered, y_test_filtered = x_test[test_filter], y_test[test_filter][:, :2]
#Обучаем модель
history = model.fit(
    x_train_filtered, y_train_filtered,
    validation_data=(x_val_filtered, y_val_filtered),
    epochs=10,
    batch_size=32
)
#Оцениваем точность на тестовой выборке
test_loss, test_accuracy = model.evaluate(x_test_filtered, y_test_filtered)
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
#Анализ результатов
#Выводим результаты обучения
#График ошибки
plt.plot(history.history['loss'], label='Ошибка на обучении')
plt.plot(history.history['val_loss'], label='Ошибка на проверке')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.legend()
plt.show()
#график точности
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на проверке')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()
#Добавление условия для входного вектора X = {0.1, -0.1} и эталонного выхода Y = {0.5, -0.5}
#Создаем отдельную модель с архитектурой 2-4-2 для работы с этим условием
model_custom = Sequential()
#Первый слой: 4 нейрона, входной размер 2
model_custom.add(Dense(4, input_dim=2, activation='sigmoid',
                       kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)))
#Второй слой: 2 нейрона (выходной слой)
model_custom.add(Dense(2, activation='sigmoid',
                       kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)))
#Компиляция модели
model_custom.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
#Входной вектор X и эталонный выход Y
X = np.array([[0.1, -0.1]])  # Входной вектор
Y = np.array([[0.5, -0.5]])  # Эталонный выход
#Обучение модели на одном примере
history_custom = model_custom.fit(
    X, Y,
    epochs=100,  #Увеличиваем количество эпох для лучшей сходимости
    batch_size=1,  #Поскольку у нас один пример, batch_size = 1
    verbose=1  #Выводим информацию об обучении
)
#Оценка модели
loss_custom, accuracy_custom = model_custom.evaluate(X, Y, verbose=0)
print(f"Ошибка на обучающем примере: {loss_custom:.4f}")
print(f"Точность на обучающем примере: {accuracy_custom:.4f}")
#Анализ результатов
#График ошибки
plt.plot(history_custom.history['loss'], label='Ошибка на обучении (Custom)')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

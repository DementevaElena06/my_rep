import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#Загружаем данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Поскольку MNIST имеет 10 классов, а архитектура 2-4-2 предполагает 2 выхода, мы будем использовать только первые два класса (цифры 0 и 1) для упрощения задачи
#Фильтруем данные для классов 0 и 1
train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]
#Изображения имеют размер 28x28 и, следовательно, является
#двухмерными. Поскольку наш персептрон способен считывать только
#одномерные данные, преобразуем их.
#Преобразуем изображения в одномерные массивы и нормализуем значения пикселей
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
#Уменьшаем размерность входных данных до 2 с помощью PCA
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
#Преобразуем метки классов в one-hot encoding
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
#Опишем архитектуру сети:
#Реализация нейронной сети с архитектурой 2-4-2
#Создаем модель
model = Sequential()
#Первый слой: 4 нейрона, входной размер 784 (размер изображения 28x28)
#Инициализация весов из интервала [-0.3, 0.3]
model.add(Dense(4, input_dim=2, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.3, maxval=0.3)))
#второй слой: 2 нейрона (выходной слой)
model.add(Dense(2, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.3, maxval=0.3)))
#Компиляция модели
optimizer = SGD(learning_rate=0.35)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
#Обучение модели
history = model.fit(
    x_train_pca, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)
#Оценка модели на тесте
test_loss, test_accuracy = model.evaluate(x_test_pca, y_test)
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
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
#Входной вектор X и эталонный выход Y
X = np.array([[0.1, -0.1]], dtype=np.float32)  # Входной вектор
Y = np.array([[0.5, -0.5]], dtype=np.float32)  # Эталонный выход
#Предсказание для пользовательского входа
prediction = model.predict(X)
print(f"Предсказание для входа {X}: {prediction}")
#Оценка модели на пользовательском входе
loss_custom = tf.reduce_mean(tf.square(prediction - Y)).numpy()
print(f"Ошибка на пользовательском примере: {loss_custom:.4f}")

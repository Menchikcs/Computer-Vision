import pandas as pd #робота з CSV таблицями
import numpy as np #математичні операції
import tensorflow as tf #створює нейронку
from tensorflow import keras #працює з шарами нейронки, частина
from tensorflow.keras import layers #створення шарів
from sklearn.preprocessing import LabelEncoder #перетворює текстові вітки в числа
import matplotlib.pyplot as plt #для побудови графіків

#2 - зчитали файл
df = pd.read_csv('data/figures2.csv')
# print(df.head())

#3 - перетворюємо назви елементів у числа
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label']) #fit_transform перетворює назви на числа

#4 вибираємо стовпці для навчання
X = df[['area', 'perimeter', 'corners', 'area_to_perimeter_ratio']]
y = df['label_enc']

#5 - створення моделі
model = keras.Sequential([layers.Input(shape = (4,)),
                          layers.Dense(16, activation = 'relu'),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(16, activation='softmax'),
])

#6 компіляція моделі
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#1 - підбирає який краще використати алгоритм, 2 - функція втрат, 3 - точність

#7 навчання
history = model.fit(X, y, epochs = 500, verbose = 0)

#8 візуалізація
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Learning process')
plt.legend()
plt.show()

#9 тестування
text = np.array([25, 20, 0])
pred = model.predict(text)
print(f'Probability of label: {pred}')
print(f'Model prediction: {encoder.inverse_transform([np.argmax(pred)])}')
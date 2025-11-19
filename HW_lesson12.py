import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

#завантажуємо файл
train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                            image_size=(128, 128), batch_size=30, label_mode='categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                            image_size=(128, 128), batch_size=30, label_mode='categorical')

#нормалізація зображення
normalization =layers.Rescaling(1./255) #формат бібліотеки
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

#будуємо модель
model = models.Sequential()

model.add(layers.Conv2D(32,                     # кількість фільтрів
                        (3, 3),             # розмір фільтра
                        activation='relu',            # функція активації
                        input_shape=(128, 128, 3)))   # форма вхідного зображення (RGB)
model.add(layers.MaxPooling2D((2,2))) #зменшуємо карту ознак у 2 рази

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

#компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#навчання моделі
history = model.fit(train_ds, epochs=10, validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)

#перевірка
class_name = ['apple', 'banana', 'orange']

img = image.load_img('images/oranges.png', target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array/255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
predict_index = np.argmax(prediction[0])

print(f'Probability by classes: {prediction[0]}')
print(f'Model prediction: {class_name[predict_index]}')
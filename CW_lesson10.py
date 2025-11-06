import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#1 - створюємо функцію для генерації простих фігур
def generate_img(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == 'circle':
        cv.circle(img, (100,100), 50, color, -1)
    elif shape == 'square':
        cv.rectangle(img, (50,50), (150,150), color, -1)
    elif shape == 'triangle':
        points = np.array([[100,40], [40,160], [160,100]])
        cv.drawContours(img, [points], 0, color, -1)
    return img

#2 - формуємо набори даних
X = [] #список ознак
y = [] #список міток

colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0)
}
shapes = ['circle', 'square', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_img(bgr, shape)
            mean_color = cv.mean(img)[:3] #(b, g, r, alpha)
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features)
            y.append(f'{color_name}_{shape}')

#3 - розділяємо дані 70 на 30б 70 - для навчання, 30 - для перебірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
#X_train - ознаки для навчання, X_test - ознаки для перевірки, y_train - правильні відповіді для навчання, y_test - правильні відповідв для перевірки

#4 - навчаємо модель
model = KNeighborsClassifier(n_neighbors=3) #бажано ставити непарні числа
model.fit(X_train, y_train)

#5 - перевіпряємо точність

accuracy = model.score(X_test, y_test)
print(f'Models accuracy: {round(accuracy * 100, 2)}%')

#6 - тестуємо зображення
test_img = generate_img((0,0,255), 'square')
mean_color = cv.mean(test_img)[:3]
prediction = model.predict([mean_color])
print(f'Prediction: {prediction[0]}')

cv.imshow('img', test_img)
cv.waitKey(0)
cv.destroyAllWindows()
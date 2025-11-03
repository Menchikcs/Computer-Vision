import cv2 as cv
net = cv.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt',
                              'data/MobileNet/mobilenet.caffemodel')
classes = []
with open("data/MobileNet/synset.txt", 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

img = cv.imread('images/MobileNet/cat.jpg')

#4 крок - адаптуємо зображення під нейронку
blob = cv.dnn.blobFromImage(cv.resize(img, (224, 224)),1.0 / 127.5,(224, 224),(127.5, 127.5, 127.5))

#5 крок - кладемо зображення в мережу
net.setInput(blob)
preds = net.forward()

#6 крок - знаходимо індекс класу з найбільшою імовірністю
idx = preds[0].argmax()

#7 крок - дістаємо назву класу і впевненість
label = classes[idx] if idx < len(classes) else 'unknown'
conf = preds[0][idx].item() * 100

#8 крок - виводимо результат в консоль
print('Class:', label)
print('Probability:', conf)

#9 крок - підписуємо зображення
text = f'{label}: {int(conf)}%'
cv.putText(img, text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
cv.imshow('MobileNet', img)
cv.waitKey(0)
cv.destroyAllWindows()
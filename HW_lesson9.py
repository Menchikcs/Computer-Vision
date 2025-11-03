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
img1 = cv.imread('images/MobileNet/dog.jpg')
img2 = cv.imread('images/MobileNet/Parrot.jpg')
img2 = cv.resize(img2, (img2.shape[1]//3,img2.shape[0]//3))

#4 крок - адаптуємо зображення під нейронку
blob = cv.dnn.blobFromImage(cv.resize(img, (224, 224)), 1.0 / 127.5,(224, 224),(127.5, 127.5, 127.5))
blob1 = cv.dnn.blobFromImage(cv.resize(img1, (224, 224)), 1.0 / 127.5,(224, 224),(127.5, 127.5, 127.5))
blob2 = cv.dnn.blobFromImage(cv.resize(img2, (224, 224)), 1.0 / 127.5,(224, 224),(127.5, 127.5, 127.5))

#5 крок - кладемо зображення в мережу
net.setInput(blob)
preds = net.forward()
net.setInput(blob1)
preds1 = net.forward()
net.setInput(blob2)
preds2 = net.forward()

#6 крок - знаходимо індекс класу з найбільшою імовірністю
idx = preds[0].argmax()
idx1 = preds1[0].argmax()
idx2 = preds2[0].argmax()

#7 крок - дістаємо назву класу і впевненість
label = classes[idx] if idx < len(classes) else 'unknown'
conf = preds[0][idx].item() * 100
label1 = classes[idx1] if idx1 < len(classes) else 'unknown'
conf1 = preds1[0][idx1].item() * 100
label2 = classes[idx2] if idx2 < len(classes) else 'unknown'
conf2 = preds2[0][idx2].item() * 100

#8 крок - виводимо результат в консоль
print('Class:', label)
print('Probability:', conf)
print('Class:', label1)
print('Probability:', conf1)
print('Class:', label2)
print('Probability:', conf2)

#9 крок - підписуємо зображення
text = f'{label}: {int(conf)}%'
cv.putText(img, text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
text1 = f'{label1}: {int(conf1)}%'
cv.putText(img1, text1, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
text2 = f'{label2}: {int(conf2)}%'
cv.putText(img2, text2, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
cv.imshow('Cat', img)
cv.imshow('Dog', img1)
cv.imshow('Parrot', img2)
cv.waitKey(0)
cv.destroyAllWindows()
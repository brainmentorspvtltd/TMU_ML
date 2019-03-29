import cv2
import numpy as np

f_01 = np.load('face_1.npy').reshape((100, 50 * 50 * 3))  # face_1
f_02 = np.load('face_1.npy').reshape((100, 50 * 50 * 3))  # face_2

names = {
    0: 'ravi',
    1: 'unknown',
}

labels = np.zeros((200, 1))
labels[:100, :] = 0.0  # first 100 for user_1 (0)
labels[100:, :] = 1.0  # next 100 for user_2 (1)

# combine all info into one data array
data = np.concatenate([f_01, f_02])  # (200, 7500)

def distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum())

def knn(x, train, k=5):
    m = train.shape[0]
    dist = []
    for i in range(m):
        dist.append(distance(x, train[i]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    # print("Index...",indx)
    sorted_labels = labels[indx][:k]
    # print("Sorted...",sorted_labels)
    counts = np.unique(sorted_labels, return_counts=True)
    # print("Count...",counts)
    return counts[0][np.argmax(counts[1])]

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    ret, frame = capture.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_component = frame[y:y + h, x:x + w, :]
            fc = cv2.resize(face_component, (50, 50))
            # print(fc.shape) - 50x50x3
            lab = knn(fc.flatten(), data)
            text = names[int(lab)]
            cv2.putText(frame, text, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('face recognition', frame)
        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break
    else:
        print('Error')

capture.release()
cv2.destroyAllWindows()


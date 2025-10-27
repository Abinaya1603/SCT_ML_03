
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


CAT_DIR = "dataset/cats"  
DOG_DIR = "dataset/dogs" 

def load_images(folder, label, img_size=(64, 64), limit=None):
    data = []
    files = os.listdir(folder)
    if limit:
        files = files[:limit]
    for file in files:
        try:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append((img_gray.flatten(), label))
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return data

cat_data = load_images(CAT_DIR, 0, limit=1000)
dog_data = load_images(DOG_DIR, 1, limit=1000)

dataset = cat_data + dog_data
np.random.shuffle(dataset)

X = np.array([i[0] for i in dataset])
y = np.array([i[1] for i in dataset])

print(f"Dataset Loaded: {len(X)} images")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Training SVM classifier... (this may take a few minutes)")

svm = SVC(kernel='linear')  # you can also try 'rbf'
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
for i in range(6):
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx].reshape(64, 64)
    label = "Dog" if y_pred[idx] == 1 else "Cat"
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

import cv2
import os
import numpy as np

dataset_path = r"D:\opencv\data\face_dataset"
trainer_path = r"D:\opencv\data\trainer.yml"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

for i, file_name in enumerate(os.listdir(dataset_path)):
    img_path = os.path.join(dataset_path, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    faces.append(img)
    labels.append(1)  # Label '1' for you

recognizer.train(faces, np.array(labels))
recognizer.save(trainer_path)
print("Training completed! Trainer saved at D:\\opencv\\data\\trainer.yml")

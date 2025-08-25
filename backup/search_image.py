# thu vien
import math
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np

# ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights = "imagenet")
    extract_model = Model(inputs =vgg16_model.inputs,outputs=vgg16_model.get_layer("fc1").output )
    return extract_model

# ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("xu ly: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # trich dac trung
    vector = model.predict(img_tensor)[0]
    # chuan hoa vector = chia cho L2 norm
    vector = vector / np.linalg.norm(vector)
    return vector

# dinh nghia anh can tim kiem
search_image = "testimage/fox.jpeg"

# khoi tao model
model = get_extract_model()

# trich xuat dac trung anh muon tim kiem
search_vector = extract_vector(model, search_image)

# load 4700 vector tu vectors.pkl ra bien
vectors = pickle.load(open("vectors.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))

# tinh khoang cach tu search_vector den tat ca cac vector
distance = np.linalg.norm(vectors - search_vector, axis = 1)

# sap xep va lay ra K vector co khoang cach ngan nhat
K = 16
ids = np.argsort(distance)[:K]

# tao output
nearest_image = [(paths[id], distance[id]) for id in ids]

# ve len man hinh cac anh gan nhat do
import matplotlib.pyplot as plt

axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize = (10, 5))

for id in range(K):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id + 1))

    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()
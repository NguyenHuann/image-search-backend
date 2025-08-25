# tao model
from tensorflow.keras.applications.efficientnet import EfficientNetB0
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# tien xu li anh
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def image_preprocessing(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # chuan hoa theo efficientnet
    return x

# trich xuat dac trung
from PIL import Image

def extract_vector(model, image_path):
    img = Image.open(image_path)
    tensor = image_preprocessing(img)
    vector = model.predict(tensor)[0]
    return vector / np.linalg.norm(vector) # chuan hoa L2

# luu vector va duong dan anh
import os
import pickle

vectors = []
paths = []
dataset_dir = 'dataset'

for file in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, file)
    try:
        vector = extract_vector(model, path)
        vectors.append(vector)
        paths.append(path)
    except Exception as e:
        print('lỗi xử lý ảnh:', path, '|', str(e))
# luu lai de dung sau
pickle.dump(vectors, open('vectors.pkl', 'wb'))
pickle.dump(paths, open('paths.pkl', 'wb'))
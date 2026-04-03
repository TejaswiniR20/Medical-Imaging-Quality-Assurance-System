import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# =========================
# LOAD TRAINED MODEL
# =========================
model = tf.keras.models.load_model("mobilenetv3_best_model.keras")

# =========================
# YOUR 5 TEST IMAGES
# =========================
image_paths = [

r"D:\Medical_Project\code\Medical-Imaging-Quality-Assurance-System-preprocessing\data\clahe_balanced\test\PNEUMONIA\00000261_002.png",

r"D:\Medical_Project\code\Medical-Imaging-Quality-Assurance-System-preprocessing\data\clahe_balanced\test\PNEUMONIA\00011553_045.png",

r"D:\Medical_Project\code\Medical-Imaging-Quality-Assurance-System-preprocessing\data\clahe_balanced\test\PNEUMONIA\00017747_039.png",

r"D:\Medical_Project\code\Medical-Imaging-Quality-Assurance-System-preprocessing\data\clahe_balanced\test\NORMAL\00000305_002.png",

r"D:\Medical_Project\code\Medical-Imaging-Quality-Assurance-System-preprocessing\data\clahe_balanced\test\NORMAL\00000516_000.png"

]

# =========================
# CLASS LABELS
# =========================
class_names = ["NORMAL", "PNEUMONIA"]

# =========================
# PREPROCESS IMAGES
# =========================
processed_images = []

for path in image_paths:

    img = image.load_img(path, target_size=(224,224))

    img_array = image.img_to_array(img)

    img_array = preprocess_input(img_array)

    processed_images.append(img_array)

processed_images = np.array(processed_images)

# =========================
# PREDICT
# =========================
predictions = model.predict(processed_images)

# =========================
# SHOW RESULTS
# =========================
for i, pred in enumerate(predictions):

    probability = pred[0]

    label = class_names[int(probability > 0.5)]

    confidence = probability if label=="PNEUMONIA" else 1-probability

    print("\nImage:", image_paths[i])

    print("Prediction:", label)

    print("Confidence:", round(float(confidence),3))
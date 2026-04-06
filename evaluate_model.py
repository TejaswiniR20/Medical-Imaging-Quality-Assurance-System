import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("mobilenetv2_best_model.keras")

# ======================
# TEST DATA
# ======================
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(

    r"data/clahe_balanced/test",

    target_size=(224,224),

    batch_size=32,

    class_mode="binary",

    shuffle=False,

    color_mode="rgb"

)

# ======================
# EVALUATE
# ======================
loss, accuracy = model.evaluate(test_generator)

print("\nBalanced MobileNetV2 Test Accuracy:", round(accuracy*100,2), "%")
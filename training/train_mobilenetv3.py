import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight

# =====================
# SETTINGS
# =====================
IMG_SIZE = 224
BATCH_SIZE = 32

EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 35   # keeps training under 2 hours

BASE_PATH = r"D:\Medical_Project\code\Medical-Imaging-Quality-Assurance-System-preprocessing\data\clahe_balanced"

# =====================
# DATA
# =====================
train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=30,
    zoom_range=0.3,

    width_shift_range=0.15,
    height_shift_range=0.15,

    shear_range=0.2,

    brightness_range=[0.75,1.25],

    horizontal_flip=True,

    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(

    BASE_PATH + r"\train",

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True
)

val_data = val_datagen.flow_from_directory(

    BASE_PATH + r"\val",

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)

# =====================
# CLASS WEIGHTS
# =====================
weights = class_weight.compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_data.classes),

    y=train_data.classes
)

cw_dict = dict(enumerate(weights))

# =====================
# MODEL
# =====================
base_model = MobileNetV3Large(

    weights="imagenet",

    include_top=False,

    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers:
    layer.trainable = False

# classifier
x = GlobalAveragePooling2D()(base_model.output)

x = BatchNormalization()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(base_model.input, output)

# =====================
# CALLBACKS
# =====================
callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=6,

        restore_best_weights=True
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-7,

        verbose=1
    ),

    ModelCheckpoint(

        "mobilenetv3_best_model.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    )
]

# =====================
# PHASE 1
# =====================
model.compile(

    optimizer=Adam(learning_rate=1e-4),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

print("\nPHASE 1")

model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS_PHASE1,

    class_weight=cw_dict,

    callbacks=callbacks
)

# =====================
# PHASE 2 (STRONG FINE TUNE)
# =====================
print("\nFine tuning last 150 layers")

for layer in base_model.layers[-150:]:
    layer.trainable = True

model.compile(

    optimizer=Adam(learning_rate=1e-6),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

print("\nPHASE 2")

model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS_PHASE2,

    class_weight=cw_dict,

    callbacks=callbacks
)

# =====================
# SAVE FINAL MODEL
# =====================
model.save("mobilenetv3_medical_model.keras")

print("\nDONE")
print("Saved:")
print("mobilenetv3_best_model.keras")
print("mobilenetv3_medical_model.keras")
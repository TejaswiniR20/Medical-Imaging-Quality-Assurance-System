import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ======================
# PATHS
# ======================
train_dir = "data/clahe_Result/train"
val_dir   = "data/clahe_Result/val"

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# ======================
# STRONG AUGMENTATION
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=15,
    zoom_range=0.15,

    width_shift_range=0.05,
    height_shift_range=0.05,

    horizontal_flip=True,

    brightness_range=[0.9,1.1]
)

val_datagen = ImageDataGenerator(rescale=1./255)

# ======================
# LOAD DATA
# ======================
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="rgb"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="rgb"
)

# ======================
# CLASS WEIGHTS
# ======================
labels = train_generator.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weight_dict = dict(enumerate(class_weights))

print(class_weight_dict)

# ======================
# BASE MODEL
# ======================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

# ======================
# CLASSIFIER HEAD
# ======================
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ======================
# COMPILE
# ======================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# CALLBACKS
# ======================
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_pneumonia_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ======================
# TRAIN HEAD
# ======================
history = model.fit(
    train_generator,
    validation_data=val_generator,

    epochs=20,

    class_weight=class_weight_dict,

    callbacks=[lr_scheduler, early_stop, checkpoint]
)

# ======================
# FINE TUNE
# ======================
print("Fine tuning started")

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,

    epochs=10,

    class_weight=class_weight_dict,

    callbacks=[lr_scheduler, early_stop, checkpoint]
)

# ======================
# SAVE FINAL MODEL
# ======================
model.save("best_mobilenetv2_model.keras")

print("Training complete")
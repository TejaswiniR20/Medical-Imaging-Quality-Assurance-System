import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# =========================
# 1. PATHS (UPDATED)
# =========================
train_dir = "data/clahe_balanced/train"
val_dir   = "data/clahe_balanced/val"

IMG_SIZE = (224,224)
BATCH_SIZE = 32

EPOCHS_PHASE1 = 18
EPOCHS_PHASE2 = 12

# =========================
# 2. AUGMENTATION (STRONGER)
# =========================
train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,
    zoom_range=0.2,

    width_shift_range=0.1,
    height_shift_range=0.1,

    shear_range=0.15,

    brightness_range=[0.85,1.15],

    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# =========================
# 3. LOAD DATA
# =========================
train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True
)

val_generator = val_datagen.flow_from_directory(

    val_dir,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)

print("Class indices:", train_generator.class_indices)

# =========================
# 4. CLASS WEIGHTS
# =========================
labels = train_generator.classes

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(labels),

    y=labels
)

class_weight_dict = dict(enumerate(weights))

print("Class weights:", class_weight_dict)

# =========================
# 5. MODEL
# =========================
base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)
)

# freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

# custom head
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# 6. COMPILE
# =========================
model.compile(

    optimizer=Adam(learning_rate=1e-4),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

model.summary()

# =========================
# 7. CALLBACKS
# =========================
lr_scheduler = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=3,

    min_lr=1e-6,

    verbose=1
)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True
)

checkpoint = ModelCheckpoint(

    "mobilenetv2_best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)

# =========================
# 8. TRAIN PHASE 1
# =========================
print("\nPHASE 1 TRAINING")

model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS_PHASE1,

    class_weight=class_weight_dict,

    callbacks=[lr_scheduler, early_stop, checkpoint]
)

# =========================
# 9. FINE TUNE (STRONG)
# =========================
print("\nFINE TUNING STARTED")

# unfreeze last 50 layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(

    optimizer=Adam(learning_rate=1e-5),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS_PHASE2,

    class_weight=class_weight_dict,

    callbacks=[lr_scheduler, early_stop, checkpoint]
)

# =========================
# 10. SAVE FINAL MODEL
# =========================
model.save("best_mobilenetv2_balanced_model.keras")

print("\nTraining completed")
print("Saved files:")
print("best_mobilenetv2_balanced_model.keras")
print("mobilenetv2_medical_final.keras")
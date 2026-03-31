import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# =========================
# 1. PATHS
# =========================
train_dir = "data/clahe_Result/train"
val_dir   = "data/clahe_Result/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# =========================
# 2. AUGMENTATION (IMPROVED)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# =========================
# 3. DATA LOADING
# =========================
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("Class indices:", train_generator.class_indices)

# =========================
# 4. CLASS WEIGHTS (IMPORTANT FIX)
# =========================
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# =========================
# 5. MOBILE NET MODEL (TRANSFER LEARNING)
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# 6. COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 7. CALLBACKS
# =========================
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_pneumonia_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# =========================
# 8. TRAINING (PHASE 1)
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler, early_stop, checkpoint]
)

# =========================
# 9. FINE TUNING (PHASE 2)
# =========================
print("🔧 Starting Fine Tuning...")

base_model.trainable = True

# freeze first layers, train last layers only
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler, early_stop, checkpoint]
)

# =========================
# 10. SAVE FINAL MODEL
# =========================
model.save("final_pneumonia_model.keras")
print("✅ Model saved successfully!")
# train_resnet_fer.py
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from models_resnet_fer import build_resnet  # ✅ 从刚才那个文件导入

train_dir = 'data/train'
val_dir = 'data/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_model = build_resnet(input_shape=(48,48,1), num_classes=7)

emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4, decay=1e-6),
    metrics=['accuracy']
)

emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64
)

emotion_model.save_weights('emotion_model_resnet.h5')

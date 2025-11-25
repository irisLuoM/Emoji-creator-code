# webcam_demo.py
import cv2
import numpy as np
from keras.optimizers import Adam
from models_resnet_fer import build_resnet

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# 构建同样结构的模型，然后加载训练好的权重
emotion_model = build_resnet(input_shape=(48,48,1), num_classes=7)
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4, decay=1e-6),
    metrics=['accuracy']
)
emotion_model.load_weights('emotion_model_resnet.h5')  # ✅ 用训练脚本存的权重

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = cropped_img.astype("float32") / 255.0
        cropped_img = np.expand_dims(cropped_img, axis=-1)   # (48,48,1)
        cropped_img = np.expand_dims(cropped_img, axis=0)    # (1,48,48,1)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        cv2.putText(
            frame,
            emotion_dict[maxindex],
            (x+20, y-60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

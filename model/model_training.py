from keras.callbacks import ReduceLROnPlateau
from model_definition import create_model
import numpy as np

# 데이터 로드
X_train = np.load('train_data.npy')
y_train = np.load('train_labels.npy')

# 모델 정의
model = create_model()

# ReduceLROnPlateau 콜백 정의
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    X_train, y_train,
    batch_size=8,
    epochs=50,
    validation_split=0.2,
    callbacks=[reduce_lr]
)

# 학습 모델 저장
model.save('model5.h5')
print("모델 학습 및 저장 완료.")

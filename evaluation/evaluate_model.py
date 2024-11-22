from keras.models import load_model
from sklearn.metrics import f1_score
import numpy as np

# 모델 로드
model = load_model('model5.h5')

# 테스트 데이터 로드
X_test = np.load('test_data.npy')
y_test = np.load('test_labels.npy')

# 테스트 데이터에 대한 예측
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

# F1-score 계산
f1 = f1_score(y_test, binary_predictions)
print(f"F1 Score: {f1:.2f}")

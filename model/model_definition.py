from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling3D

def create_model(input_shape=(64, 128, 128, 1)):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling3D())
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

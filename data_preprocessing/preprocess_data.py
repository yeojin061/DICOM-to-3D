import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from tqdm.notebook import tqdm

# 데이터 디렉토리와 파일명 설정
data1_folder = '/content/drive/MyDrive/Colab Notebooks/LUNA16/subset0/data1'
data0_folder = '/content/drive/MyDrive/Colab Notebooks/LUNA16/subset0/data0'

# 양성과 음성 데이터의 mhd 파일 경로를 가져오기
def get_file_paths(folder, extension):
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(extension)]

data1_mhd_paths = get_file_paths(data1_folder, '.mhd')
data0_mhd_paths = get_file_paths(data0_folder, '.mhd')

# 데이터 전처리 함수 정의
def normalize(volume):
    """체적 데이터 정규화"""
    min_value = -1000
    max_value = 400
    volume[volume < min_value] = min_value
    volume[volume > max_value] = max_value
    volume = (volume - min_value) / (max_value - min_value)
    volume = volume.astype("float32")
    return volume

def resize_volume(img, target_shape):
    """체적 크기 조절"""
    current_shape = img.shape
    depth = current_shape[0] / target_shape[0]
    width = current_shape[1] / target_shape[1]
    height = current_shape[2] / target_shape[2]
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img

def preprocess_data(mhd_paths):
    processed_data = []
    file_names = []
    # 원하는 크기 설정
    desired_shape = (64, 128, 128)

    for mhd_path in tqdm(mhd_paths, desc="스캔 처리 중"):
        # 읽어온 mhd 파일 전처리
        ct_scan = sitk.ReadImage(mhd_path)
        ct_array = sitk.GetArrayFromImage(ct_scan)
        ct_array = normalize(ct_array)
        # 비율 계산
        ct_array = resize_volume(ct_array, desired_shape)
        processed_data.append(ct_array)
        file_names.append(os.path.basename(mhd_path))  # 파일명 저장

    return processed_data, file_names

# 전처리된 데이터 및 파일명 얻기
processed_data0, file_names0 = preprocess_data(data0_mhd_paths)
processed_data1, file_names1 = preprocess_data(data1_mhd_paths)

# 데이터와 파일명 저장
save_path_data0 = '/content/drive/MyDrive/Colab Notebooks/LUNA16/subset0/data0mhd.npy'
save_path_data1 = '/content/drive/MyDrive/Colab Notebooks/LUNA16/subset0/data1mhd.npy'

np.save(save_path_data0, [processed_data0, file_names0])
np.save(save_path_data1, [processed_data1, file_names1])

print("데이터 처리 및 저장 완료.")

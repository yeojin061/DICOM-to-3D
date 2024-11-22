# DICOM-to-3D
인체를 3D화하여 인체의 구조 및 기능, 상호작용 등을 원할하게 이해할 수 있도록 돕는 시스템

## Overview
This repository contains the code I personally contributed to during the development of the 3D Medical Imaging Education Simulation project.  
본 리포지토리는 '3D 의료영상 교육 시뮬레이션' 프로젝트에서 제가 기여한 코드만 포함하고 있습니다.

The project focuses on using LUNA16 dataset-based CT images to train models for detecting lung nodules, with results prepared for visualization in a 3D simulation environment.  
이 프로젝트는 LUNA16 데이터셋 기반 폐 CT 이미지를 활용하여 폐 종양을 탐지하는 모델을 학습하고, 3D 시뮬레이션 환경에서 결과를 시각화하는 것을 목표로 합니다.

## My Contributions
1. Preprocessing LUNA16 CT scan data for training.
   LUNA16 CT 데이터를 학습에 적합하도록 전처리
2. Training and fine-tuning a TensorFlow-based 3D convolutional neural network (CNN).
   TensorFlow 기반의 3D CNN 모델 학습 및 튜닝
3. Preparing data for Unity integration and verifying predictions in the simulation environment.
   Unity 통합을 위한 데이터 준비 및 시뮬레이션 환경에서 예측 결과 검증

## Notes
This repository includes only my contributions. Full project implementation, including PACS/DICOM integration and Unity scripting, is not included here.  
이 리포지토리에는 제 기여만 포함되어 있습니다. PACS/DICOM 통합 및 Unity 스크립팅을 포함한 전체 프로젝트 구현은 여기에 포함되지 않습니다.  
Due to data dependency issues, the code is not executable as is.  
데이터 의존성 문제로 인해 현재 상태로는 실행이 불가능합니다.

# Project Structure

## Folder Structure
- data_preprocessing/preprocess_data.py: Code for loading and preprocessing LUNA16 CT scan data.  
  LUNA16 CT 스캔 데이터를 로드하고 전처리하는 코드를 포함합니다.
- model/model_definition.py: Code for defining the TensorFlow-based 3D CNN model.  
  TensorFlow 기반 3D CNN 모델을 정의하는 코드를 포함합니다.
- model/model_training.py: Code for training the 3D CNN model.  
  3D CNN 모델을 학습시키는 코드를 포함합니다.
- evaluation/evaluate_model.py: Code for evaluating the trained model.  
  학습된 모델을 평가하는 코드를 포함합니다.

## How to Use
1. Preprocess the LUNA16 CT scan data using data_preprocessing/preprocess_data.py.  
   해당 코드를 사용하여 CT 데이터를 전처리합니다.
2. Define the model using `model/model_definition.py`.  
   해당 코드를 사용하여 모델을 정의합니다.
3. Train the model using `model/model_training.py`.  
   해당 코드를 사용하여 모델을 학습합니다.
4. Evaluate the trained model using `evaluation/evaluate_model.py`.  
   해당 코드를 사용하여 학습된 모델을 평가합니다.

# RIST_LIDAR_2024 (4INLAB)
=======
<h1 align="center">3D 포인트 클라우드 데이터 분류를 위한 PointNet 시스템</h1>

<p align="center"><b> ▶️ PointNet 구현과 함께 클래스 불균형 해결, 모델 학습 및 검증 기능 제공 ◀️ </b></p>  

# 목차
- [개요](#개요)
- [주요 구성 요소](#주요-구성-요소)
- [사용 방법](#사용-방법)
  - [설치](#설치)
  - [기본 사용법](#기본-사용법)
  - [사용자 정의](#사용자-정의)
- [향후 개선 사항](#향후-개선-사항)
- [참고 자료](#참고-자료)

# 개요
이 저장소는 3D 포인트 클라우드 데이터를 분류하기 위한 PointNet 구현을 포함하고 있습니다. 전체 프로세스는 Docker를 기반으로 컨테이너화되어 있으며, 데이터 불균형 해결 및 학습/검증 손실과 정확도 시각화를 지원합니다. PointNet의 주요 기능은 CSV 데이터 처리, Oversampling, 학습된 모델 저장 및 최적화된 검증을 포함합니다.

# 주요 구성 요소
- **`PointNetDataset`**: 포인트 클라우드 데이터와 라벨을 로드하고, Oversampling을 통해 클래스 불균형 문제를 해결.
- **모델 정의 (`PointNet`)**:
  - Convolution 레이어로 특징 추출.
  - Fully Connected 레이어로 최종 분류 수행.
  - Adaptive Max Pooling을 통해 전역 특징 집계.
- **데이터 로더**:
  - 학습, 검증, 테스트 데이터셋을 위한 로더 제공.
  - 배치 데이터에 패딩 처리를 적용.
- **학습 및 검증**:
  - Cross-Entropy Loss 및 가중치를 사용하여 클래스 불균형 해결.
  - Adam Optimizer를 통한 모델 학습.
  - 학습 및 검증 결과를 에포크별로 시각화.
- **결과 시각화**:
  - 학습 및 검증 손실/정확도를 그래프로 저장.

# 사용자 정의
- **`클래스 수 조정`**: k는 분류해야 할 클래스(카테고리)의 수를 나타냅니다. 모델을 초기화할 때 적절한 k 값을 설정하여 사용할 데이터셋의 클래스 수에 맞게 조정하세요. 
- **`Oversampling 활성화`**: PointNetDataset 초기화 시 oversample=True로 설정.
- **`배치 크기 및 에포크 조정`**: 스크립트에서 batch_size 및 num_epochs 값을 수정 필요요.

# 코트 구성
  - 프로젝트의 디렉토리 구조와 각 구성 요소를 다음과 같이 설명함:
    ```bash
    RIST_LIDAR_2024/
    │
    ├── dataset/
    │   ├── test/
    │   │   └── merged-20241002-000022.pcd   # 테스트용 PCD(Point Cloud Data) 파일
    │   └── train/
    │       └── merged-20240714-215656_000000.csv
    │       └── merged-20240926-215934_000000.csv
    │       └── ...                          # 학습용 CSV 데이터 파일
    │
    ├── model/
    │   └── lidar_poinnet_model_3class_oversample.pth   # 학습이 완료된 PointNet 모델 파일
    │
    ├── results/
    │   ├── test_results_pcd.csv            # 테스트 결과 (예측 레이블 포함)
    │   ├── training_validation_accuracy.png # 학습 및 검증 정확도 그래프
    │   └── training_validation_loss.png     # 학습 및 검증 손실 그래프
    │
    ├── LICENSE                             # 프로젝트 라이선스 정보
    ├── pointnet_test_pcd.py                # 학습된 모델을 테스트하는 Python 스크립트
    ├── pointnet_training_3class.py         # PointNet 모델을 학습시키는 Python 스크립트
    ├── README.md                           # 프로젝트에 대한 설명 및 사용법 문서
    └── requirements.txt                    # 프로젝트 실행에 필요한 Python 패키지 리스트
    ```

# 사용 방법
## 설치
1. 저장소 복제:
   ```bash
   git clone https://github.com/4INLAB-Inc/RIST_LIDAR_2024.git
   cd RIST_LIDAR_2024
2. 필수 Python 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```
## 기본 사용법
1. 모델 학습:
   - 학습 데이터를 **(/dataset/train/)** 디렉토리에 저장함.
   - 학습 스크립트를 실행함:
   ```bash
        python pointnet_training_3class.py
    ```
   - **모델 디렉토리 (/model/)**: 학습이 완료된 모델 파일 (.pth)을 저장하는 디렉토리입니다. 이 디렉토리에 저장된 모델은 테스트 또는 배포 시 사용됨됨.
   - 학습 결과는 **(results/)** 디렉토리에 저장됨됨.

3. 모델 테스트:
   - 테스트할 예시 PCD 파일을 **(dataset/test/)** 디렉토리에 저장함함.
   - 테스트 스크립트를 실행하여 모델 성능을 평가함:
   ```bash
        python pointnet_test_pcd.py
    ```
   - 결과는 3D 시각화 및 CSV 파일 **(results/test_results_pcd.csv)**로 저장됩니다. 각 점의 예측된 클래스 레이블을 포함합니다.

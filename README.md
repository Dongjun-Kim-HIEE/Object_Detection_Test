# 동전 / 배터리 객체 탐지

기존 COCO 80종 YOLO11n과 사용자 데이터 6종 모델을 함께 실행합니다.
단일 가중치에 86종을 재학습한 모델은 아닙니다. 두 모델을 실행하므로 기본 모델만 쓸 때보다 느립니다.

## 처음 실행하기

Python 환경이 설치된 PC에서 다음 명령을 실행합니다.

```powershell
git clone https://github.com/Dongjun-Kim-HIEE/Object_Detection_Test.git
cd Object_Detection_Test
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

아래 실행 예제의 `python`은 설치한 환경의 Python을 사용하세요.
PowerShell에서는 `.\.venv\Scripts\python.exe`로 바꿔 실행할 수 있습니다.

데이터셋과 모델 가중치, 학습 결과는 Git에 포함하지 않습니다.
공유받은 원본 데이터셋을 다음 구조로 배치하세요. 데이터셋에 포함된
README와 라이선스/출처 표기도 함께 유지하세요.

```text
Object_Detection_Test/
├── train_custom.py
├── detect.py
├── yolo11n.pt
├── battery.yolov11/
│   ├── train/images/ 및 train/labels/
│   ├── valid/images/ 및 valid/labels/
│   └── test/images/ 및 test/labels/
└── coin.v1i.yolov11/
    ├── train/images/ 및 train/labels/
    └── valid/images/ 및 valid/labels/
```

기본 가중치 `yolo11n.pt`도 이 폴더에 배치하세요.
학습된 사용자 모델을 공유받았다면 `models/coins_battery.pt`에 배치하세요.
`combined_dataset`은 학습 스크립트가 생성합니다.
현재 학습 및 탐지는 CPU를 사용합니다.

## 공동 작업

저장소 소유자가 GitHub의 Settings → Collaborators에서 친구를 초대하고,
친구는 초대를 수락한 뒤 저장소를 복제합니다.
코드 변경은 개인 브랜치에서 커밋하고 GitHub Pull Request로 병합하세요.
각자 학습한 설정과 검증 지표를 비교하고, 모델 파일은 별도로 공유합니다.
GitHub에 코드를 공유하는 것만으로 여러 PC의 학습이 자동으로 합쳐지지는 않습니다.

## 학습

이 폴더에서 실행:

```powershell
python train_custom.py
```

기본값은 CPU, 640px, batch 8, 최대 30 epochs, patience 10입니다.
원본 데이터는 유지하며 `combined_dataset`에 별도 라벨을 생성합니다.
다각형은 외접 사각형으로 변환합니다. 클래스는 Coin, Pouch battery,
battery, cylindrical battery, drycell, prismatic battery 순입니다.
학습 1,044장, 검증 297장입니다. 배터리 test 100장은 학습/검증에서 제외하며
동전 test 이미지가 없어 전체 6종의 독립 테스트 성능은 아직 측정하지 않습니다.
원본 데이터 출처와 CC BY 4.0 표기는 각 데이터 폴더 README와 data.yaml을 참고하세요.

학습 결과는 `runs/coins_battery*`에 저장됩니다. 학습과 검증이 성공하면
최고 가중치를 `models/coins_battery.pt`에 복사합니다.
검증 지표는 해당 실행 폴더의 `validation.json`에 저장합니다.
학습 완료만으로 실사용 정확도를 보장하지 않으므로 실제 카메라 환경에서 확인해야 합니다.

백그라운드 학습 로그는 `training.stdout.log`, `training.stderr.log`,
프로세스 번호는 `training.pid`입니다. PC가 절전되면 학습이 멈춥니다.
중단된 학습은 실제 실행 폴더의 마지막 체크포인트로 재개합니다:

```powershell
python train_custom.py --resume runs/coins_battery/weights/last.pt
```

## 탐지

```powershell
python detect.py
python detect.py test.mp4
```

학습 완료 후 탐지 프로그램을 다시 시작하면 추가 모델을 자동으로 불러옵니다.
추가 가중치가 없으면 안내 후 기본 모델만 사용합니다.
JSON의 기존 class_id 0–79는 유지하고 추가 클래스는 80–85입니다.
`model` 필드로 base/custom을 구별합니다. 종료 키는 q입니다.

학습 API 참고: https://docs.ultralytics.com/modes/train/

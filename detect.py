import sys
import json
from pathlib import Path
import cv2
from ultralytics import YOLO

# 최초 실행 시 가중치 다운로드에 인터넷 연결이 필요합니다.
root = Path(__file__).resolve().parent
model = YOLO(str(root / 'yolo11n.pt'))
models = [('base', model, 0)]
custom_path = root / 'models' / 'coins_battery.pt'
if custom_path.exists():
    models.append(('custom', YOLO(str(custom_path)), len(model.names)))
else:
    print('추가 모델 학습 전: 기본 모델만 사용합니다.', file=sys.stderr)

# python detect.py          → PC 웹캠
# python detect.py test.mp4 → 저장된 영상
source = sys.argv[1] if len(sys.argv) > 1 else "0"
source = int(source) if source.isdecimal() else source

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"영상 입력을 열 수 없습니다: {source}")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        objects = []
        display = frame.copy()
        for model_name, detector, offset in models:
            result = detector.predict(frame, device='cpu', imgsz=640,
                                      conf=0.25, verbose=False)[0]
            for box in result.boxes:
                class_id = int(box.cls.item())
                objects.append({
                    'class_id': class_id + offset,
                    'name': result.names[class_id],
                    'model': model_name,
                    'confidence': round(float(box.conf.item()), 3),
                    'bbox_xyxy': box.xyxy[0].int().tolist(),
                })
            display = result.plot(img=display)

        # 이후 이 데이터를 API/WebSocket으로 웹 대시보드에 전달합니다.
        print(json.dumps({"objects": objects}, ensure_ascii=False))

        cv2.imshow("Detection", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

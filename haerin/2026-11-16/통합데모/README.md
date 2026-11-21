# 정해린 · 통합 데모: 사진 한 장 → 사과별 결과 화면

2026년 11월 16일 작성, 17일 무게 추가 · 발표자료 4쪽의 마지막 단계 "결과 화면"과 13쪽의 "판단 보류"를 하나로 이었다.

![평가 사진의 결과 화면](results/_MG_2669_18_결과화면.png)

## 하는 일

과수원 RGB 사진, 깊이 지도, 사과별 보이는 윤곽을 받아 사과마다 아래 값을 낸다.

| 출력 | 방법 | 근거 |
|---|---|---|
| 익음도 (익음/미숙/보류) | 윤곽의 박스를 잘라 미세조정한 ResNet18로 점수를 낸다. 점수 ≥ 1.27이면 익음, < −3.20이면 미숙, 그 사이는 보류 | [익음도 보강](../익음도보강/README.md). 두 문턱은 검증 자료에서 '익음 판정 정밀도 80%', '익음의 95%는 미숙으로 안 보냄'으로 정했다 |
| 지름 (mm) | 보이는 영역의 긴 변 × 깊이 중앙값 ÷ 초점거리 | [지름 추정](../지름추정/README.md) |
| 예상 무게 (g) | `0.0037 × 지름^2.52`. 지름을 낸 사과만 계산 | [무게 추정](../무게추정/README.md), 실험실 교차검증 14.2 g |
| 판단 보류 사유 | 화면 경계에 닿음, 깊이 부족(절반 넘게 빔), 가림 의심(채움 비율 0.80 미만) | 지름 추정에서 학습+검증으로 정한 값 |

지름의 깊이 대푯값과 보류 문턱은 `../지름추정/results/metrics.json`에서 읽는다. 무게 식과 범위는 `../무게추정/results/metrics.json`에서 읽는다. 각 실험의 설정을 바꾸면 데모도 같이 바뀐다.

무게는 사진에서 바로 맞히지 않는다. [무게 검증](../무게검증/README.md)에서 그 방식이 사과가 아니라 카메라를 읽는다는 것을 확인했다. 대신 지름 → 무게 식을 쓴다. 지름이 식을 맞춘 범위(59~93 mm) 밖이면 무게 옆에 `*`를 붙인다.

## 종호와 맞출 입력 형식

검출·분할 쪽은 사진마다 `<사진 이름에서 확장자 뺀 것>.json` 파일 하나를 폴더에 넣어 주면 된다.

```json
{"image": "_MG_2669_18.png",
 "instances": [{"id": "0", "score": 0.91, "polygon": [[512, 300], [530, 298], ...]}]}
```

- `polygon`은 **원본 PNG의 픽셀 좌표**다. 1300×1300 사진이면 사방 100 px 검은 여백을 포함한 좌표다. 꼭짓점은 픽셀 중심을 포함하는 정수 좌표로 본다.
- `id`는 그 사진 안에서만 구분되면 된다. `score`는 기록용이며 데모에서 쓰지 않는다.
- `results/*_검출입력예시.json`이 이 형식으로 만든 실제 파일이다. 데이터셋 윤곽을 그대로 옮겼다.

Ultralytics YOLOv8-seg 결과를 이 형식으로 저장하는 예다. 이 저장소의 학습된 YOLO 가중치로는 아직 실행해 보지 않았다.

```python
import json
from ultralytics import YOLO

result = YOLO("best.pt")("_MG_2669_18.png")[0]
instances = [dict(id=str(i), score=float(conf), polygon=poly.round().astype(int).tolist())
             for i, (poly, conf) in enumerate(zip(result.masks.xy, result.boxes.conf))]
json.dump(dict(image="_MG_2669_18.png", instances=instances), open("detections/_MG_2669_18.json", "w"))
```

## 결과 화면이 내놓는 값

`results/<사진>_결과.json`의 `apples` 항목이 결과 화면으로 넘기는 값이다.

| 필드 | 뜻 |
|---|---|
| `number` | 화면의 번호. 위에서 아래, 왼쪽에서 오른쪽 순서 |
| `id` | 입력 JSON의 id |
| `ripeness`, `ripeness_logit` | `ripe`/`unripe`/`hold`와 점수. 점수는 확률로 보정하지 않았다 |
| `diameter_mm` | 사용할 지름. 보류면 `null` |
| `diameter_estimate_mm` | 보류 여부와 상관없이 계산한 값(점검용) |
| `weight_g`, `weight_note` | 예상 무게. 지름 보류면 `null`. 식 범위 밖이면 `식 범위 밖` |
| `hold_reason` | 빈 문자열이면 사용. 아니면 `화면 경계`, `깊이 부족`, `가림 의심 (채움 0.58)` |
| `fill_ratio`, `depth_m`, `depth_valid_fraction`, `longer_side_px` | 지름 계산의 중간값 |
| `truth_*`, `ripeness_split_of_apple` | 정답 비교용. 검출 JSON을 넣으면 쓰지 않는다 |

## 평가 사진 다섯 장에서 돌려 본 결과

익음도 실험과 지름 실험 모두에서 평가 분할에 속한 사진 다섯 장이다. 익은 사과가 많이 찍힌 사진을 골랐으므로 아래 수는 시연 확인용이며 성능 지표가 아니다.

| 사진 | 사과 | 지름 사용 | 처리 시간 |
|---|---:|---:|---:|
| `_MG_2669_18.png` | 10 | 5 | 0.04 s |
| `_MG_2724_16.png` | 8 | 2 | 0.03 s |
| `_MG_2756_08.png` | 7 | 1 | 0.03 s |
| `_MG_2765_14.png` | 5 | 1 | 0.03 s |
| `_MG_2800_19.png` | 6 | 0 | 0.03 s |

모델을 불러온 뒤 자주 쓰는 배치 크기로 GPU를 미리 돌려 둔다. 실행할 때마다 시간은 수 ms씩 달라진다. [처리 시간 기록](results/처리시간.json)

- 사과 36개 중 지름을 낸 것은 9개다. 이 9개의 실측 대비 평균 절대오차는 3.4 mm였다. 나머지는 대부분 가림 의심으로 보류했다. 이 9개에 예상 무게 175~346 g이 나왔다. 과수원 사과는 실측 무게가 없어 무게 정답과 비교하지 못했다.
- 익음도 정답이 있고 미세조정의 학습·검증에 쓰지 않은 사과 26개의 판정은 다음과 같다.
  - 익음 판정 8개: 5개가 실제 익음, 3개는 미숙
  - 보류 9개: 익음 5, 미숙 4
  - 미숙 판정 9개: 6개가 실제 미숙, 3개는 익음을 놓침
- 기존 분류층(`--ripeness-model linear`)은 같은 사진에서 익음을 18개 골랐다. 그중 5개가 미숙이었고, 놓친 익음은 없었다. 미세조정 모델은 덜 고르고 애매하면 보류한다. 대신 놓치는 익음 사과가 생겼다.
- 사진 다섯 장의 수는 작아서 모델 비교의 근거가 아니다. 비교는 [익음도 보강](../익음도보강/README.md)의 평가 3,023장 결과를 본다. 평가 3,023장에서 미세조정 모델의 익음 판정 정밀도는 0.67, 보류 비율은 21%였다.
- 화면에는 미세조정의 학습·검증 분할에 있던 사과를 따로 표시했다. 같은 사과의 다른 사진으로 학습한 경우라 정답과 맞아도 실력으로 보지 않는다.
- 처리 시간은 RTX 5070 Laptop GPU에서 사진 한 장당 평균 **0.03 s**다. 모델 로드, 파일 읽기, 사과 검출은 빠졌다. 검출을 더한 FPS는 YOLO 쪽 시간을 합쳐 다시 재야 한다.

## 실행

이 폴더를 PowerShell에서 연다. 전이학습 환경을 쓰며 GPU가 없으면 `--device cpu`로 돌아간다.

```powershell
$pythonPath = 'C:\Users\sclov\PycharmProjects\Graduation-project\.venv\Scripts\python.exe'
& $pythonPath demo.py --image _MG_2669_18.png _MG_2724_16.png _MG_2756_08.png _MG_2765_14.png _MG_2800_19.png
```

YOLO 결과를 넣을 때는 JSON 폴더를 지정한다. 이때는 정답 비교 줄이 나오지 않는다.

```powershell
& $pythonPath demo.py --image _MG_2669_18.png --detections-dir '.\detections'
```

필요한 것은 다음과 같다.
- `D:\Apple-harvest-data\02-annotated_data_fuji.zip` — 깊이, 초점거리, 실측 지름, 데이터셋 윤곽
- `D:\Apple-harvest-data\ripeness\<분할>\images\` — RGB 사진
- `../전이학습/weights/resnet18-f37072fd.pth` — ImageNet 가중치(저장소에는 올리지 않음)
- `../익음도보강/weights/ripeness_finetuned.pt`(저장소에 올리지 않음, `../익음도보강`에서 다시 학습), `../익음도보강/results/metrics.json`, `apple_splits.csv`
- `../지름추정/results/metrics.json`, `../무게추정/results/metrics.json`
- 기존 분류층으로 돌릴 때는 `--ripeness-model linear`. 이때 `../전이학습/ripeness/results/head_aug.pt`를 쓴다. 이 모드는 교체 전 결과와 똑같이 나오는 것을 확인했다.

검출 JSON으로 넣은 결과가 데이터셋 윤곽을 그대로 쓴 결과와 같은지 `_MG_2724_16.png`로 확인했다. 익음도, 지름, 보류 사유가 모두 일치했다.

## 남은 일

- 종호의 YOLO 마스크로 같은 다섯 장을 돌려 정답 윤곽과 비교한다. 검출 누락·오검출과 윤곽 오차가 지름에 얼마나 더해지는지 본다.
- 익음도 보류는 점수 구간으로만 정했다. 실제 가림 정도와 연결하는 것은 남은 일이다. 같은 사과를 여러 프레임에서 모아 판단하면 보류를 줄일 수 있다.
- 2분 시연 영상은 여러 사진의 결과 화면을 이어 붙이거나, Apple MOTS 같은 영상 자료에서 프레임별로 돌려 만든다.

이미지·깊이 출처: Gené-Mola 등, AmodalAppleSize_RGB-D, CC BY 4.0. 결과 화면은 원본 사진을 축소하고 윤곽과 결과를 덧붙였다. [데이터 DOI](https://doi.org/10.34810/data916)

# mi-finding-smi-marks

SMI Review/Popup 화면에서 템플릿의 중심 좌표를 찾아 반환하는 OpenFaaS
함수입니다. 기존 ChatGPT 대화에 있던 실제 운영 구조에 맞춰
`handler.py`가 요청 처리 전체를 담당하고, 템플릿 매칭 함수는
`util_functions.py`에 모았습니다.

## 실행 구조

```text
Scala 호출 프로그램
  -> OpenFaaS Gateway: /function/mi-finding-smi-marks
  -> Provider / Kubernetes Service
  -> 함수 Pod의 fwatchdog
  -> /home/app/index.py (Flask :5000)
  -> /home/app/function/handler.py의 handle(req)
  -> MinIO 템플릿 조회 및 OpenCV 매칭
  -> Flask JSON Response
```

함수 Pod는 최소 replica 1개로 유지됩니다. 요청이 오면 OpenFaaS Python
템플릿의 `index.py`가 원문 요청 body를 `handler.handle(req)`에 넘기고,
반환된 Flask Response가 같은 경로를 역순으로 호출자에게 전달됩니다.

## 저장소 구조

```text
mi-finding-smi-marks.yml       OpenFaaS build/deploy 설정
http_test.py                   로컬 HTTP 실행기(기본 포트 7184)
mi_finding_smi_marks/
  __init__.py
  handler.py                   FaaS 진입점, MinIO 조회, Review/Popup 분기
  util_functions.py            전처리, 지표, Full/Partial 템플릿 매칭
  riselog.py                   stdout JSON 로그 호환 함수
  requirements.txt             함수 이미지에 설치할 Python 패키지
tests/                         FaaS 응답 계약과 매칭 로직 테스트
```

OpenFaaS 언어 템플릿이 생성하는 `index.py`와 `fwatchdog`은 이 저장소에
직접 넣지 않습니다. 빌드할 때 `python3-flask-debian` 템플릿이 제공하며,
함수 폴더는 컨테이너의 `/home/app/function`으로 복사됩니다.

## 요청과 응답

Review 호출은 `mode`에 `review` 또는 일반 문자열을 사용합니다.

```json
{
  "image": "BASE64_PNG_OR_JPEG",
  "product": "PART_ID",
  "layer": "STEP_SEQ",
  "recipe": "RECIPE_ID",
  "eqpid": "EQP_ID",
  "mode": "review"
}
```

성공:

```json
{"success": true, "message": "340,271"}
```

실패는 비즈니스 결과이므로 HTTP 200과 `-1,-1`을 반환합니다.

```json
{"success": false, "message": "-1,-1"}
```

잘못된 JSON이나 필수 필드 누락은 HTTP 400, MinIO 같은 내부 오류는 HTTP
500입니다.

Popup 호출은 다음처럼 보냅니다.

```json
{
  "image": "BASE64_PNG_OR_JPEG",
  "eqpid": "EQP_ID",
  "mode": "popup"
}
```

`popup_on_target`, `popup_next_site` 순서로 각각 찾고, 한쪽만 발견되면 다른
좌표만 `-1,-1`로 유지합니다.

```json
{"success": true, "message": "(120,80), (-1,-1)"}
```

## 템플릿과 판정 규칙

템플릿은 MinIO의 기본 bucket/prefix 아래에서 읽습니다.

```text
static/MI/GA_TEMPLATE/{product}...
static/MI/GA_TEMPLATE/popup_on_target...
static/MI/GA_TEMPLATE/popup_next_site...
```

Review는 먼저 `{product}_{layer}` prefix로 object를 조회합니다. 해당 prefix에
이미지가 하나도 없으면 기존 운영 코드처럼 `MI/GA_TEMPLATE/` 아래의 모든
Review 이미지를 다시 가져와 매칭을 계속합니다. 이때 object name에 `popup`이
포함된 템플릿은 대소문자와 관계없이 제외합니다. 특정 템플릿이 존재하지만 시각
매칭 점수만 낮은 경우에는 전체 이미지 fallback을 실행하지 않습니다.

Popup도 먼저 `popup` prefix를 조회하고, 이미지가 없을 때 같은 전체 이미지
fallback을 적용한 뒤 `popup_on_target`, `popup_next_site` 이름으로 나눠
판정합니다.

Popup은 `popup_on_target`, `popup_next_site` 각각에서 score가 `0.5`를 초과한
후보 중 최고 점수를 사용합니다.

```text
Full score >= 0.70
  -> variance ratio, histogram/SSIM, NMI >= 0.20 조건을 통과한 후보만 PASS

0.60 <= Full score < 0.70
  -> 동일한 보조 지표 조건 통과 시 PASS

Full 실패
  -> 원본 템플릿의 상/하/좌/우 0.70 edge로 재검색
  -> 오탐 데이터가 누적된 0.35 edge는 비활성화

Partial score >= 0.70 + partial 보조 조건 통과
  -> PASS

그 외
  -> FAIL; 마지막 partial 최고점을 임의로 반환하지 않음
```

원본 이미지의 CLAHE grayscale은 요청당 한 번만 계산해 모든 템플릿에서
재사용합니다. 템플릿 및 Partial edge의 CLAHE는 서로 다른 이미지이므로 각각
한 번씩 계산합니다.

Partial에서 잘라낸 edge의 offset을 원본 템플릿 좌표로 복원하므로 반환 좌표는
항상 원본 템플릿 전체의 중심입니다. Histogram correlation에는 `abs()`를
사용하지 않습니다. 음의 상관도를 높은 유사도로 오인하지 않기 위해서입니다.

## MinIO 설정

접속 주소와 인증정보는 코드에 넣지 않습니다.

- `MINIO_ENDPOINT`: 필수, 예: `minio.internal:9000`
- `MINIO_ACCESS_KEY`: 환경변수 또는 OpenFaaS Secret
- `MINIO_SECRET_KEY`: 환경변수 또는 OpenFaaS Secret
- `MINIO_SECURE`: 기본 `false`
- `MINIO_BUCKET`: 기본 `static`
- `MINIO_TEMPLATE_PREFIX`: 기본 `MI/GA_TEMPLATE/`

배포 YAML은 아래 Secret 이름을 마운트합니다.

```text
mi-minio-access-key
mi-minio-secret-key
```

함수는 환경변수를 먼저 사용하고, 없으면 `/var/openfaas/secrets/`의 파일을
읽습니다. 실제 MinIO 값은 서버의 OpenFaaS/Kubernetes 설정으로 주입하세요.

## 빌드와 배포

먼저 `mi-finding-smi-marks.yml`의 `image`를 사내 registry 주소로 바꿉니다.

```yaml
image: registry.internal/mi-finding-smi-marks:latest
```

그 다음 서버에서 실행합니다.

```bash
faas-cli template store pull python3-flask-debian
faas-cli build -f mi-finding-smi-marks.yml
faas-cli push -f mi-finding-smi-marks.yml
faas-cli deploy -f mi-finding-smi-marks.yml
```

`MINIO_ENDPOINT`는 사내 배포 설정에 추가하고, 두 MinIO Secret은 배포 전에
생성되어 있어야 합니다.

동기 호출 시간 제한은 가장 짧은 계층이 적용됩니다. 현재 함수 설정은
`exec_timeout=5m`, 함수 read/write timeout은 `5m30s`입니다. Gateway/Provider와
Scala 호출부는 이보다 길게 설정해야 하며, 기존 대화에서 권장한 관계는 다음과
같습니다.

```text
함수 exec timeout       5분
Gateway/Provider        5분 30초 이상
Scala socket timeout    6분 이상
```

## 로컬 검증

공유 대화의 로컬 HTTP 구성과 같은 방식으로 실행하려면 다음 명령을 사용합니다.

```bash
python http_test.py
```

기본 주소는 `http://0.0.0.0:7184`이며 `HOST`, `PORT` 환경변수로 변경할 수
있습니다. OpenFaaS 배포에서는 이 파일이 아니라 언어 템플릿의 `index.py`가
`handler.handle(req)`를 호출합니다.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
pytest
ruff check .
ruff format --check .
```

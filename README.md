# mi_finding

OpenCV 템플릿 매칭으로 화면 속 대상의 중심 좌표를 찾는 프로젝트입니다.  
ChatGPT의 `mi_finding_smi_marks` 대화에서 정리한 판정 흐름을 실행 가능한 패키지로 구성했습니다.

## 판정 로직

```text
Full score >= 0.70
  -> 바로 PASS

0.60 <= Full score < 0.70
  -> variance ratio, SSIM, histogram, NMI 보조 조건 통과 시 PASS

Full score < 0.60 또는 보조 조건 실패
  -> 0.70, 0.35 비율의 edge partial template으로 재검색

Partial score >= 0.70 + partial 보조 조건 통과
  -> PASS

그 외
  -> FAIL
```

색상 히스토그램 상관도에는 `abs()`를 사용하지 않습니다. 음의 상관도를 양의 유사도로 잘못 해석하지 않기 위해서입니다. Partial 매칭은 잘라낸 edge의 offset을 원래 템플릿 좌표로 환산하므로, 반환 중심점은 원본 템플릿 전체의 중심입니다.

## 설치

Python 3.10 이상이 필요합니다.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## CLI 실행

```bash
mi-finding sample/screen.png sample/templates \
  --output annotated.png \
  --json result.json
```

특정 파일명 prefix만 사용할 수도 있습니다.

```bash
mi-finding screen.png templates --prefix PRODUCT_STEP
```

종료 코드는 PASS면 `0`, FAIL이면 `1`입니다. JSON에는 점수, 중심 좌표, 매칭 방식, Full/Partial 단계와 모든 보조 지표가 포함됩니다.

## Python 사용

```python
from mi_finding import TemplateFinder
from mi_finding.io import load_templates, read_image

image = read_image("screen.png")
templates = load_templates("templates", prefix="PRODUCT_STEP")
result = TemplateFinder().find(image, templates)

print(result.to_dict())
if result.success and result.candidate:
    print(result.candidate.center)
```

임계값은 모두 바꿀 수 있습니다.

```python
from mi_finding import MatchingConfig, TemplateFinder

finder = TemplateFinder(
    MatchingConfig(
        full_min_score=0.60,
        full_direct_score=0.70,
        partial_min_score=0.70,
        variance_ratio_min=0.10,
    )
)
```

## JSON/FaaS 핸들러

`mi_finding.handler.handle()`은 원래 요청 키를 지원합니다.

```json
{
  "image": "BASE64_IMAGE",
  "product": "PART_ID",
  "layer": "STEP_SEQ",
  "eqpid": "EQP_ID",
  "mode": "normal"
}
```

템플릿 루트는 인자로 전달하거나 `MI_TEMPLATE_ROOT` 환경변수로 지정합니다. 일반 모드에서는 `{product}_{layer}`로 시작하는 템플릿만 읽습니다. 대상 템플릿이 없을 때 전체 템플릿을 무조건 검색하던 기존 fallback은 오탐 위험 때문에 제거했습니다.

```python
from flask import make_response
from mi_finding.handler import handle as find_handle

def handle(req):
    body, status = find_handle(req, template_root="MI/GA_TEMPLATE")
    return make_response(body, status)
```

Popup 모드의 템플릿 파일명은 다음 prefix를 사용합니다.

- `popup_on_target`
- `popup_next_site`

## 테스트

```bash
pytest
```

테스트에는 핵심 임계값 분기와 partial edge offset 검증이 포함되어 있습니다.

## GitHub에 올리기

```bash
git init
git add .
git commit -m "feat: add mi_finding template matcher"
git branch -M main
git remote add origin https://github.com/USERNAME/mi_finding.git
git push -u origin main
```

GitHub에서 빈 저장소를 먼저 만든 뒤 `USERNAME`을 자신의 계정명으로 바꾸면 됩니다.

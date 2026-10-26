# RAG3 새 PC 설치 가이드

기준: 2026-10-20, 현재 저장소의 Windows x64 구성. 이 문서는 **새 설치**와 **기존 문서·검색 데이터 이전**을 모두 다룬다. 새 PC에서 전체 절차를 아직 실행해 검증한 문서는 아니다.

## 0. 현재 구성과 설치 위치

| 구성 | 현재 프로젝트 기준 |
| --- | --- |
| Windows Python | 3.11.3, 프로젝트별 `.venv` |
| Python 패키지 | [requirements.reinstall.txt](requirements.reinstall.txt)의 설치 환경 스냅샷 170개. CUDA 12.8용 PyTorch 포함 |
| 문서 처리 | Docling, Docling-Haystack, RapidOCR |
| 임베딩·검색 | BGE-M3, FlagEmbedding, Haystack 3.2.0, Qdrant-Haystack, Qdrant 1.19.2 |
| 화면·MCP | PySide6, MCP Python SDK, 선택적 Codex 로컬 플러그인 |
| Linux 인프라 | WSL2의 Ubuntu, 그 안에서 실행되는 Docker Engine·Compose, Qdrant 컨테이너 |
| 실행기 | .NET 8 SDK로 `RAG3.exe` 빌드; 실행 PC에는 .NET 8 런타임 필요 |
| 추가 프로그램 | 구형 `.doc` 파일을 수집할 때만 Windows용 Microsoft Word 필요 |

**`C:\python_program\RAG3`에 설치하는 것이 현재 코드 그대로 재현하는 경로다.** [인프라 시작 스크립트](scripts/ensure-infra.ps1), [플러그인 설정](plugins/rag3/mcp.json)에 이 경로가 들어 있다. 저장된 문서의 `source_uri`도 원본의 절대 경로다. 다른 경로를 택하면 이 설정을 고치고 기존 Qdrant 메타데이터의 원본 경로도 이전해야 한다.

현재 수집·검색의 운영 데이터베이스는 Qdrant다. [compose.yaml](compose.yaml)에 PostgreSQL 서비스도 정의되어 있지만, UI 시작 스크립트는 Qdrant만 기동한다. PostgreSQL이 필요한 별도 데이터를 복원할 때만 해당 서비스를 추가로 시작한다.

## 1. 기존 PC에서 이전할 자료 준비

새 PC에서 **빈 컬렉션으로 시작한다면 이 절을 건너뛴다.** 기존 검색 결과와 원본 파일까지 옮기려면 다음을 함께 보관한다.

| 자료 | 이전 방법 |
| --- | --- |
| 프로젝트 소스 | Git 저장소 또는 프로젝트 폴더. 이 가이드와 `requirements.reinstall.txt`도 포함 |
| `data/models/` | 폴더 전체 복사. BGE-M3와 Docling 모델이 Git에 포함되지 않음 |
| `data/sources/` | 폴더 전체 복사. Qdrant의 원본 경로와 파일 해시 검증에 필요 |
| Qdrant 컬렉션 | 컬렉션별 스냅샷 생성·다운로드. `data/` 복사만으로는 벡터가 이전되지 않음 |
| `.env` | 안전한 매체로 보관하거나 새 PC에서 재작성. Git 제외 파일이며 DB 암호가 들어 있음 |
| 선택 사항 | `data/checkpoints/`, `data/parsed/`, `data/chunks/`, `data/embeddings/`, `data/ui-state.json` 및 필요한 작업 기록 |

수집 중인 작업이 끝난 뒤 Qdrant 스냅샷을 만든다. 아래 PowerShell 예시는 **기존 PC**의 프로젝트 폴더에서 실행하며, `$backupDir`은 실제 백업 드라이브로 바꾼다.

```powershell
cd C:\python_program\RAG3
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\ensure-infra.ps1
$backupDir = 'D:\RAG3-transfer\qdrant'
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
$names = (Invoke-RestMethod 'http://127.0.0.1:16333/collections').result.collections.name
foreach ($name in $names) {
    $snapshot = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:16333/collections/$name/snapshots"
    $snapshotName = $snapshot.result.name
    Invoke-WebRequest -Uri "http://127.0.0.1:16333/collections/$name/snapshots/$snapshotName" `
        -OutFile (Join-Path $backupDir "$name--$snapshotName")
}
```

스냅샷 파일과 복사한 `data/`를 같은 이전 매체에 보관한다. [Qdrant의 공식 스냅샷 문서](https://qdrant.tech/documentation/operations/snapshots/)는 컬렉션 단위 백업과 복원을 설명한다. 현재 [backup-foundation.ps1](scripts/backup-foundation.ps1)은 별도 PostgreSQL 백업도 수행하지만, 모델·원본 파일까지 묶는 완전한 RAG3 이전 도구는 아니다.

## 2. 새 PC의 Windows·WSL 준비

1. Windows x64에 **Python 3.11**과 Git을 설치한다. 검증된 기존 환경의 Python 버전은 3.11.3이다. [Python 다운로드](https://www.python.org/downloads/)
2. NVIDIA GPU를 사용할 경우 호환 드라이버를 설치한다. 현재 의존성은 `torch==2.11.0+cu128`이며, 아래 검증 스크립트는 CUDA GPU가 보이는지 검사한다. CPU 전용 구성은 이 가이드의 검증 범위 밖이다. [NVIDIA 드라이버](https://www.nvidia.com/Download/index.aspx)
3. 실행기를 **빌드**하려면 .NET 8 SDK를 설치한다. 이미 빌드된 `RAG3.exe`만 사용할 경우 .NET 8 런타임이 필요하다. [다운로드](https://dotnet.microsoft.com/download/dotnet/8.0)
4. 관리자 PowerShell에서 `wsl --install -d Ubuntu`를 실행하고 안내에 따라 재시작한다. Ubuntu를 최초 실행해 초기 계정을 만든 다음 기본 배포판으로 지정한다. [Microsoft의 WSL 설치 안내](https://learn.microsoft.com/en-us/windows/wsl/install)

```powershell
wsl --install -d Ubuntu
# 재시작 및 Ubuntu 최초 설정 후:
wsl --set-default Ubuntu
wsl --list --verbose
```

Ubuntu 안에 Docker Engine과 Compose 플러그인을 [Docker의 Ubuntu 공식 설치 절차](https://docs.docker.com/engine/install/ubuntu/)에 따라 설치한다. RAG3의 시작 스크립트는 `wsl.exe -u root -- docker ...`를 호출하므로 아래 명령으로 **WSL 내부**의 Docker와 Compose가 작동하는지 확인한다. Windows에서 `docker.exe`가 보이는지만 확인하는 것으로는 충분하지 않다.

```powershell
wsl -d Ubuntu -u root -- docker info
wsl -d Ubuntu -u root -- docker compose version
```

## 3. 소스와 Python 패키지 설치

프로젝트를 `C:\python_program\RAG3`에 복원한다. Git을 쓴다면 저장소 URL을 자신의 주소로 바꾼다. 이 가이드와 `requirements.reinstall.txt`가 Git에 아직 반영되지 않았다면 함께 별도로 복사한다.

```powershell
New-Item -ItemType Directory -Force C:\python_program | Out-Null
git clone <저장소-URL> C:\python_program\RAG3
cd C:\python_program\RAG3
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.reinstall.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.reinstall.txt`는 기존 PC의 `.venv`에서 `pip freeze`로 얻은 **170개 패키지의 정확한 버전 목록**이며, CUDA 12.8 PyTorch 패키지 인덱스도 지정한다. 설치에는 인터넷이 필요하다. 기존 `requirements.lock.txt`는 137개만 기록하고 있어 FlagEmbedding, MCP, PySide6, tiktoken 등 현재 설치된 일부 패키지가 빠져 있다. 따라서 새 PC 복원에는 단독으로 사용하지 않는다. `.venv` 자체를 다른 PC에서 복사하지 말고 다시 만든다.

## 4. 모델 파일 준비

가장 확실한 방법은 기존 PC의 `data/models/` 전체를 **같은 상대 경로**로 복사하는 것이다. 이 안에 `data/models/huggingface/`의 BGE-M3와 `data/models/docling/`의 Docling 모델이 있다.

기존 모델 캐시가 없고 인터넷에서 새로 받을 경우:

```powershell
cd C:\python_program\RAG3
New-Item -ItemType Directory -Force data\models\huggingface, data\models\docling | Out-Null
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', cache_folder=r'data\models\huggingface')"
& .\.venv\Scripts\docling-tools.exe models download -o data\models\docling
```

RAG3는 `data/models/` 아래에서 BGE-M3를 찾고 Docling의 `artifacts_path`를 `data/models/docling/`로 지정한다. [Docling 공식 모델 미리받기 안내](https://docling-project.github.io/docling/usage/advanced_options/)

## 5. Qdrant 시작

`.env`를 새 PC에 복원하거나 프로젝트 루트에 만든다. 실제 암호로 바꾸고 Git에 넣지 않는다. PostgreSQL을 당장 실행하지 않더라도 Compose 파일이 아래 세 값을 참조한다.

```dotenv
POSTGRES_USER=rag3
POSTGRES_PASSWORD=<새-PC에서-설정할-암호>
POSTGRES_DB=rag3
```

Compose에서 두 볼륨을 `external`로 선언했으므로 먼저 만든다. 새 PC에 **빈 볼륨**을 만드는 단계다. 그다음 프로젝트의 Qdrant 시작 스크립트를 실행한다.

```powershell
cd C:\python_program\RAG3
wsl -d Ubuntu -u root -- docker volume create rag3_qdrant_data
wsl -d Ubuntu -u root -- docker volume create rag3_postgres_data
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\ensure-infra.ps1
Invoke-RestMethod 'http://127.0.0.1:16333/collections'
```

현재 [compose.yaml](compose.yaml)은 Qdrant `v1.19.2`를 Windows의 `127.0.0.1:16333`으로 연결한다. PostgreSQL이 별도로 필요할 때만 `wsl -d Ubuntu -u root -- docker compose -f /mnt/c/python_program/RAG3/compose.yaml up -d postgres`를 실행한다.

## 6. 기존 컬렉션·원본 복원 (이전하는 경우)

1. 기존 PC의 `data/models/`와 `data/sources/`를 새 PC의 `C:\python_program\RAG3\data\` 아래에 복사한다. 필요한 체크포인트·중간 결과도 같은 위치에 복사한다.
2. Qdrant가 실행 중이고 복원 대상 컬렉션이 아직 없는지 확인한다.
3. 이전해 온 **각 컬렉션의 스냅샷**을 업로드한다. 아래 예시의 변수 두 개를 실제 컬렉션명과 파일 경로로 바꿔 반복한다.

```powershell
$collection = 'my_rag1'
$snapshotPath = 'D:\RAG3-transfer\qdrant\my_rag1--snapshot.snapshot'
curl.exe --fail-with-body -X POST `
    "http://127.0.0.1:16333/collections/$collection/snapshots/upload?priority=snapshot" `
    -F "snapshot=@$snapshotPath"
Invoke-RestMethod "http://127.0.0.1:16333/collections/$collection"
```

Qdrant 공식 문서는 업로드 복원에 `priority=snapshot`을 사용한다. 기존 PC와 새 PC의 컬렉션별 포인트 수를 비교하고, RAG3에서 검색과 원본 파일 열람을 각각 확인한다. **Qdrant 스냅샷만으로 원본 파일은 복원되지 않는다.** RAG3의 원본 조회는 Qdrant 메타데이터의 `source_uri`, `data/sources/`의 실제 파일, SHA-256을 함께 검사한다.

## 7. 실행기 빌드와 동작 확인

```powershell
cd C:\python_program\RAG3
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify-dev-env.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-launcher.ps1
.\RAG3.exe
```

`verify-dev-env.py`의 `DEV_ENV_CORE_VERIFICATION=OK`와 Qdrant의 컬렉션 목록을 확인한다. UI가 뜬 뒤 컬렉션 목록·검색·원본 열람을 시험한다. `RAG3.exe`는 Git에서 제외되므로 새 PC에서 빌드한다. UI를 `RAG3.exe` 없이 시작하려면 `.\.venv\Scripts\python.exe scripts\ingest-ui.py`를 실행한다.

구형 `.doc`도 사용할 경우 새 PC에 Microsoft Word를 설치한 뒤 실제 `.doc` 파일 하나로 DOC→DOCX 변환 경로를 시험한다. `.docx`에는 이 조건이 없다.

## 8. 선택 사항: Codex·ChatGPT 연결

Codex 로컬 플러그인을 사용할 경우 [plugins/rag3/mcp.json](plugins/rag3/mcp.json)의 `RAG3_PROJECT_ROOT`가 실제 설치 경로인지 확인한다. UI 시작 시 자동 등록 검사도 수행하지만, 수동 설치와 확인은 다음과 같다.

```powershell
cd C:\python_program\RAG3
codex plugin marketplace add C:\python_program\RAG3
codex plugin add rag3@rag3-local
.\.venv\Scripts\python.exe scripts\ensure-rag3-codex-plugin.py --check-only
```

새 로컬 Codex 대화에서 `@RAG3`를 선택해 컬렉션 목록을 조회한다. 로컬 플러그인은 RAG3의 MCP 서버를 `stdio`로 실행하며, Qdrant와 `.venv`가 필요하다. [OpenAI의 로컬 플러그인·마켓플레이스 안내](https://developers.openai.com/plugins/build/plugins)와 프로젝트의 [RAG3_PLUGIN_SETUP.md](RAG3_PLUGIN_SETUP.md)를 참고한다.

ChatGPT 웹·클라우드 연결은 로컬 플러그인 설치만으로 이뤄지지 않는다. 해당 기능이 필요할 때만 [RAG3_PLUGIN_SETUP.md](RAG3_PLUGIN_SETUP.md)의 터널·사용자 지정 MCP 연결 절차를 별도로 수행한다. 새 PC의 터널 자격 증명은 새 환경에 맞게 안전하게 다시 설정한다.

## 9. 설치 완료 기준과 흔한 실패 위치

| 확인 | 성공 기준 / 확인할 곳 |
| --- | --- |
| Python | `.venv`의 Python 3.11, `pip check` 오류 없음 |
| 모델·GPU | `verify-dev-env.py`가 `BGE_M3_DIM=1024`, `DEV_ENV_CORE_VERIFICATION=OK` 출력 |
| Qdrant | `http://127.0.0.1:16333/collections` 응답; 실패하면 WSL 내부 `docker info`, `docker compose version` 확인 |
| UI | `RAG3.exe` 창 열림; 실패하면 `.venv`, .NET 런타임, Qdrant 확인 |
| 데이터 이전 | 컬렉션 포인트 수, 검색, `data/sources/`의 원본 열람 확인 |
| Codex | `--check-only`가 `already_installed`를 보고하고 새 로컬 대화에서 실제 도구 호출 성공 |

`requirements.reinstall.txt`는 **현재 PC의 패키지 버전 스냅샷**으로, 새 PC에서의 설치 성공이나 장기적인 패키지 배포 가능성을 보증하지 않는다. 설치 오류가 나면 Python 버전, GPU 드라이버, CUDA PyTorch 휠, 각 패키지 인덱스 접근 상태를 순서대로 확인한다.

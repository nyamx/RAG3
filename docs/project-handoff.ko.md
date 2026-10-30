# RAG3 프로젝트 인계서

기준: 2026-10-26 저장소 코드와 공개 문서. 이 인계서를 작성하면서 애플리케이션을 재기동하거나 전체 적재 시험을 다시 실행하지 않았다. 실측과 과거 실험은 [검증 기록](verification.md), [파일 전달 실험](chatgpt-file-handoff.md)에 명시된 범위로 해석한다. 영어판은 [Project handoff](project-handoff.en.md)다.

## 1. 프로젝트가 제공하는 것

RAG3는 Windows PC에 있는 문서를 수집·색인하고 데스크톱 UI, 로컬 Codex MCP 플러그인, ChatGPT 웹·Android 앱의 MCP 플러그인에서 같은 컬렉션을 검색하는 시스템이다. PDF의 텍스트뿐 아니라 가능한 경우 페이지 범위와 원본을 보존해 검색 결과를 확인할 수 있게 한다. ChatGPT 연결은 로컬 MCP 서버 앞에 Secure MCP Tunnel과 사용자별 커스텀 플러그인을 두는 방식이다.

핵심 경계는 **문서 전체를 AI 대화 첨부로 전달하는 기능**과 **색인된 문서의 관련 내용만 검색해 모델에 제공하는 기능**의 차이다. 후자는 실제 사용 경로다. 전자는 호스트 파일 저장까지 확인했지만 대화 첨부 등록과 모델의 원본 PDF 직접 읽기는 확인되지 않았다.

## 2. 전체 실행 구성

~~~text
Windows 사용자
  ├─ PySide6 UI ── 인증된 /api/v1/command ─┐
  ├─ Codex 로컬 MCP ── /mcp ───────────────┤
  └─ ChatGPT 웹·Android ── MCP Tunnel ─ /mcp ┤
                                            ▼
                              공용 HTTP/MCP 백엔드 (Python)
                              ├─ SQLite 작업 저널·스케줄러
                              ├─ 상주 CPU Docling 파서 자식
                              ├─ 장문 PDF용 임시 CUDA 파서 자식
                              ├─ BGE-M3 GPU 모델 서비스
                              ├─ 수집·검색·원본 제공
                              └─ Qdrant (WSL2의 Docker)
                                     ├─ dense: text-dense
                                     └─ sparse: bm25
~~~

정상 HTTP 실행 경로는 [scripts/rag3-mcp-server.py](../scripts/rag3-mcp-server.py) 한 백엔드 프로세스다. UI는 별도 프로세스이지만 작업·검색 상태를 소유하지 않는다. [RAG3.exe](../launcher/Program.cs)는 선택적인 .NET UI 실행기이며 백엔드 대체물이 아니다. Qdrant, 터널 런타임, 파서 자식은 별도 프로세스다. 옛 stdio 서버·트레이·독립 검색 worker는 현재 정상 기동 경로로 취급하지 않는다.

## 3. 기술 선택과 이유

| 기술 | 적용 위치 | 선택 이유와 구현상의 의미 |
| --- | --- | --- |
| Python 3.11, PySide6 | 백엔드·데스크톱 UI | 문서 처리·ML 라이브러리를 같은 언어에서 연결하고 Windows UI는 별도 클라이언트로 유지한다. |
| Docling, pypdfium2 | PDF·구조화 문서 파싱과 페이지 렌더링 | Docling에서 PDF OCR·표 구조 처리를 켜고, PDFium으로 페이지 수 확인과 원본 페이지 이미지를 만든다. 평문 일부 형식은 Docling을 거치지 않는다. |
| Haystack, tiktoken | 문서 변환·토큰 단위 분할, Qdrant 문서 저장 연동 | 수집 구성 요소를 재사용하고 o200k_base 기준으로 청크를 나눈다. 현재 다중 방식 검색과 점수 결합은 RAG3의 검색 코드가 제어한다. |
| BGE-M3, FlagEmbedding, PyTorch CUDA | 1,024차원 dense 임베딩 | 의미 기반 검색을 로컬 모델로 수행한다. GPU 모델은 공용 서비스가 지연 로드하고 재사용한다. |
| Qdrant 1.19.2 | 컬렉션, dense·native BM25 sparse 벡터, 메타데이터 필터, RRF | 의미·키워드 검색을 같은 청크에 적용하고 서버 측 필터·융합을 사용한다. |
| SQLite WAL | 작업·단계·재개 정보·UI 변경 버전 | 별도 DB 서버 없이 수집 상태를 영속화한다. Qdrant 청크 상태와 원본 해시를 함께 확인해야 재개할 수 있다. |
| MCP Streamable HTTP | Codex·ChatGPT 연결 | UI와 AI 클라이언트가 같은 백엔드·도구 경계를 공유한다. ChatGPT 웹은 별도 터널·플러그인 설정이 필요하다. |
| WSL2 + Docker Compose | Windows의 Qdrant 실행 | Qdrant 서비스와 영속 볼륨을 앱 코드와 분리한다. PostgreSQL 정의는 존재하지만 기본 검색·수집 경로에서는 기동하지 않는다. |
| SHA-256 관리 원본 | 원본 보존·조회·재개 검증 | 색인된 청크와 실제 파일의 동일성을 확인하고 변경된 원본으로 묵시적으로 재개하지 않는다. |

버전과 설치 조건의 기준은 [requirements.txt](../requirements.txt), [requirements-ui.txt](../requirements-ui.txt), [compose.yaml](../compose.yaml), [Windows 설치 안내](install-windows.md)다. 모델 가중치, 실제 문서, 데이터 볼륨, 비밀키는 Git 저장소에 포함되지 않는다.

RapidOCR 패키지는 의존성에 고정돼 있지만, 현재 파이프라인 코드는 Docling의 OCR 기능을 켤 뿐 OCR 엔진을 RapidOCR로 명시 선택하지 않는다. 새 설치에서 실제 OCR 백엔드는 별도 확인이 필요하다.

## 4. 문서 수집 구현

1. UI는 로컬 파일을 작업 대기열에 추가하고 사용자가 **Run**을 눌러 실행한다. MCP의 파일/텍스트 제출은 즉시 실행 요청을 만든다. 두 경로 모두 공용 스케줄러와 작업 ID를 사용한다.
2. 원본을 data/sources 아래에 보존하고 SHA-256을 기록한다. MCP 첨부 참조는 [attachments.py](../rag3/attachments.py)가 공개 HTTPS 주소만 확인한 뒤 파일로 스트리밍한다. URL·리다이렉트·파일명·형식·불완전 다운로드를 검사한다. 서명 URL은 작업 기록에 남기지 않는다.
3. TXT/JSON/JSONL/TSV는 가벼운 평문 경로로 분할한다. PDF와 그 밖의 지원 구조화 형식은 Docling이 변환한다. 레거시 DOC는 설치된 Microsoft Word를 이용해 DOCX 파서 사본을 만든다.
4. [ingestion_settings.py](../rag3/ingestion_settings.py)의 현재 기본값은 청크 1,024토큰, 겹침 128토큰, BGE 배치 8개, 청크 그룹 32개다. 파싱된 텍스트·문서 ID·원본 해시·가능한 페이지 정보가 Qdrant에 먼저 staged 상태로 기록된다.
5. 공용 GPU 서비스가 청크 dense 벡터를 계산한다. Qdrant가 같은 내용으로 native BM25 sparse 벡터를 생성한다. 청크가 ready로 전환된 뒤 소스별 dense/BM25/ready 개수를 검증해야 작업을 완료로 표시한다.
6. SQLite의 작업·진행 기록과 Qdrant의 staged/ready 포인트가 재개 근거다. 서버 중단 뒤 작업은 자동 중복 실행하지 않고 확인 가능한 경우 PAUSED로 남긴다. 원본 해시, 파이프라인 식별자, 청크 수와 포인트 상태가 맞지 않으면 묵시적 복구를 거부한다.

GPU 4 GiB 환경 때문에 파서와 임베딩 모델을 동시에 계속 올리지 않는다. 8페이지 이상 PDF의 기본 경로에서는 BGE 작업을 비우고 모델을 내려 CUDA 파서 자식을 실행한 후 종료시킨다. 다음 임베딩이나 의미 검색이 BGE를 다시 로드한다. 짧은 PDF와 다른 구조화 형식은 상주 CPU 파서가 처리한다. 이 임계값은 RAG3_CUDA_PARSE_MIN_PAGES로 조정한다. CUDA 파서가 GPU를 점유하는 동안 의미 검색은 대기하거나 오류를 반환할 수 있고, BM25 전용 검색은 BGE가 없어도 실행할 수 있다.

## 5. 검색·읽기 구현

- [retrieval.py](../rag3/retrieval.py)는 BGE-M3 dense 질의와 Qdrant native BM25 질의를 선택적으로 실행한다. 두 방법의 융합은 Qdrant RRF로 수행하며 동일 점수는 포인트 ID로 안정 정렬한다. 메타데이터 필터는 후보 검색 단계에 적용된다.
- UI와 MCP는 같은 검색 경로를 호출한다. 모든 MCP 검색은 비어 있지 않은 `collections` 배열을 반드시 받아 컬렉션별로 독립 검색한다. `rag3_search_flexible`은 후보 수와 메타데이터 필터도 받고, 결과의 컬렉션 이름을 `rag3_fetch`에 전달해야 한다.
- [rag3-mcp-server.py](../scripts/rag3-mcp-server.py)는 검색 결과를 출처 정보와 청크 ID로 정리한다. rag3_fetch는 한 청크를 문자 오프셋으로 이어 읽게 한다. 검색 발췌만으로 원문 전체를 읽었다고 판단하지 않는다.
- [collections.py](../rag3/collections.py)는 컬렉션·문서 목록, 문서 조회·삭제, 장절 메타데이터 보정을 담당한다. 장절 보정은 기본 dry run이고 실제 쓰기 전에 대상 포인트 소속을 검증한다.
- [sources.py](../rag3/sources.py)는 컬렉션·문서 ID에서 관리 원본을 역참조하고 파일 경로·해시를 검증한다. rag3_get_source는 사용자용 원본 다운로드 정보를, rag3_get_page는 PDF 실제 페이지의 이미지를 제공한다.

## 6. 모듈별 유지보수 지도

| 영역 | 핵심 파일 | 구현 책임과 수정 시 확인점 |
| --- | --- | --- |
| 진입점·MCP | [rag3-mcp-server.py](../scripts/rag3-mcp-server.py) | Streamable HTTP 기동, 도구 스키마, 검색·원본 응답, UI 명령 라우팅. 도구 변경 시 로컬·원격 클라이언트 모두 확인. |
| 백엔드 경계 | [backend_boundary.py](../rag3/backend_boundary.py) | 단일 인스턴스 잠금, 세션별 컬렉션, 요청 시간 제한, /health, 로컬 UI 토큰 API. |
| 수집 스케줄러 | [ingestion_backend.py](../rag3/ingestion_backend.py), [ingestion_manager.py](../rag3/ingestion_manager.py) | 단일 실행 슬롯, 단계 전환, 일시정지·재개·취소·복구, CPU/CUDA 파서와 GPU 서비스 조정. 이전 manager의 인터페이스를 현 백엔드가 재사용한다. |
| 작업 저널 | [ingestion_jobs.py](../rag3/ingestion_jobs.py), [mcp_jobs.py](../rag3/mcp_jobs.py) | SQLite WAL·진행 ID·상태 조회; MCP 입력을 공용 작업으로 변환. DB 스키마 변경 시 복구 경로 확인. |
| 파싱 | [persistent_parser.py](../rag3/persistent_parser.py), [ingest-worker.py](../scripts/ingest-worker.py), [windows_job.py](../rag3/windows_job.py) | 상주 CPU 파서/임시 CUDA 파서의 자식 수명, Docling 실행, Windows 자식 프로세스 회수. |
| 수집 파이프라인 | [haystack_native.py](../rag3/haystack_native.py), [ingestion_light.py](../rag3/ingestion_light.py), [ingestion_settings.py](../rag3/ingestion_settings.py) | 문서 변환·청크 정규화·원본 보존·평문 직행·파이프라인 식별자. 청크 설정을 바꾸면 기존 체크포인트 호환성도 검토. |
| 임베딩·GPU | [gpu_model_service.py](../rag3/gpu_model_service.py), [bge_dense.py](../rag3/bge_dense.py), [retrieval_embedding.py](../rag3/retrieval_embedding.py) | BGE 모델 소유, 질의/문서 큐, 기한·취소, CUDA 파서에 GPU 양도. |
| 벡터 저장·검색 | [qdrant_store.py](../rag3/qdrant_store.py), [qdrant_bm25.py](../rag3/qdrant_bm25.py), [qdrant_clients.py](../rag3/qdrant_clients.py), [retrieval.py](../rag3/retrieval.py) | 컬렉션 스키마·BM25 설정·클라이언트 재사용·dense/BM25/RRF 검색. |
| 원본·첨부 보안 | [attachments.py](../rag3/attachments.py), [sources.py](../rag3/sources.py), [source_identity.py](../rag3/source_identity.py), [source_guard.py](../rag3/source_guard.py), [source-viewer.html](../rag3/source-viewer.html) | 첨부 다운로드 검증, SHA-256, 관리 경로 검증, 원본 부분 전송·PDF 페이지 렌더와 MCP UI 다운로드 카드. 카드는 모델 대화 첨부가 아니다. |
| 데스크톱 | [ingest-ui.py](../scripts/ingest-ui.py), [ui_backend_client.py](../rag3/ui_backend_client.py), [ui_state.py](../rag3/ui_state.py), [Program.cs](../launcher/Program.cs) | PySide6 화면·상태, 비동기 인증 HTTP 클라이언트, 선택적 단일 창 실행기. 쓰기 타임아웃 뒤 자동 재제출하지 않음. |
| 기동·연결 | [rag3-backend-login.py](../scripts/rag3-backend-login.py), [manage-rag3-login.ps1](../scripts/manage-rag3-login.ps1), [ensure-infra.ps1](../scripts/ensure-infra.ps1), [rag3-tunnel-autostart.py](../scripts/rag3-tunnel-autostart.py), [ensure-rag3-codex-plugin.py](../scripts/ensure-rag3-codex-plugin.py), [mcp.json](../plugins/rag3/mcp.json) | Windows 로그인 작업, WSL Docker Qdrant 확인·기동, 저장된 터널 재연결, 로컬 Codex 플러그인 설치·주소. |

## 7. 데이터·설정·보안 경계

| 항목 | 위치·성격 |
| --- | --- |
| SQLite 작업 저널 | data/ingestion_jobs.sqlite. WAL을 사용하므로 백업 시 DB 파일 하나만 복사하지 말고 SQLite 온라인 백업 방식으로 보존한다. |
| 관리 원본 | data/sources/해시/ 아래. Qdrant 스냅샷만으로는 원본 다운로드를 복원할 수 없다. |
| MCP 다운로드 작업 | data/mcp_jobs/ 아래 임시·진행 파일. 중단된 서명 URL 다운로드는 새 첨부 참조로 재제출한다. |
| Qdrant 데이터 | rag3_qdrant_data Docker 볼륨. 코드 체크아웃과 별개로 백업·이관한다. |
| 모델 | data/models/ 아래 로컬 BGE-M3·Docling 파일. Git에 포함되지 않는다. |
| 로그인 설정 | data/backend-login-config.json. 포트·Qdrant 주소·백엔드 전체 쓰기/삭제 정책 등 비밀값 없는 설정이다. |
| 터널 자격 증명 | 사용자별 로컬 저장 프로필. 저장소·문서·로그에 포함하지 않는다. |

HTTP 서버 기본 바인딩은 127.0.0.1이다. UI 명령 API는 백엔드가 발급하고 재시작 시 교체하는 X-RAG3-Token을 사용한다. MCP는 이 UI 토큰을 사용하지 않으며 RAG3_MCP_WRITE_ENABLED와 RAG3_MCP_DELETE_ENABLED가 **백엔드 전체** 쓰기·삭제 권한을 제어한다. 수동 서버 실행 시 코드 기본값은 둘 다 허용이므로 원격 연결 전에 정책을 명시적으로 확인해야 한다. 세션 없는 MCP 연결의 레거시 컬렉션 선택은 서버 프로세스 전체에 적용되지만 검색·원본 호출은 이 선택을 무시하고 컬렉션 이름을 명시적으로 요구한다. Compose의 Qdrant 포트는 현재 호스트 IP 제한 없이 게시되므로 네트워크 노출을 별도 점검한다.

## 8. 시작·검증·인계 시 순서

1. [Windows 설치 안내](install-windows.md)를 따라 의존성, 모델, WSL2 Docker, Qdrant를 구성한다. 현재 인프라 스크립트는 C:\python_program\RAG3 경로를 사용한다.
2. [사용법](usage.md)에 따라 백엔드와 UI를 실행하고 http://127.0.0.1:8000/health에서 올바른 백엔드 PID와 상태를 확인한다. UI 창을 닫아도 백엔드는 계속 실행된다.
3. Codex 로컬 플러그인과 ChatGPT 웹 터널은 각각 별도로 연결한다. ChatGPT 웹 연결에는 사용자 자신의 터널·워크스페이스 접근권한이 필요하다.
4. [사용례](examples.md)로 임시 컬렉션을 만들고 검색한다. 전체 수집 회귀 확인은 [검증 스크립트 안내](verification.md)를 따른다.
5. 운영 데이터 변경이나 실서버 테스트 전에 활성 수집 작업을 확인한다. 시험 서버에는 별도 RAG3_TEST_DATA_ROOT와 RAG3_BACKEND_TEST_INSTANCE를 자식 프로세스까지 전달하고 임시 Qdrant 컬렉션을 사용한다. 운영 SQLite를 공유한 채 테스트 서버를 띄우면 복구 처리가 운영 작업을 변경할 수 있다.
6. 백업·이관에는 SQLite 온라인 백업, Qdrant 스냅샷/볼륨, 관리 원본, 모델 파일을 함께 계획한다. 비밀키는 별도로 안전하게 이전한다.

## 9. 변경할 때 영향 범위

- **입력 형식 추가:** [constants.py](../rag3/constants.py)의 MCP 허용 확장자, 첨부 검증, UI 파일 선택, Docling/평문 분기와 원본 MIME 제공 경로를 함께 확인한다. 지원 형식을 UI와 MCP에 동일하다고 가정하지 않는다.
- **청크·임베딩·컬렉션 스키마 변경:** 파이프라인 식별자, Qdrant 벡터 이름·차원, 기존 컬렉션 호환성, 재개 체크포인트를 묶어서 설계한다. 기존 적재 데이터를 조용히 새 스키마로 읽지 않는다.
- **MCP 도구 변경:** 서버 도구 스키마, 읽기/쓰기/삭제 권한, 로컬 Codex 플러그인과 ChatGPT 웹 플러그인의 새 연결 상태를 각각 확인한다. 새 서버 코드가 이미 열린 AI 대화의 도구 정의를 즉시 갱신한다고 가정하지 않는다.
- **장애 복구·백업 변경:** SQLite 저널, Qdrant 포인트, 관리 원본의 일관성과 중단 후 수동 재개 동작을 함께 검증한다. UI 쓰기 요청의 타임아웃은 실패 확정이 아니므로 상태를 먼저 다시 조회한다.

## 10. 확인된 결과와 남은 일

- 2026-10-25 기록의 한 로컬 실험에서 535페이지 PDF를 252청크로 색인하는 데 30.7분이 걸렸고 dense·BM25·hybrid 검색 모두 원본을 찾았다. 4 GiB Quadro T1000 장치 전체의 최대 사용량은 3,873 MiB였다. 한 장치·문서의 측정값이며 일반 성능 보증이 아니다.
- 공용 백엔드, UI, 로컬 Codex MCP, 기존 터널을 통한 ChatGPT 웹 읽기 경로는 개발 PC에서 확인됐다. **다른 PC의 빈 환경에 처음부터 설치하는 전 과정은 아직 검증하지 않았다.**
- RAG3에서 ChatGPT로 원본 PDF를 보내는 실험은 호스트 저장·재다운로드·해시 일치까지만 성공했다. Library UI에서는 파일을 찾지 못했고, 현재 대화 첨부 등록과 모델의 원본 파일 직접 읽기는 실패했다. 내부 Library 등록 여부는 미확인이다. [상세 실험](chatgpt-file-handoff.md).
- 유지보수 우선순위는 새 PC 설치 재현, 하드코딩된 인프라 경로 제거, Qdrant 포트 노출 검토, 설치/운영 백업 절차 검증, 변경 전후 동일 조건의 성능 기준 확보다. 기능 추가 시 수집 체크포인트 호환성과 원본 해시·출처 정보 보존을 먼저 확인한다.

소스 이용은 [MIT 라이선스](../LICENSE)를 따른다.

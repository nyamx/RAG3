# RAG3 공용 백엔드 Stage C 구현 지시서

대상: `C:\python_program\RAG3`에서 **Stage C를 구현할 후임 AI 개발자**. 설계안을 다시 작성하는 일이 아니라 코드 수정, 격리 검증, 진행 기록과 커밋까지 수행한다. 이 문서는 [전체 전환 지시서](RAG3_SHARED_BACKEND_IMPLEMENTATION_TASK.md)의 C 단계 실행본이다. 먼저 [진행 기록](RAG3_SHARED_BACKEND_PROGRESS.md)과 실제 HEAD를 대조한다. 기준 브랜치는 `codex/shared-backend`이며 작성 시점 HEAD는 `b62a2a8`(Stage B 보완)이다. 그 뒤의 변경이 있으면 실제 코드를 우선 조사하고 차이를 기록한다.

## 1. 이번 단계의 완료 범위

- **백엔드 HTTP 프로세스 하나가 적재 스케줄러, 작업 상태, 문서 임베딩, MCP/API 검색을 소유**한다. 별도 `ingestion_manager` 서버와 전체 적재/임베딩 worker 프로세스는 새 실행 경로에서 제거한다. 무거운 Docling 파싱에 한해 동시에 최대 1개의 임시 **CPU 전용** 자식 프로세스를 허용한다. Qdrant는 별도 서버다.
- 백엔드의 MCP 검색과 인증된 `/api/v1/command` 검색, 적재 임베딩은 **Stage B의 동일한 `GPU_BGE_SERVICE` 객체**를 사용한다. 이 경로에서 다른 BGE 모델을 생성하거나 CPU 사본을 만들지 않는다. BM25 전용 검색은 BGE를 요청하지 않는다. 실제 UI 검색 worker·컬렉션 worker와 플러그인/터널/로그인 기동의 최종 전환은 D 단계다.
- SQLite의 기존 작업 ID·목록·상태, 제출·Run·Pause·Resume·Retry·Restart·Cancel 의미, 첨부 다운로드, 원본, 진행 기록, Qdrant의 `parsed` 선기록과 점진적 `ready` 검색 가능성을 보존한다. 활성 적재 문서는 최대 1개다. 작업 처리 자체는 스케줄러에 제출된 사실을 통지해 깨우며 상시 0.25초 폴링에 의존하지 않는다.
- **C의 통과는 백엔드 MCP/API 경로의 적재 중 검색**으로 판정한다. UI 창에서의 실제 제출·검색 동등성, 설치 플러그인과 터널 연결 검증은 D의 통과 조건이다. 다만 C 중에도 구 UI의 작업 버튼이 두 번째 스케줄러를 자동 기동해서는 안 된다. 아래의 임시 호환 경로를 구현한다.
- 이미 통과한 A/B 전체 시험을 다시 수행하지 않는다. C에서 바꾼 계약의 국소 회귀, 새 동시성/복구 시험, 실제 GPU/Qdrant 통합 시험을 수행한다. Stage B의 모델 큐·인코딩 로직은 변경 필요가 입증되지 않으면 그대로 둔다.

## 2. 구현 전 반드시 확인할 현행 경로

| 코드 | 현행 동작과 C의 변경점 |
| --- | --- |
| `rag3/ingestion_manager.py` | `client()`가 연결 실패 시 `python -m rag3.ingestion_manager --serve`를 **자동 기동**한다. `Scheduler.__init__()`는 즉시 `recover()`로 DB를 수정하고 폴링 스레드를 시작한다. `start()`는 `scripts/ingest-worker.py` 전체 worker를 띄운다. 이를 백엔드 소유 인스턴스 하나와 명시적 시작 순서로 바꾼다. 조사 목적으로 `client('health')`를 부르지 말 것: 관리자 자동 기동을 유발한다. |
| `scripts/ingest-worker.py` | coordinator가 파싱 자식과 임베딩 자식을 각각 띄운다. `parse_source()`는 32청크씩 `parsed`를 선기록하며 Docling 호출의 `pdf_cpu`가 기본값이다. `embed_source()`는 32개를 직접 BGE에 보내고 `update_vectors`와 `set_payload`를 따로 호출한다. 새 경로는 backend 내부에서 임베딩하고 Docling 파서 자식만 둔다. |
| `rag3/ingestion_jobs.py`, `rag3/ingestion_settings.py` | 작업과 `ingestion_progress(identity, phase, total_chunks)`는 **SQLite**에 있다. Qdrant에는 청크의 `parsed`/`ready` 상태가 있다. `PIPELINE_SCHEMA_VERSION`, SHA, 컬렉션, 분할 길이·겹침, 모델 경로가 progress identity에 들어간다. 이 둘을 혼동하지 말고 기존 체크포인트를 검증한다. |
| `rag3/mcp_jobs.py`, `scripts/rag3-mcp-server.py` | `JobManager`와 `_ui_command('jobs.*')`가 관리자 RPC를 사용한다. 백엔드 `GPU_BGE_SERVICE`는 생성되어 있으나 MCP/API 검색은 `rag3.retrieval.search()`를 통해 다른 BGE 캐시를 사용한다. 두 경로 모두 백엔드 소유 객체로 연결한다. |
| `rag3/retrieval.py`, `scripts/ingest-ui.py` | `search()`의 의미 검색은 `_QUERY_EMBEDDERS`의 직접 모델을 만든다. UI는 D 전까지 관리자 `client()`를 직접 사용하고 검색 worker도 별도다. C에서 UI의 **작업 제어 호출만** 새 백엔드로 프록시해 중복 관리자 자동 기동을 막는다. UI 검색 worker의 완전 전환은 D다. |

Stage B 서비스 계약은 `rag3/gpu_model_service.py`와 진행 기록에 있다. `submit_query()`/`submit_documents()`는 `GPURequest`를 반환하고, 다른 스레드에서 `.result()`를 기다릴 수 있다. 모델 소유 GPU 스레드의 Future 완료 콜백에서 같은 서비스의 다음 결과를 동기 대기하지 않는다. 서비스가 내부적으로 32개를 8개씩 나누더라도 **요청 결과는 전체 요청이 끝날 때** 나온다. 따라서 적재는 최대 8개를 **요청 한 건**으로 제출해 결과마다 취소 확인·Qdrant upsert·진행 기록을 수행한다.

## 3. 데이터 보호와 전환 순서

1. `git status`/HEAD와 실행 중인 백엔드·구 관리자·worker·적재/다운로드 작업을 확인한다. 실제 작업을 임의로 중단하지 않는다. 관리자 포트 `127.0.0.1:18747`을 단순 HTTP 호출로 조사하거나 `client()`를 호출해 새 관리자를 띄우지 않는다. 작업 목록·SQLite `ingestion_progress`·체크포인트·원본 경로와 대상 Qdrant 컬렉션 스키마를 **읽기 전용**으로 목록화한다.
2. 코드 전환 전 SQLite WAL을 고려한 **SQLite 온라인 백업**을 만들고 백업본에서 `integrity_check`, 작업/진행 기록 개수 및 복구 열기 시험을 한다. DB 파일만 무작정 복사하지 않는다. 실제 Qdrant 컬렉션·원본·기존 미추적 산출물은 삭제/정리하지 않는다. 시험은 별도 임시 SQLite DB와 새 임시 Qdrant 컬렉션/소스만 사용한다.
3. 구 관리자가 현재 작업 또는 첨부 다운로드를 소유하면 완료/안전한 일시정지 전까지 컷오버하지 않는다. 유휴 관리자 종료와 포트/자식 프로세스 해제를 **확인한 뒤** 새 스케줄러를 시작한다. 확인할 수 없으면 두 스케줄러를 동시에 가동하지 말고 전환 실패를 명시한다. 기존 관리자 포트의 단독 바인딩과 새 백엔드 잠금은 서로 다른 보호 장치이므로 기존 포트만으로 이중 소유를 막았다고 판단하지 않는다.
4. **최초 컷오버**에서는 위 백업과 구 관리자 종료를 확인한다. 이후 백엔드 시작은 **소유권 확보 → 한 번의 recovery → 제출 수락/스케줄러 실행** 순서로 처리한다. 임포트나 health만으로 `Scheduler()`를 생성해 SQLite를 변경하지 않는다. 재시작 직전부터 있던 `QUEUED`만 `requested=0`으로 바꾸고, 기동 뒤 들어온 새 MCP 제출은 `requested=1`, UI 제출은 `requested=0`을 유지한다. 둘이 섞여 초기화에 의해 새 작업이 꺼지는 경합을 막는다.
5. 수정은 적재 scheduler/파서/임베딩/검색 연결 단위로 작게 나누고 격리 시험 후 합친다. 현재 `run_job()`/`parse_source()`의 `client.delete()`는 동일 SHA의 포인트를 삭제할 수 있다. 검사 중 또는 실제 컬렉션 시험에서 호출하지 않는다. Restart나 PARSING 재개에서 삭제가 필요한 경우 해당 작업·컬렉션·SHA의 범위와 사용자 명시적 동작을 검증하고 기존 `ready` 완료 문서를 지우지 않는다.

## 4. 구현 계약

### 스케줄러와 상태

- `rag3/ingestion_manager.py`의 제출·다운로드·중복 SHA 검사·작업 제어·완료 검증에서 필요한 로직을 재사용/분리해 **백엔드 내부 스케줄러 하나**가 소유한다. 전체 worker를 자식으로 띄우거나 구 manager RPC를 백엔드 내부에서 호출하지 않는다. 큐 선택은 `status='QUEUED' AND requested=1 ORDER BY created_at,job_id`; 동시 활성 문서는 1개. 제출/Run/Resume/Retry/다운로드 완료 때 이벤트로 깨운다. 유휴 시 무한 바쁜 루프를 만들지 않는다.
- `JobManager`의 공개 MCP 도구 이름·필수 인자·주요 반환 필드와 인증된 UI API `jobs.*` 계약은 유지하고, 이 호출들을 같은 내부 스케줄러/SQLite에 연결한다. 구 UI가 D 이전에 직접 부르는 `ingestion_manager.client()`는 **관리자 자동 기동 없는 얇은 백엔드 API 프록시**로 바꾸거나 UI 호출부의 작업 제어만 백엔드 API로 연결한다. 후자가 적다면 선택할 수 있으나, 기존 UI의 Submit/Run/Pause/Resume/Cancel을 C 기간에 조용히 끊지 않는다. UI의 검색/컬렉션 worker 전환은 D로 남긴다. 백엔드가 없을 때 백엔드 미기동 오류를 표시하고 다른 스케줄러를 만들지 않는다.
- `recover()`와 정상 완료 시 정리는 분리한다. recovery가 `COMPLETE`의 progress/원본 사본을 무심코 정리하지 않게 하고, 작업별 검증 성공 뒤에만 기존 완료 정리 규칙을 적용한다. 실패/중단 시 원본·DB 작업·progress와 임시 파싱 결과를 재개 가능성에 따라 보존한다. 작업 범위가 확인된 임시 파일만 정리한다.

| 기동 전 상태 | 새 백엔드 기동 시 처리 |
| --- | --- |
| `RUNNING`, `RESUMING`, `PAUSING`, `CANCELLING` | 원본과 identity를 포함한 체크포인트가 유효하면 `PAUSED`, 없으면 `FAILED`; `requested=0`, 자동 실행 금지. 사유 기록. |
| `DOWNLOADING`, 원본 SHA 없는 `PAUSED` | `FAILED`, 기록 보존, 사용자가 첨부 원본 참조를 다시 제출. |
| 기동 **전** `QUEUED` | 작업 기록 유지, `requested=0`; 명시적 Run 이후 시작. 기동 **후** 신규 MCP/UI 요청은 각각 즉시 실행/대기 정책 유지. |
| 유효한 `PAUSED` | 원본·progress·포인트 보존; 명시적 Resume만 허용. |
| `FAILED`, `CANCELLED` | 기록·원본 보존; 명시적 Retry/Restart만 허용. |
| `COMPLETE` | 변경·재적재하지 않음. 기존 검색/다운로드 호환성 유지. |

`checkpoint_available()`의 phase 존재만으로 **identity·원본 SHA·Qdrant 상태까지 유효하다고 간주하지 않는다.** 호환되지 않는 미완료 체크포인트는 기록·원본·기존 포인트를 보존하고 이유를 노출한다. `PIPELINE_SCHEMA_VERSION`을 관성적으로 올려 모든 작업을 무효화하거나, 맞지 않는 identity를 자동 덮어쓰기/삭제하지 않는다. 실제 파싱·청킹 의미가 바뀌었다면 호환 여부를 코드와 격리 데이터로 입증하고, 불가능한 작업만 명시적으로 중단한다. 수동 Restart는 해당 작업의 삭제/재처리 범위와 영향을 호출자에게 명확히 알린 뒤 수행한다.

### CPU 파서와 GPU 임베딩

- TXT/JSON/JSONL/TSV 등 현재 plain text 경로는 백엔드에서 처리한다. 현재 helper가 `rag3/haystack_native.py`의 Docling 등 무거운 최상위 import를 동반하므로 필요한 가벼운 분할·원본 보존·정규화 helper를 분리하거나 동등하게 지연 로드해 서버 기동/health와 단순 텍스트 적재가 불필요한 Docling/OCR을 로드하지 않게 한다. PDF/Office/이미지 등 Docling이 필요한 형식은 별도 CPU 파서 자식 **최대 1개**에서만 처리하고 `build_parsing_pipeline(pdf_cpu=True)` 및 자식의 GPU 비가시 환경을 명시한다. CPU 파서는 BGE/torch CUDA 모델을 만들지 않는다. Windows Cancel은 현행 `taskkill /F /T`와 같은 하위 프로세스 트리 종료를 보장한다. 파싱 출력·정규화·청크 ID/메타데이터·32개 `parsed` 선기록은 기존 의미를 유지한다.
- 백엔드 임베딩 단계는 `parsed` 포인트를 읽고 **최대 8개씩** `GPU_BGE_SERVICE.submit_documents(texts)`로 보낸다. GPU owner 밖에서 Qdrant I/O를 한다. 다음 8개 제출 전과 결과 upsert 직전에 Pause/Cancel/deadline을 재확인한다. GPU 호출 중 Cancel은 곧바로 요청을 접수하지만 최대 8개 호출은 비선점이다. `GPURequest.future`가 취소 오류로 일찍 끝나더라도 실제 CUDA 호출 종료 전에는 새 적재 작업 시작/완료 확정이나 결과 쓰기를 하지 않는다. 완료 후 다음 배치 0개, 취소된 결과의 Qdrant upsert 0회를 검증한다. 이미 Qdrant 쓰기가 시작된 뒤 도착한 Cancel은 그 쓰기를 강제 중단할 수 없으므로 실제 완료 상태를 기록해 응답한다. Pause는 유효한 체크포인트를 남겨 재개할 수 있게 한다.
- 첫 질의/적재 요청에만 모델을 로드하고 이후 같은 GPU 모델을 유지한다. 임베딩 실패/CUDA OOM은 작업에 명시적 오류를 남기고 CPU BGE로 자동 전환하지 않는다. 검색 기한과 백엔드 API의 90초 timeout은 **실행 중인 CUDA 호출을 중단한다는 뜻이 아니다**. 타임아웃 후 쓰기 작업의 상태는 SQLite에서 확인 가능해야 하며 취소로 잘못 보고하지 않는다.

### Qdrant 기록과 백엔드 검색

- `parsed` 선기록은 유지한다. 교체 대상 `parsed` 포인트를 `with_payload=True, with_vectors=True` 등으로 읽고, `point.id`, **전체 payload**의 `content`, 중첩 `meta` 및 기타 필드를 복사한 뒤 `meta._rag3_ingest_state`만 `ready`로 바꾼다. 현행 `text-dense`와 `bm25`(`models.Document` 입력 및 현재 컬렉션 옵션)를 포함해 **완성 포인트 한 번의 upsert**로 기록한다. 기존 컬렉션 스키마와 실제 대상 포인트의 추가 named vector(`text-sparse` 등)를 확인·보존한다. 추가 벡터를 정확히 보존할 수 없으면 해당 포인트의 쓰기를 중단하고 오류를 표시한다. `ready` 포인트는 건드리지 않고 BGE sparse를 새로 계산하지 않는다. 기존 source/dense/BM25/ready 개수 검증과 결과 필드 계약을 유지한다.
- 백엔드 MCP/API의 semantic/hybrid 검색은 Stage B 서비스가 만든 **query 벡터를 기존 Qdrant 검색·필터·RRF 로직에 공급**한다. `rag3/retrieval.py`에 명시적 벡터/인코더 주입 같은 작은 경계를 둘 수 있지만, 백엔드에서는 `_QUERY_EMBEDDERS`·`BGEM3OnePassEmbedder`를 호출하지 않는다. 순수 BM25는 서비스 요청 0회. 기존 컬렉션 선택·세션 격리·top_k·metadata filter·랭크 융합·ID/결과/오류 계약을 유지한다. UI API `search`와 MCP 검색 모두 같은 백엔드 경로를 호출한다. UI 프로세스가 D 이전에 별도 검색 worker를 유지하는 것은 C의 백엔드 단일 모델 합격 범위에 포함하지 않고 진행 기록에 명확히 쓴다.

## 5. 실행·합격 시험

| 시험 | 합격 기준 |
| --- | --- |
| 소유권/콜드 기동 | 테스트 DB에서 백엔드 1개·스케줄러 1개, 별도 관리자/전체 worker/임베딩 자식 0개. health/import에서 BGE·Docling 로드 0회. 유휴 manager 공존 차단, 백엔드 중단 시 구 UI 호출이 관리자를 자동 기동하지 않음. |
| 상태 전환 | 위 상태 표를 **기동 전 자료**로 재현하고 기동 후 신규 MCP/UI 제출을 구별한다. 유효·무효 identity, SHA 변경, 누락된 원본, `PARSING`/`CHUNKS_STORED`/`EMBEDDING` 각각 검사. 이전 `QUEUED`만 요청 해제, 신규 MCP 즉시 실행, 신규 UI Run 대기. Resume/Retry/Restart와 중복 SHA 제한 유지. |
| CPU/GPU 경계 | Docling 파서 자식에서 GPU 비가시·CPU 설정, 파서 최대 1개, 자식 트리 Cancel 확인. 적재와 검색이 백엔드의 같은 BGE 객체를 사용하고 실제 모델 생성 1회, CPU BGE 0개, 적재 중 GPU 상주. 실제 GPU 부재/OOM은 명시적 실패, BM25는 독립. |
| 선점/취소 | 17개 청크는 적재 요청 `8→8→1`; 첫 8개 진행 중 질의가 오면 다음 GPU 호출은 질의. 현재 호출 중 Cancel/기한 만료 후 추가 배치·해당 결과 upsert 0회. Pause/Resume 뒤 중복/누락 없이 완료. |
| 임시 Qdrant 컬렉션 | `parsed` 선기록 뒤 처리된 8개씩 검색 가능. 완성 upsert 전후 point ID, 전체 payload와 섹션 정보, dense/BM25, `ready`를 비교. 구 스키마의 추가 벡터 보존 또는 명시적 중단; 기존 ready 포인트 불변. 최종 source/dense/BM25/ready 개수 및 검색 결과 검증. 실제 사용자 컬렉션은 수정하지 않음. |
| 백엔드 MCP/API | 동일 임시 컬렉션에 적재 중 semantic·BM25·hybrid 검색 가능. MCP와 인증 UI API 검색의 ID·필터·순위·주요 오류 의미가 일치; BM25-only BGE 요청 0회. 첨부, job list/status, Pause/Cancel/Resume/Retry의 주요 반환 필드/권한 회귀. UI 창의 실제 검색·제출 비교는 D. |
| 장애/재기동 | 활성 작업 중 백엔드 비정상 종료 후 새 기동에서 원본·SQLite progress·기존 포인트 보존, `PAUSED`/`FAILED` 전환 및 수동 재개. 새 작업 자동 재개 없음. 일관된 SQLite 백업본 복구 가능. Qdrant 연결 오류 시 작업 상태가 무한 재시도되지 않음. |

테스트는 fake GPU/시계/차단 이벤트와 임시 DB·컬렉션으로 결정적으로 먼저 수행한다. 그 뒤 이 PC에서 실제 GPU와 Qdrant를 사용하는 **짧은 격리 통합 시험 한 번**으로 모델 1회 로드, 임시 적재 중 MCP/API 검색, 점진 upsert, CPU 파서를 확인한다. 실제 GPU 모델 시험이 불가능하면 이유와 미검증 항목을 남기고 Stage C를 PASS로 표시하지 않는다. 기존 A/B 시험은 변경한 공유 경계에 문제가 나타날 때 필요한 부분만 추가 실행한다. `py_compile`, 관련 시험, `git diff --check`도 확인한다.

## 6. 인계와 종료 조건

`RAG3_SHARED_BACKEND_PROGRESS.md`에 C 구현 커밋·변경 파일, 실제 상주/임시 프로세스 수, 스케줄러/파서/검색 연결 경로, 상태 전환 및 데이터 보존 결과, 시험 명령과 PASS/FAIL, 실제 모델 로드 횟수·배치 순서·검색 대기/처리 시간·RAM/VRAM을 기록한다. 측정하지 않은 값은 **미측정**으로 쓴다. 기존 작업/컬렉션을 건드리지 않았는지와 UI 직접 검색 worker 등 D에 남은 경로를 분명히 적는다. C 코드와 기록만 범위를 지정해 별도 커밋한다. 기존 미추적 자료는 커밋하거나 정리하지 않는다.

완료 보고에서 **백엔드 MCP/API의 단일 BGE 검증**과 **최종 UI/플러그인 전환 미완료**를 구분한다. 기능·데이터 보존·실제 GPU/Qdrant 합격 시험 중 하나라도 빠지면 C 완료라고 하지 말고 재현 방법과 다음 AI의 첫 작업을 인계한다.

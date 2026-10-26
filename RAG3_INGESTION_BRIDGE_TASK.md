# RAG3 ingestion 개선 — Bridge 작업지시서

> 이 문서는 2026-10-20 당시의 작업 계획을 기록한 것입니다. 현재 구현은 `rag3/ingestion_jobs.py`의 SQLite 진행 상태와 Qdrant 청크를 사용하며, 아래에 언급된 `snapshot.json`, `manifest.json`, `rag3/checkpoint.py` 방식은 제거되었습니다. 현재 동작은 `MCP.md`를 참고하세요.

## 요청

`C:\python_program\RAG3`의 UI·MCP ingestion 작업을 **하나의 작업 목록과 하나의 실행 대기열**로 관리하라. 기존 MCP 도구의 이름·필수 입력·주요 응답 필드와 ingestion 처리 방식은 가능한 유지한다. 새 프레임워크나 별도 작업 서버 제품은 도입하지 않는다. 사용자의 요청에 따라 **기존 Qdrant 컬렉션 `chatgpt_001`은 2026-10-20에 삭제되었고, 현재 컬렉션 목록은 비어 있다.** 같은 날 `data/checkpoints`, `parsed`, `chunks`, `embeddings`, `mcp_jobs`, `sources`의 기존 내용과 `data/ui-state.json`도 정리했다. 이번 전환은 새 작업 기록으로 시작한다. 구현 과정에서 구 컬렉션·구 데이터 삭제나 이관을 다시 수행하지 않는다.

## 완료 기준

1. UI와 MCP에서 등록한 문서별 작업은 공통 SQLite DB에 기록된다. UI를 다시 열어도 두 출처의 작업이 보이며, MCP가 제출한 작업의 상태 변화가 **5초 이내** UI에 반영된다.
2. UI와 MCP를 함께 사용해도 실행 중인 ingestion worker는 **기본 최대 1개**다. 나머지는 대기한다. CPU 스레드 사용을 제한하고, 실제 문서로 CPU·메모리 사용을 측정해 PC에서 다른 작업을 할 수 있는지 확인한다.
3. UI와 MCP의 Pause·Cancel·Resume은 같은 작업 ID를 대상으로 작동한다. Pause는 기존 안전한 체크포인트에서 멈추고, Cancel은 해당 worker를 즉시 종료한다. 둘 다 마지막 유효 체크포인트를 보존한다. 재개는 사용자가 요청할 때만 한다.
4. 완료 확인 후 작업 결과와 `COMPLETE`를 DB에 남기고, 해당 작업의 `snapshot.json`과 `manifest.json`은 **둘 다 삭제**한다. 미완료 작업의 유효한 체크포인트는 남긴다.
5. UI의 작업 표는 자동 갱신된다. 기존 컬렉션 `Refresh`와 `Refresh Contents` 버튼 및 버튼 사용 안내는 제거한다. 컬렉션·문서 목록도 필요한 시점에 자동 갱신된다.
6. 전환 후 `ui-jobs.json`과 MCP의 개별 `job.json`은 운영 중 더 이상 생성·갱신하지 않는다. 새 작업부터 공통 DB에 기록한다.

## 현재 코드에서 출발할 곳

| 파일 | 현재 역할 | 필요한 변경 |
|---|---|---|
| `scripts/ingest-ui.py` | `self.jobs`와 `data/ui-jobs.json`으로 UI 작업 관리, `QProcess` 실행, 수동 Refresh 버튼 | 공통 DB 조회·제출·제어, 5초 자동 갱신, 불필요한 버튼 제거 |
| `rag3/mcp_jobs.py` | `data/mcp_jobs/<job_id>/job.json` 관리, 작업마다 worker 즉시 실행 | 공통 DB와 실행 대기열 사용, 기존 도구가 공통 작업을 제어하도록 연결 |
| `scripts/rag3-mcp-server.py` | 제출·상태·Pause·Resume·Cancel MCP 도구 노출 | 공통 작업에 맞는 설명·응답, 작업 목록 도구 추가 |
| `scripts/ingest-worker.py` | 파싱→청킹→임베딩→Qdrant 저장·검증을 한 프로세스에서 순서대로 실행 | 기존 체크포인트를 유지하고 완료 신호·정리 순서 연결 |
| `rag3/checkpoint.py` | `snapshot.json`·`manifest.json` 저장 | 완료 후 pair 정리, 유효성 검사 |
| `rag3/haystack_native.py` | BGE-M3 내부 인코딩 batch 기본 8 | 실제 메모리 측정 후 필요하면 전체 chunk 처리 방식을 제한된 batch로 개선 |

`scripts/ingest-ui.py`의 `ui-jobs.json` 저장·복원 코드와 `scripts/test-ui-job-recovery.py`는 이미 존재하는 **미커밋 변경**이다. 새 DB 구현으로 교체할 때 이 변경을 확인하고, 다른 사용자 변경은 덮어쓰지 말라.

## 최소 구현 구조

### 작업 DB와 실행 관리자

- Python 표준 `sqlite3`로 `data/ingestion_jobs.sqlite`를 만든다. 작업별 최소 정보는 `job_id`, `origin(UI/MCP)`, `collection`, 원본 경로·SHA, `status`, `stage`, `result`, `error`, 생성·수정 시각, 실행 요청 여부다. 임베딩 벡터나 첨부파일 서명 URL은 DB에 넣지 않는다.
- UI와 MCP의 작업 등록·상태 조회·제어 요청을 **하나의 로컬 실행 관리자**로 모은다. 관리자는 worker를 기본 1개만 실행하고, 종료 후 요청된 다음 작업을 시작한다. UI 창이나 MCP 연결을 닫아도 관리자가 살아 있으면 진행 상태는 유지한다. 두 번째 관리자가 같은 작업을 실행하지 못하도록 단일 인스턴스만 허용한다. 첫 ingestion 전에 새 컬렉션을 생성·선택하도록 하고, 삭제된 컬렉션을 가리키는 UI/MCP 기본 선택값은 새 컬렉션으로 바꾼다.
- 관리자가 없으면 UI/MCP가 필요할 때 기동한다. 기동 실패나 제어 전달 실패는 사용자에게 오류로 표시한다. UI가 MCP 작업에 Pause/Cancel을 누를 때 자기 `QProcess`를 종료하지 말고 관리자에 해당 `job_id`를 전달한다. 첨부파일 다운로드도 관리자가 소유해 UI/MCP의 작업 제어가 일관되게 작동하도록 한다.
- 작업 상태는 `QUEUED`, `DOWNLOADING`, `RUNNING`, `PAUSING`, `PAUSED`, `RESUMING`, `CANCELLING`, `CANCELLED`, `FAILED`, `COMPLETE`로 통일한다. 요청 접수와 실제 종료를 구분한다. 관리자는 DB에서 **실행 요청이 표시된 대기 작업만** 시작한다. 관리자가 다시 시작할 때 이전 `RUNNING`은 유효한 체크포인트가 있으면 `PAUSED`, 없으면 `FAILED`로 표시하고 **자동 실행하지 않는다**. 과거 `QUEUED`의 실행 요청도 해제해 새 제출 때문에 과거 작업이 뜻밖에 시작되지 않게 한다.
- UI와 MCP에서 같은 SHA를 같은 컬렉션에 중복 실행하지 않는다. 현재 체크포인트 경로가 `data/checkpoints/<source_sha>/`로 컬렉션을 구분하지 못하므로, **다른 컬렉션의 같은 SHA에 미완료 체크포인트가 있으면 실행을 거부하고 이유를 표시**한다. 완료 후 pair가 삭제되면 다른 컬렉션에 제출할 수 있다. 경로 체계 전체 변경은 이번 작업에서 하지 않는다.

### Pause·Cancel·Resume

| 명령 | 동작 |
|---|---|
| Pause | `PAUSING` 표시 후 현재 worker의 **기존 체크포인트 경계**에서 `PAUSED`. 파싱·청킹 뒤 embedder 직전 또는 임베딩 뒤 writer 직전이다. 긴 단계 중에는 바로 멈추지 않을 수 있음을 UI/MCP에 표시한다. |
| Cancel | 대기 작업은 즉시 `CANCELLED`; 실행 작업은 해당 프로세스 종료 확인 뒤 `CANCELLED`. 마지막으로 완성된 checkpoint pair를 남긴다. 체크포인트 이전에 취소했다면 재시도는 처음부터다. |
| Resume | 체크포인트 pair와 `CheckpointIdentity`를 검증한 뒤 같은 작업 ID로 `RESUMING`. 유효한 pair가 없으면 재개 가능한 것처럼 표시하지 않는다. |
| Retry/Restart | UI의 기존 기능을 유지한다. Retry는 체크포인트가 있으면 재개, 없으면 처음부터 실행한다. Restart는 해당 작업의 처리 체크포인트만 폐기하고 다시 실행한다. |

worker의 현재 두 체크포인트 시점은 유지한다. **배치별 영속 체크포인트나 단계별 별도 OS 프로세스는 이번 범위에 넣지 않는다.** 컴퓨터 강제 종료 중의 완전한 복구나 재부팅 후 자동 실행도 요구하지 않는다.

### 완료와 파일 정리

1. worker가 Qdrant에 해당 소스의 결과가 실제로 기록됐는지 확인한다. 현재의 source point/BM25 검증은 유지한다.
2. 관리자가 worker 성공 결과와 정상 종료를 확인하고 DB에 `COMPLETE`와 결과를 저장한다. 종료 코드만 보고 완료로 처리하지 않는다.
3. DB 저장 성공 후 **그 소스의** `snapshot.json`과 `manifest.json`을 삭제한다. 삭제 실패 시 다음 관리자 시작 때 완료 작업의 남은 pair 정리만 재시도한다. 완료 전이나 DB 기록 실패 시 pair를 지우지 않는다.
4. `PAUSED`·`FAILED`·`CANCELLED`의 유효한 pair는 보존한다. 원본 `data/sources/<sha>/`와 Qdrant 문서는 체크포인트 정리 대상이 아니다.

현재 worker가 직접 manifest에 `COMPLETE`를 기록하는 부분과, UI/MCP가 완료 manifest로 성공을 판정하는 부분은 DB 기준으로 바꾼다. 저수준 `CheckpointStore` 저장·로드 시험은 실제 완료 정리와 구분한다.

## MCP 도구 변경: 최소 범위

| MCP 도구 | 지시 |
|---|---|
| `rag3_ingest_file`, `rag3_ingest_text` | 기존 입력 형식과 `job_id` 반환을 유지한다. 즉시 worker를 여는 대신 공통 대기열에 등록한다. 응답 상태는 실제 `DOWNLOADING` 또는 `QUEUED`로 표시한다. |
| `rag3_ingestion_status(job_id)` | DB에서 UI/MCP 작업 모두 조회한다. `status`, `stage`, `result`, `error`, `origin`, `checkpoint_available`을 반환한다. 기존 필드는 가능하면 유지한다. |
| `rag3_pause_ingestion`, `rag3_cancel_ingestion`, `rag3_resume_ingestion` | 기존 도구 이름과 `job_id` 인자를 유지하고 관리자에 명령을 전달한다. 요청 중 상태와 완료 상태를 구분한다. 기존 쓰기 권한 검사를 유지한다. |
| **신규 `rag3_list_ingestion_jobs`** | AI가 MCP 연결을 다시 시작한 뒤에도 job ID를 찾을 수 있게 한다. 읽기 전용, 최신 작업부터 정렬, 컬렉션·상태 필터와 결과 수 제한/페이지 이동을 제공한다. 원본 로컬 경로나 서명 URL은 노출하지 않는다. |

도구 설명에서 “MCP 작업만 조회”, “제출 즉시 실행”, “Pause가 즉시 완료” 같은 구 동작 설명을 삭제한다. `MCP.md`, `RAG3_PLUGIN_SETUP.md`, `plugins/rag3/skills/rag3/SKILL.md`의 도구 목록·사용 안내·개수도 실제 새 도구에 맞게 고친다. 새 Retry/Restart/대기열 관리 MCP 도구는 만들지 않는다. AI는 실패 작업의 원본을 다시 제출할 수 있고, UI의 Retry/Restart는 유지된다.

## UI 자동 갱신과 버튼 제거

- 5초 `QTimer`는 **작업 DB만** 조회한다. UI/MCP 작업을 함께 표시하고 각 행의 `job_id`, 출처, 컬렉션, 파일명, 단계, 상태를 보여준다. 표 갱신 시 선택 행과 스크롤 위치를 유지한다.
- 컬렉션 `Refresh`, `Refresh Contents` 버튼, 툴팁, enable/disable 코드와 “버튼을 누르라”는 문구를 제거한다. 내부 조회 함수는 자동 갱신에 필요하면 남긴다.
- 컬렉션 목록은 시작·생성·삭제·선택기 열기 때 갱신한다. 문서 목록은 해당 탭 진입·컬렉션 변경·작업 완료 때 갱신한다. 외부 MCP 변경을 위해 컬렉션 목록과 열린 문서 탭은 별도의 낮은 빈도로 갱신한다. **Qdrant 전체 목록을 5초마다 읽지 않는다.** 실패는 화면에 표시하고 다음 자동 갱신에서 재시도한다.

## 새 작업으로 전환하는 순서

1. 작업 시작 전에 실행 중인 ingestion·다운로드가 없는지 다시 확인한다. 구 작업 JSON·checkpoint·관리 원본은 이미 정리되어 **새 DB로 이관할 항목이 없다**. 삭제된 컬렉션을 대상으로 한 구 작업 ID와 snapshot은 재개할 수 없다. 새 코드가 구 `ui-jobs.json` 및 MCP `job.json` 경로를 다시 만들지 않게 한다.
2. 공통 DB와 단일 실행 관리자를 빈 작업 목록으로 시작한다. 새 컬렉션을 만들고 UI/MCP의 기본 선택값이 실제 존재하는 컬렉션을 가리키게 한다.
3. UI/MCP 제출·상태·제어를 관리자에 연결한다. MCP 작업 목록 도구를 추가하고 관련 설명 문서를 갱신한다.
4. worker 완료 후 DB 기록과 checkpoint pair 정리를 연결한다. **새로 생성한** 작업을 Pause한 뒤 기존 체크포인트 형식으로 Resume가 되는지 확인한다.
5. UI의 5초 작업 표시와 나머지 자동 갱신을 연결하고 필요 없는 Refresh 버튼·코드·문구를 제거한다.
6. 기본 동시 worker 1개와 CPU 스레드 제한을 검증한다. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`와 PyTorch 설정은 해당 라이브러리를 불러오기 전에 적용한다. 그래도 CPU가 PC 사용을 방해하면 worker 우선순위·코어 사용 범위를 제한한다. 실제 큰 문서 하나만으로도 메모리 부족이 계속되면, **그때** 전체 chunk를 한꺼번에 넘기는 임베딩 처리만 제한된 batch와 임시 저장 방식으로 개선한다. 모든 batch 결과를 다시 하나의 큰 리스트에 쌓는 방식은 개선으로 인정하지 않는다. 기존 checkpoint 두 지점은 유지한다.

## 검증과 보고

- UI와 MCP에서 여러 파일을 동시에 제출해도 활성 worker가 1개인지 확인한다. 대기 작업과 실행 작업 각각에서 Pause·Cancel·Resume을 확인한다.
- 새 UI/MCP 작업을 만든 뒤 UI를 다시 열었을 때 두 출처의 작업이 보이고, 상태 변화가 5초 이내 반영되는지 확인한다. 관리자 재시작 후 중단된 작업이 자동 실행되지 않는지도 확인한다.
- 완료 작업의 DB 결과는 남고 `snapshot.json`·`manifest.json`은 없는지 확인한다. 일시정지·실패·취소 작업은 유효한 체크포인트가 남는지 확인한다.
- 기존 `scripts/test-ui-job-recovery.py`, `scripts/test-ui-worker-e2e.py`, `scripts/test-worker-lifecycle-e2e.py`, `scripts/test-mcp-files-lifecycle-e2e.py`를 새 동작에 맞게 고친다. `scripts/test-haystack-checkpoint-resume.py`의 저수준 저장·로드 시험은 유지한다. 실제 E2E는 **새 임시 컬렉션**에서 실행한다.
- 변경 파일, 새 컬렉션 이름, 테스트 결과, 활성 worker 최대 수, 실제 CPU·peak 메모리 측정값과 남은 제한을 보고한다. 자원 문제가 여전히 재현되면 완료로 보고하지 말고 원인과 추가 수정 결과를 제시한다.

## 범위에서 제외

재부팅 후 자동 재개, 전원 차단 시점의 완전한 복구 보장, 임베딩 batch마다 영속 체크포인트, 파싱·청킹·임베딩의 별도 OS 프로세스, 새로운 작업 관리 프레임워크는 구현하지 않는다. 완료 뒤에도 작업 이력 DB와 관리 원본의 용량은 늘 수 있으며, 이번 정리 대상은 완료된 **처리 체크포인트 pair**다.

## 기존 버전과의 호환성

- **유지:** 기존 UI의 문서 추가·실행 흐름, MCP의 `rag3_ingest_file`·`rag3_ingest_text`·상태/제어 도구 이름과 필수 인자, 기존 응답의 `job_id`·`status` 등 주요 필드. 신규 작업의 checkpoint 형식과 두 저장 지점은 유지한다.
- **초기화:** 삭제된 Qdrant 컬렉션의 문서·검색 결과·문서 ID, 구 작업 JSON의 job ID·상태, 구 checkpoint의 재개는 새 시스템으로 이어지지 않는다. 새로운 컬렉션과 새로운 작업부터 시작한다.
- **확인 필요:** `scripts/ingest-worker.py`를 UI/MCP 밖에서 직접 호출하는 외부 자동화가 있다면 Bridge가 사용처를 확인하고 기존 CLI 인자 호환 여부를 보고한다. 직접 실행 작업을 공통 목록에 넣는 추가 기능은 이번 필수 범위에 넣지 않는다.

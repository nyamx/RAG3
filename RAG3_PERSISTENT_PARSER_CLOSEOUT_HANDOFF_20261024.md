# RAG3 상주 파서 작업: 새 채팅 단일 인계서

작성: 2026-10-24 KST. **새 채팅의 개발 AI는 이 문서 하나로 현재 작업을 이어서 마무리한다.** 이전 설계를 다시 평가하거나 합격 항목을 늘리지 않는다. 이 문서 작성 과정에서 애플리케이션 코드와 운영 프로세스는 변경하지 않았다.

## 1. 목표와 완료 판정

목표는 이미 작성된 상주 CPU 파서·반복 초기화 제거 코드를 안전하게 마무리하고, 실제 백엔드에 적용하는 것이다. 새 프로그램을 만들거나 Stage A–D를 다시 구현하지 않는다.

**완료**는 아래 네 조건을 모두 충족한 상태다.

1. 아래 3절의 확인된 파서 수명 결함을 고치고 해당 경합 시험을 통과한다.
2. 격리 환경에서 **실제 스케줄러 Cancel + 지연된 Qdrant upsert**, Qdrant 재연결, SQLite 다중 프로세스 초기화를 각각 한 번의 표적 시험으로 통과한다.
3. 변경 소스와 진행 기록만 선별 커밋해 재현 가능한 Git 스냅샷을 만든다.
4. 운영 작업이 없는 것을 확인한 뒤 백엔드를 한 번 통제 재시작한다. `/health`가 새 파서 PID/ready를 표시하고, 임시 컬렉션의 적재와 UI/MCP 검색이 성공하며 기존 82개 작업·운영 컬렉션이 보존된다.

장시간 부하·정확한 처리량 개선율·새 로그인 후 터널 복구는 별도 운영 관찰로 기록한다. 이전 Stage D에는 실제 로그인 복구 증거가 있다. 이 세 항목이 아직 없다면 해당 항목만 **미측정/미검증**으로 표시하고 기능 완료를 다시 17개 부분 판정으로 되돌리지 않는다. 새 로그인 시험을 수행한다면 모든 코드·기능 확인이 끝난 뒤 딱 한 번만 한다.

## 2. 현재 상태: 새 채팅에서 재조사하지 말고 한 번 확인

- 작업공간 `C:\python_program\RAG3`, 브랜치 `codex/shared-backend`, 마지막 확인 HEAD `22c1ed9`. Git 추적 변경: `rag3/backend_boundary.py`, `collections.py`, `haystack_native.py`, `ingestion_backend.py`, `ingestion_jobs.py`, `retrieval.py`, `sources.py`, `scripts/ingest-worker.py`, `scripts/rag3-mcp-server.py`, `RAG3_SHARED_BACKEND_PROGRESS.md`. 새 미추적 소스 6개: `rag3/persistent_parser.py`, `source_guard.py`, `qdrant_clients.py`, `qdrant_bm25.py`, `qdrant_store.py`, `retrieval_embedding.py`. **아직 커밋 없음.** 다른 수많은 미추적 사용자 자료·시험 산출물은 건드리지 않는다.
- 운영 SQLite는 마지막 감사에서 `quick_check=ok`, 82개 작업(`COMPLETE 72 / FAILED 5 / CANCELLED 5`), 활성 적재 0건이었다. 백업은 `artifacts/backups/persistent-parser-code-20261024_195808` 등 기존 산출물에 있다. 새 채팅에서는 당시 82건에 고정하지 말고 **시작 시점의 실제 수와 상태**를 기준선으로 보존한다. 운영 Qdrant 컬렉션에 시험 문서를 넣지 않는다.
- **현재 운영 백엔드는 새 코드를 쓰지 않는다.** 읽기 전용 확인에서 PID `14540`이 17:07에 시작됐고 `/health`에는 `cpu_parser` 필드가 없었다. 해당 확인 시점 이후 상태는 새 채팅에서 `/health` 한 번으로 확인한다. 격리 시험을 늘려도 운영 반영은 되지 않는다.
- CPU 파서 준비 37.90초, 상주 RSS 약 1.64 GiB, private bytes 약 2.12 GiB. 작은 격리 문서에서 PDF 4.204→3.468→3.357초, DOCX 0.099→0.048초. 성능 수치는 범용 처리량 약속이 아니다. BGE GPU 모델은 기존 백엔드에서 한 번 로드해 재사용한다.

## 3. 코드에서 확인된 수정 대상: 이 세 군데만 먼저 처리

### A. Pause 결과 경합 — 확정 결함

`rag3/persistent_parser.py`의 `parse()` 약 277–285행은 결과 이벤트가 도착한 뒤 `cancelled()`가 참이면 `PAUSED`를 반환하지만 `_request_id`, `_job_id`, `busy`를 정리하지 않는다. Pause 시 백엔드는 파서 자식을 강제 종료하지 않으므로 다음 Resume이 `state!='ready'`로 실패할 수 있다. **결과 수신과 Pause가 겹칠 때도 요청을 정확히 한 번 종료하고, 살아 있는 파서를 `ready`로 되돌린다.** 아직 실행 중인 요청이면 기존 Cancel/재기동 경로로 보낸다. 결과 직전·직후 Pause를 순서 제어한 결정적 시험 하나를 추가한다.

### B. 재기동·기동 실패 경계 — 좁은 수명 보강

`persistent_parser.py`의 `_supervise()` 약 69–82행은 재시작 이벤트를 지운 뒤 이전 자식을 회수한다. 이전 세대 프로토콜 리더의 늦은 EOF가 재시작 이벤트를 다시 세우면 새 정상 자식이 불필요하게 재시작될 가능성이 있다. **세대가 일치하는 이벤트만 재기동 요청을 소유**하도록 만들고, 늦은 EOF 주입 시험 하나로 확인한다.

`rag3/ingestion_backend.py`의 `boot()` 약 268–282행은 자식을 시작한 뒤 `recover()`를 호출하지만, `scripts/rag3-mcp-server.py`의 `scheduler.stop()` `finally`는 `boot()` 반환 뒤에만 설치된다. `recover()` 예외 시 시작된 자식을 정리하도록 부트 경계를 고친다. 복구 실패 주입 시험 하나로 자식·후손 부재를 확인한다.

### C. 반복 SHA 계산 — 사실 그대로 기록

8청크마다 전체 원본을 다시 해시하던 비용은 제거됐다. 다만 `BackendScheduler.start()` 약 402행, Resume의 `validate_identity()`, `_execute()`의 `OriginalSourceGuard`에서 작업 시작 시 전체 해시가 신규 2회·Resume 3회 남는다. 이것은 기능 배포를 막는 결함은 아니다. **이번 마무리에서 SHA 수명 설계를 다시 바꾸지 않는다.** 최종 보고서에 잔여 비용을 정확히 적고, 별도 최적화로 분리한다.

## 4. 이미 얻은 증거: 다시 실행하지 않는다

- 같은 자식에서 PDF/DOCX 연속 파싱과 파이프라인 재사용, 실제 Qdrant의 `parsed→ready`, GPU 임베딩이 확인됐다. PPTX/XLSX/HTML/MD/CSV/PNG 및 TXT/JSON/JSONL/TSV 격리 적재도 이후 별도 시험에서 통과했다.
- 실제 문서에 있는 검색어를 사용한 BM25/Semantic/Hybrid 검색은 각각 성공했다. 초기 GPU 통합시험의 최종 FAIL은 **시험 문서에 없는 BM25 검색어** 때문이었다. PDF/DOCX 적재 코드를 이 이유로 다시 고치지 않는다.
- 파서 `busy` 중 BM25 12건, Word COM DOC 변환, 부모 강제 종료 후 자식 회수, 외부 BM25 설정 변경, 실제 Qdrant 지연 upsert의 모의 Cancel 경계, 평문 적재 등은 각각 PASS 증거가 있다. 이들은 **실제 스케줄러 Cancel과 지연 write가 결합된 한 번의 시험**을 대체하지는 않는다.
- 기존 `artifacts/analysis/persistent-parser/acceptance-matrix-20261024-44.json`과 `RAG3_SHARED_BACKEND_PROGRESS.md` 182행 이후의 표는 21:54 시점이라 낡았다. 그 뒤 통과한 6개 결과가 반영되지 않았다. 최신 결과 색인은 `artifacts/analysis/persistent-parser/readiness-audit-20261024-51.json` 및 같은 폴더의 개별 JSON이다. **기존 증거를 표에 반영할 뿐, 표를 채우려고 통과 시험을 재실행하지 않는다.**

## 5. 실행 순서와 시험 상한

1. 시작 시 `git status`, HEAD, 운영 적재 0건, SQLite `quick_check`와 기존 백업의 존재를 **한 번** 확인한다. 코드와 사용자 자료를 섞어 정리하거나 `git clean`을 실행하지 않는다.
2. 3절 A/B만 수정한다. 각 결함의 순서 제어 가짜 파서 시험을 실행한다. 실패한 해당 시험만 수정 후 재실행한다. 같은 실패가 새 근거 없이 두 번 반복되면 다른 전체 시험을 돌리지 말고 원인·재현·막힌 지점을 보고한다.
3. 격리 SQLite·임시 Qdrant 컬렉션을 사용해 남은 **세 표적 시험**을 각 한 번 수행한다: 실제 스케줄러 Cancel+지연 upsert 후 Resume/Restart 보호, 재사용 Qdrant 클라이언트의 서버 중단·복구, 빈/구형 시험 SQLite의 다중 프로세스 동시 스키마 준비. Qdrant 중단 시험은 **별도 시험용 서버/포트**에서만 하고 운영 Qdrant를 멈추지 않는다. 실패한 것만 수정·재시험한다. 예전 GPU/형식 전체 스위트를 반복하지 않는다.
4. 변경 범위를 선별 리뷰하고 위 소스 15개와 진행 기록만 명시적으로 `git add`/커밋한다. 신규 6개 모듈 누락 여부를 확인한다. `artifacts/`, `.qdrant_runtime/`, 사용자 문서, 기존 기타 미추적 파일을 커밋하지 않는다. 커밋 해시와 변경 파일을 기록한다.
5. 운영 작업이 0건이고 SQLite 온라인 백업·무결성 확인이 끝난 뒤 백엔드를 한 번 통제 재시작한다. 새 `/health.cpu_parser`의 PID/ready와 BGE 콜드 상태를 확인하고, 임시 컬렉션에서 문서 1개 적재 중 MCP/HTTP 검색 및 UI 경로를 확인한다. **서로 다른 두 임시 문서**를 파싱할 때 파서 PID가 유지되는지 확인한다. 임시 자료만 정리하고 시작 시 기록한 운영 작업·컬렉션을 재확인한다.
6. 기존 17항목 표를 **마지막에 한 번만** 새 증거로 갱신한다. 기능 완료 여부와 별도 운영 관찰 항목을 나눠 한 페이지로 보고한다. 재로그인/장시간 부하가 필요하면 이 시점에만 한 번 계획하고 결과를 덧붙인다.

## 6. 인계·완료 보고 형식

최종 보고에는 `(a)` 수정한 실제 결함, `(b)` 표적 시험 3개의 PASS/FAIL과 증거, `(c)` 커밋 해시, `(d)` 운영 백엔드 새 PID·`/health.cpu_parser`·임시 적재/검색 결과, `(e)` 시작 시점의 운영 작업 및 컬렉션 보존, `(f)` 상주 RAM과 남은 시작 시 2/3회 해시, `(g)` 남긴 운영 관찰만 적는다. 현재 Python SQLite 3.40.1의 WAL 버전 위험과 런타임 업그레이드는 별도 작업으로 기록한다. 격리 시험 PASS를 운영 적용 PASS라고 쓰지 않는다. 미검증 항목을 새 설계 논의로 되돌리지 않는다.

이 문서는 **마무리 실행 지시서**다. 이전 장문의 `RAG3_PERSISTENT_CPU_PARSER_IMPLEMENTATION_TASK.md`와 17항목 표는 구현 배경·증거 색인으로만 사용한다. 새 기능 제안과 무관한 레거시 정리는 진행하지 않는다.

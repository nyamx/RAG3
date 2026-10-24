# RAG3 연결 및 운영 가이드 — Stage D

**출처 기준:** 2026-10-24 Stage D 구현·Bridge 실환경 검증. **상태: PARTIAL.** 로컬 HTTP 백엔드, 실제 Qt UI, 설치된 Codex 플러그인의 새 작업 도구 호출은 확인했다. 기존 원격 터널의 외부 MCP 종단 호출, 실제 Windows 재로그인, 전체 장애 복구와 변경 전후 성능 비교는 미검증이다. 자세한 판정은 [RAG3_SHARED_BACKEND_PROGRESS.md](RAG3_SHARED_BACKEND_PROGRESS.md)와 `STAGE_D_ACCEPTANCE_STATUS.md`를 따른다.

## 연결 구조

| 이용 경로 | 연결 방식 | 검증 현황 (2026-10-24) |
| --- | --- | --- |
| 이 PC의 Codex | 설치된 `rag3@rag3-local` → 로컬 `http://127.0.0.1:8000/mcp` (Streamable HTTP) | 플러그인 `0.1.2` 활성, 새 Codex 작업에서 실제 읽기 도구 호출 PASS |
| PySide6 UI (`RAG3.exe`) | 인증된 loopback `POST /api/v1/command` → 공용 백엔드 | 실제 Qt UI의 작업·컬렉션 조회, 검색 UI/MCP 동등성 검증 PASS |
| ChatGPT 웹/클라우드 | 기존 Secure MCP Tunnel → 같은 로컬 `/mcp` | 터널 런타임 ready 확인, **원격 MCP E2E 미검증** |
| ChatGPT 데스크톱 별도 설치 | 제품 및 설치 상태에 따라 다름 | 이 제품 UI의 직접 호출 미검증; Codex CLI 성공으로 대체하지 않음 |

기본 상주 RAG3 Python 프로세스는 공용 HTTP/MCP 백엔드 하나다. UI를 열 때만 UI 프로세스가 추가된다. Qdrant, 외부 터널 런타임은 별도 서비스이며 적재 중 임시 CPU 파서가 실행될 수 있다. UI를 닫아도 백엔드 및 MCP 연결은 계속 유지된다. 구 stdio 플러그인, 상주 MCP 트레이, 구 검색·컬렉션 worker를 정상 기동에 사용하지 않는다.

## 1. Windows 로그인 백엔드

저장소: `C:\python_program\RAG3`

- `scripts/rag3-backend-login.py`: 로그인 예약 작업이 직접 실행하는 단일 백엔드 진입점. 설정 검증 → 필요 시 Qdrant 부트스트랩 → 공용 서버 기동 → 준비 후 저장 터널 재연결 도우미 실행.
- `scripts/manage-rag3-login.ps1`: `-Operation Inspect`, `Plan`, `Install` 지원. `Install`은 사용자 범위 예약 작업 `RAG3 Shared Backend`를 등록하고 기존 충돌 설정을 임의로 덮어쓰지 않는다.
- `data/backend-login-config.json`: 포트, Qdrant URL, 백엔드 전체 쓰기·삭제 권한, 인프라 기동 및 터널 자동 재연결에 대한 비밀값 없는 설정. 민감한 터널 자격 증명은 이 파일에 저장하지 않는다.

```powershell
cd C:\python_program\RAG3
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\manage-rag3-login.ps1 -Operation Inspect
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\manage-rag3-login.ps1 -Operation Plan
# 등록이 필요한 경우에만 승인 후 실행:
# powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\manage-rag3-login.ps1 -Operation Install
Get-ScheduledTask -TaskName 'RAG3 Shared Backend'
```

현재 PC의 등록 설정은 포트 `8000`, Qdrant `127.0.0.1:16333`, 쓰기·삭제 허용이다. 이 설정은 **원격 클라이언트별 정책이 아니라 공용 백엔드 전체 정책**이다. 정책 변경은 활성 작업 확인과 백엔드 재시작이 필요하며, UI만 재시작해서는 반영되지 않는다. 재시작 전에 DB 온라인 백업과 활성 적재·다운로드·GPU 큐 상태를 확인한다. `data/ingestion_jobs.sqlite`의 WAL DB를 파일 하나만 복사하지 않는다.

`RAG3.exe`는 UI만 실행한다. UI를 닫아도 백엔드 PID는 유지된다. 인증 UI API는 로컬 전용이며 백엔드가 발급한 포트별 토큰을 `X-RAG3-Token` 헤더로 전송한다. 토큰을 플러그인 manifest, URL, 로그, 터널 프로필에 복사하지 않는다. 백엔드가 내려가 있으면 UI에서 연결 실패로 보고하며 구 manager를 자동 기동하지 않는다.

## 2. Codex 로컬 플러그인

저장소 `plugins/rag3/mcp.json`과 설치 캐시는 이제 stdio가 아니라 다음 형식이다.

```json
{
  "mcpServers": {
    "rag3": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

로컬 마켓플레이스는 `.agents/plugins/marketplace.json`, 플러그인 이름은 `rag3@rag3-local`이다. 현재 설치된 플러그인 버전 `0.1.2`는 HTTP 방식으로 갱신됐으며, 새 Codex 작업에서 `rag3_list_collections` 실제 도구 호출을 검증했다. 설치·확인 예시는 다음과 같다.

```powershell
codex plugin marketplace list --json
codex plugin list --json
& C:\python_program\RAG3\.venv\Scripts\python.exe C:\python_program\RAG3\scripts\ensure-rag3-codex-plugin.py --check-only
# 기존 stdio 설치를 의도적으로 전환하고 HTTP 백엔드가 정상일 때만:
# & C:\python_program\RAG3\.venv\Scripts\python.exe C:\python_program\RAG3\scripts\ensure-rag3-codex-plugin.py --activate-http
```

설치 헬퍼는 실제 HTTP MCP 초기화·도구 조회를 선행 검증하고 사용자가 비활성화한 플러그인을 임의 활성화하지 않는다. 기존 설치본을 바꾸기 전에 설치 상태와 복구용 백업을 확인한다. `RAG3.exe`나 UI 시작은 Codex 플러그인을 자동 설치하거나 갱신하지 않는다. 연결 정의를 바꾼 후에는 새 Codex 작업을 연다. 기존 대화에 자동으로 도구가 갱신된다고 가정하지 않는다.

**접근 권한:** `RAG3_MCP_WRITE_ENABLED=0`은 백엔드 전체 쓰기를 차단하며 `RAG3_MCP_DELETE_ENABLED=0`은 삭제만 차단한다. 삭제 도구는 정확한 대상 확인 값을 요구한다. 플러그인에서 예전 stdio `env`를 설정해도 공용 백엔드 권한은 바뀌지 않는다.

## 3. ChatGPT 웹/클라우드와 터널

로컬 `127.0.0.1` 주소는 클라우드 ChatGPT에서 직접 접속할 수 없다. 기존 OpenAI Secure MCP Tunnel이 같은 `http://127.0.0.1:8000/mcp`를 전달하도록 구성돼 있다. 저장된 DPAPI 키, alias, 프로필, 터널 ID는 유지하며 URL/자격 증명을 문서·로그에 적지 않는다. `scripts/rag3-tunnel-autostart.py`는 백엔드 준비 후 호출되는 **비상주 연결 도우미**이고, 터널 자체 런타임은 별도 프로세스다.

2026-10-24 진단 결과 터널 런타임 `healthy=True`, `ready=True` 및 저장된 인증정보 복호화 성공은 확인했다. 다만 외부 MCP 주소를 특정하지 못해 외부에서 `initialize`와 읽기 도구 호출을 완료하지 못했다. **이 항목은 미검증이며 Stage D 전체 PASS 조건을 충족하지 않는다.** 터널 상태나 로컬 Codex MCP 호출 성공만으로 원격 연결을 입증해서는 안 된다. ChatGPT 측에서 연결을 새로 고쳐야 한다면 기존 등록을 먼저 확인하고, 불필요한 새 터널·새 키를 생성하지 않는다.

연결/복구 점검: 로컬 `/health`에서 PID와 준비 상태 확인 → 로컬 `/mcp`에서 `initialize`, `tools/list`, 읽기 호출 → 기존 `tunnel-client` 상태와 프로필 확인 → 실제 **외부** 종단에서 동일한 MCP 읽기 호출. 원격 연결을 확인하지 못하면 문서에 정확히 미검증으로 기록한다.

## 4. 검색, 적재, 데이터 보호

UI와 MCP는 동일한 검색·작업·컬렉션 백엔드를 사용한다. BGE-M3는 `cuda:0`에서 공유 로드하고 BM25-only는 GPU 모델 로드를 요구하지 않는다. Semantic, Hybrid RRF, 비융합 Hybrid의 UI↔MCP 결과 ID·순위 동등성은 2026-10-24 실환경에서 검증했다. RRF 점수 동점은 안정적 `point_id` 순서로 정렬한다. 데이터는 `data/ingestion_jobs.sqlite`와 Qdrant에 있으며 서버 재시작 시 기존 작업을 자동으로 중복 제출하지 않는다.

**격리 시험 주의:** `scripts/test-ui-worker-e2e.py`와 `scripts/test-mcp-http-client.py`를 운영 설정 그대로 실행하지 않는다. 임시 Qdrant 컬렉션만 지정해도 운영 작업 DB를 공유하거나 복구 처리할 수 있다. 자식 프로세스까지 전달되는 `RAG3_TEST_DATA_ROOT`·`RAG3_BACKEND_TEST_INSTANCE`와 임시 원본 경로를 확인하고, 운영 컬렉션에는 읽기만 수행한다.

원본 첨부 수집은 MCP의 파일 참조를 사용하며 PDF/DOC/DOCX/MD/TXT/JSON/JSONL/CSV/TSV를 지원한다. 사용자에게 원본을 Base64로 인코딩하도록 요구하지 않는다. 터널의 레거시 인라인 페이로드 메시지 제한은 별도로 적용된다. 상세 MCP 도구 목록과 의미는 [MCP.md](MCP.md)를 참조한다.

## 5. 남은 검증과 판정

Stage D의 **PARTIAL** 판정은 현재 실제 측정 및 호출 범위를 반영한다. 남은 항목은 외부 터널 E2E, 실제 Windows 로그아웃/로그인 후 기동, 백엔드 중단·Qdrant 장애 복구 및 체크포인트 검증, 쓰기 시간초과 결정적 재현, 전환 이전과 이후 동일 조건의 성능 비교다. 최초 CUDA 로딩 및 일부 검색 시간은 측정했지만 전체 자원·처리량과 신뢰할 수 있는 이전 기준선은 미측정이다. 구현·검증 로그와 백업 위치는 `RAG3_SHARED_BACKEND_PROGRESS.md`에 남긴다.

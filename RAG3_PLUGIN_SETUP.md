# RAG3를 Codex·ChatGPT에 연결하기

기준: 2026-10-18. 이 문서는 **이 PC의 로컬 Codex 플러그인**과 **ChatGPT 웹/앱의 원격 연결**을 구분한다. 플러그인 이름은 `RAG3`, 로컬 마켓플레이스 이름은 `rag3-local`이다.

## 먼저 알아둘 것

| 사용 위치 | 연결 방식 | RAG3 PC에서 실행할 것 | 현재 검증 상태 |
| --- | --- | --- | --- |
| 이 PC의 Codex 로컬 작업 | RAG3 플러그인이 MCP 서버를 `stdio`로 직접 실행 | Qdrant, RAG3 `.venv` | CLI 설치 및 실제 AI 도구 호출 검증 완료 |
| 이 PC의 ChatGPT 데스크톱 로컬 Chat/Work | 로컬 플러그인 설치 후 로컬 실행 대화에서 사용 | Qdrant, RAG3 `.venv` | 공식 설치 절차 확인; 이 제품 화면의 실제 호출은 별도 검증 필요 |
| ChatGPT 웹/앱의 클라우드 Chat/Work | RAG3의 로컬 HTTP MCP 서버에 Secure MCP Tunnel로 연결하고 ChatGPT에 사용자 지정 MCP 플러그인 등록 | Qdrant, RAG3 트레이 서버, `tunnel-client` | 공식 절차 확인; 터널·ChatGPT 연결 및 실제 호출은 아직 미검증 |

로컬 플러그인은 `plugins/rag3`에 있고, 마켓플레이스는 `.agents/plugins/marketplace.json`에 있다. RAG3의 MCP 도구는 현재 16개다. `scripts/rag3-mcp-server.py`가 기존 ingestion·검색 코드를 호출한다. 로컬 `stdio` 방식은 트레이 HTTP 서버가 꺼져 있어도 Qdrant와 `.venv`가 준비되면 사용할 수 있다. [OpenAI 플러그인 패키징 문서](https://developers.openai.com/plugins/build/plugins)

## 1. Codex에 설치하기

현재 PC에서는 아래 명령으로 설치했다. PowerShell을 열고 실행한다.

```powershell
codex plugin marketplace add C:\python_program\RAG3
codex plugin add rag3@rag3-local
codex plugin marketplace list --json
codex plugin list --json
```

마지막 결과에서 `rag3@rag3-local`의 `installed`와 `enabled`가 모두 `true`인지 확인한다. 이후 **새 로컬 Codex 작업**을 열고 `@RAG3`를 선택하거나 “RAG3 컬렉션 목록을 보여줘”처럼 요청한다. 현재 PC에서는 Codex CLI AI가 자연어 요청을 받고 실제 `rag3_list_collections` 도구를 호출하는 것까지 검증했다. 결과 기록은 `artifacts/rag3-plugin-natural-language-after-skill.jsonl`에 있다. [OpenAI 로컬 마켓플레이스 문서](https://developers.openai.com/plugins/build/plugins)

플러그인의 `mcp.json`에는 현재 설치 경로 `C:\python_program\RAG3`가 들어 있다. 프로젝트를 다른 곳으로 옮기면 `plugins/rag3/mcp.json`의 `RAG3_PROJECT_ROOT`를 바꾸고 플러그인을 다시 설치해야 한다. 기존 대화에 도구가 바로 추가되지 않을 수 있으므로 새 작업을 연다. 이 플러그인은 쓰기와 삭제 도구도 켜며, 삭제 도구는 서버에서 지정된 확인 값을 추가로 요구한다.

### UI를 실행할 때 자동 확인

`RAG3.exe`를 더블 클릭하거나 `scripts/ingest-ui.py`를 정상 실행하면 UI가 별도 백그라운드 프로세스로 Codex 플러그인 상태를 확인한다. `rag3-local` 마켓플레이스나 `rag3@rag3-local` 플러그인이 없으면 위 설치 명령을 자동 실행한다. 이미 설치된 플러그인은 다시 설치하지 않는다. 사용자가 **비활성화한 플러그인은 그대로 둔다**. 경로가 다른 `rag3-local` 마켓플레이스가 이미 있으면 덮어쓰지 않고 충돌로 기록한다. 자동 업데이트도 하지 않는다.

확인 결과는 `data/logs/codex-plugin-status.json`, 실행 기록은 `data/logs/codex-plugin-install.log`에서 볼 수 있다. 설치 명령을 직접 시험하려면:

```powershell
& C:\python_program\RAG3\.venv\Scripts\python.exe C:\python_program\RAG3\scripts\ensure-rag3-codex-plugin.py --check-only
& C:\python_program\RAG3\.venv\Scripts\python.exe C:\python_program\RAG3\scripts\ensure-rag3-codex-plugin.py
```

UI의 `--smoke-test` 모드는 자동 확인을 실행하지 않는다. Codex CLI가 설치되어 있지 않거나 설치 명령이 실패해도 RAG3 UI와 트레이 서버 시작을 막지 않고 로그에 원인을 남긴다.

## 2. 이 PC의 ChatGPT 데스크톱 Chat/Work에서 사용하기

ChatGPT 데스크톱 앱의 Plugins Directory에서 로컬 소스 `rag3-local`의 `RAG3`를 찾아 설치하고, **로컬에서 실행되는 새 대화**를 연다. 입력창에서 `@RAG3`를 선택한 뒤 “RAG3에서 컬렉션 목록을 보여줘”로 확인한다. 설치된 로컬 플러그인이 디렉터리에 보이지 않으면 앱을 새로고침하고 로컬 마켓플레이스 등록 상태를 확인한다. 로컬 `stdio` MCP 플러그인은 로컬 실행 대화에서만 쓸 수 있다. [ChatGPT 플러그인 사용 안내](https://learn.chatgpt.com/docs/plugins), [데스크톱 전용 플러그인 안내](https://developers.openai.com/plugins/deploy/desktop-only)

Codex CLI에서 설치한 상태와 ChatGPT 데스크톱 앱의 설치 표시가 실제로 동기화되는지는 이 PC에서 별도로 확인해야 한다. 따라서 데스크톱 앱의 플러그인 목록, 새 대화의 도구 목록, 실제 호출 결과를 차례로 확인한다.

## 3. ChatGPT 웹/앱의 클라우드 Chat/Work에서 사용하기

웹이나 클라우드 대화는 이 PC의 `stdio` 프로세스 또는 `127.0.0.1`에 직접 접속할 수 없다. 이 경우 RAG3 트레이 HTTP 서버와 OpenAI Secure MCP Tunnel을 연결하고 ChatGPT에 **사용자 지정 MCP 서버**로 등록한다. RAG3 UI를 실행하면 트레이에서 Streamable HTTP 서버가 기본적으로 `http://127.0.0.1:8000/mcp`에 뜬다. UI 창을 닫아도 트레이 서버는 남는다. 트레이 메뉴에서 종료하면 연결이 끊긴다. [OpenAI Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)

1. [OpenAI Platform의 Tunnel 설정](https://platform.openai.com/settings/organization/tunnels)에서 터널을 만들고 `tunnel_id`를 받는다. 대상 ChatGPT 워크스페이스와 필요한 Platform 조직을 터널에 연결한다. 생성·수정에는 **Tunnels Read + Manage**, 실행 및 선택에는 **Tunnels Read + Use** 권한이 필요하다. ChatGPT 워크스페이스의 사용자 지정 MCP 서버 허용 정책도 확인한다.
2. 공식 [tunnel-client](https://github.com/openai/tunnel-client)를 설치한다. 현재 PC에는 아직 설치되지 않았다. 런타임 API 키는 제한된 권한으로 만들고, 터미널 기록·문서·로그에 값을 넣지 않는다. 아래의 `<TUNNEL_ID>`는 실제 값으로 바꾼다. CLI 버전에 따른 옵션은 먼저 `tunnel-client help quickstart`로 확인한다.

   ```powershell
   tunnel-client help quickstart
   tunnel-client init --sample sample_mcp_remote_no_auth --profile rag3-local --tunnel-id <TUNNEL_ID> --mcp-server-url http://127.0.0.1:8000/mcp
   tunnel-client doctor --profile rag3-local --explain
   tunnel-client run --profile rag3-local
   ```

   `CONTROL_PLANE_API_KEY`는 공식 안내에 따라 로컬 환경에서 안전하게 제공한다. `tunnel-client run`은 ChatGPT에서 도구를 찾고 호출하는 동안 계속 실행되어야 한다. 위 HTTP 샘플과 옵션은 [tunnel-client 설정 문서](https://github.com/openai/tunnel-client/blob/master/docs/configuration.md)를 따른다. 이 PC에 구성된 클라이언트는 `data/tools/tunnel-client/v0.0.16/bin/tunnel-client.exe`이며, 연결 도우미는 `data/tools/tunnel-client/연결 시작.cmd`이다. 기존에 연결했다면 다시 등록하거나 키를 만들 필요가 없다.
3. [ChatGPT Plugins](https://chatgpt.com/plugins)에서 **+ → Add custom MCP server**를 선택한다. 이름을 `RAG3`로 지정하고 Connection은 **Tunnel**, 위의 `tunnel_id`를 선택한다. 표시되는 인증·위험 확인 절차를 마치고 **Create as a plugin**을 누른다. 도구 목록을 검토한 뒤 설치한다.
4. 새 Chat 또는 Work 대화에서 `@RAG3`를 선택하고 **읽기 요청**인 “RAG3 컬렉션 목록을 보여줘”로 실제 호출을 확인한다. 검색도 시험하고, 쓰기 기능을 쓸 때는 트레이 서버의 권한 설정을 별도로 검토한다. 서버의 도구 설명·스키마를 바꾸면 ChatGPT 연결 화면에서 **Refresh**하고 새 대화를 연다. [OpenAI 사용자 지정 MCP 서버 연결 절차](https://developers.openai.com/plugins/deploy/connect-chatgpt)

트레이 HTTP 서버와 로컬 stdio 서버는 기본적으로 **쓰기·삭제 도구를 허용**한다. 따라서 웹 연결에서도 컬렉션 생성, 파일·텍스트 수집, 작업 제어와 삭제를 사용할 수 있다. 삭제는 대상 이름/ID의 정확한 확인 값과 기존 작업 상태 검사를 계속 요구한다. 운영자가 읽기 전용으로 제한하려면 `RAG3_MCP_WRITE_ENABLED=0`, 삭제만 제한하려면 `RAG3_MCP_DELETE_ENABLED=0`을 지정한다. 외부 연결은 기존 인증 터널과 접근 권한을 유지한다. 공개 제품으로 배포할 경우 사설 터널만으로 공개 플러그인 제출을 대신할 수 없으며, 인증된 안정적인 HTTPS `/mcp` 엔드포인트가 필요하다. [OpenAI 연결 및 배포 안내](https://developers.openai.com/plugins/deploy/connect-chatgpt), [Secure MCP Tunnel의 범위](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)

서버 코드나 환경 설정 변경 후에는 트레이 메뉴에서 **Exit RAG3 background**를 눌러 기존 서버를 종료한 뒤 다시 실행한다. 기본 허용 상태에서는 별도 환경변수가 필요 없다. 읽기 전용으로 제한할 때만 아래처럼 실행한다. 컬렉션·문서 삭제는 MCP 도구의 정확한 확인 값을 요구한다.

```powershell
cd C:\python_program\RAG3
$env:RAG3_MCP_WRITE_ENABLED = '0'  # 모든 쓰기를 차단할 때만
# 삭제만 차단하려면 위 줄 대신 아래 설정을 사용한다.
# $env:RAG3_MCP_DELETE_ENABLED = '0'
& .\RAG3.exe
```

이미 실행 중인 트레이 서버는 UI를 다시 열어도 환경 변수가 바뀌지 않는다. 서버를 종료하고 다시 시작해야 한다.

**원본 첨부파일 수집:** `rag3_ingest_file`은 ChatGPT 첨부파일 참조(`file`)를 받아 PDF/DOC/DOCX/MD 원본을 PC로 직접 내려받는다. 사용자가 Base64로 변환하거나 PDF를 텍스트로 바꿀 필요가 없다. **RAG3의 25 MiB 파일 상한과 40 MiB 요청 상한은 제거했다.** 다운로드는 작업으로 즉시 접수되며, `rag3_ingestion_status`로 `DOWNLOADING` 상태와 진행 바이트를 확인할 수 있다. 일시정지·재개·취소도 지원한다. MCP 터널에는 파일 본문 대신 작은 참조 정보만 전달하므로 원본 파일에 10 MiB 터널 메시지 한도가 적용되지 않는다. 이전 로컬 클라이언트용 Base64 입력에는 터널 자체의 메시지 제한이 계속 적용된다. [ChatGPT 파일 입력 규격](https://developers.openai.com/plugins/reference), [Secure MCP Tunnel 제한](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)

서버를 업데이트한 뒤 ChatGPT의 RAG3 연결 화면에서 **Refresh**하여 새 파일 입력 규격을 반영하고 새 대화를 연다. 도구 입력에 `file` 객체와 `openai/fileParams` 메타데이터가 표시되는지 확인한다. 이전 대화에 남아 있는 `content_base64` 필수 입력 정의로는 첨부파일을 전달할 수 없다.

## 자동 등록의 범위

UI 시작 시 구현된 자동화는 **이 PC의 Codex 로컬 플러그인**에 한정된다. ChatGPT 웹·클라우드용 터널 생성과 사용자 지정 MCP 플러그인 등록은 OpenAI 계정/조직/워크스페이스 권한, 런타임 키, 연결 확인이 필요해서 UI가 임의로 대신하지 않는다. 나중에 원격 연결 정보를 사용자가 설정하고 권한을 부여하면 `tunnel-client` 상태 점검과 자동 실행을 추가할 수 있다.

## 문제 해결

| 현상 | 확인할 곳 |
| --- | --- |
| Codex에서 `@RAG3`가 안 보임 | `codex plugin marketplace list --json`, `codex plugin list --json`, `data/logs/codex-plugin-status.json`; 새 로컬 대화 시작 |
| 자동 설치 상태가 `codex_missing` | Codex CLI 설치 여부와 PATH; UI를 다시 실행하면 재확인 |
| `project_path_mismatch` | `plugins/rag3/mcp.json`의 `RAG3_PROJECT_ROOT`와 실제 프로젝트 경로 일치 여부 |
| `marketplace_conflict` | 기존 `rag3-local`이 다른 프로젝트를 가리킴; 사용자 설정을 확인한 뒤 수동으로 정리 |
| ChatGPT 터널이 안 보임 | 터널과 대상 워크스페이스 연결, Tunnels Read + Use 권한, `tunnel-client doctor` |
| 터널은 연결됐지만 도구 호출 실패 | RAG3 트레이 서버 상태, `data/logs/mcp-server.log`, `data/logs/mcp-tray.log`, `tunnel-client` 상태 |

RAG3 서버의 도구별 입력 및 동작은 [MCP.md](MCP.md)에 정리되어 있다.

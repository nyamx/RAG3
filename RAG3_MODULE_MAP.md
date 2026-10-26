| 영역 | 주요 소스 | 구현 내용 |
|---|---|---|
| 데스크톱 UI | [ingest-ui.py](scripts/ingest-ui.py), [ui_state.py](rag3/ui_state.py) | 파일 대기열, 작업 버튼과 상태, 컬렉션·검색 화면. UI는 `.txt`를 허용 |
| 수집 | [ingest-worker.py](scripts/ingest-worker.py), [haystack_native.py](rag3/haystack_native.py) | 원본 보존 → 전체 문서 Docling 파싱 또는 일반 텍스트 직접 읽기 → 청크 Qdrant 기록 → 별도 프로세스 BGE-M3 임베딩 및 BM25 벡터 기록 |
| 재개 | [ingestion_jobs.py](rag3/ingestion_jobs.py) | SQLite 진행 기록과 Qdrant 청크 상태로 일시정지·재개 처리. 스냅샷 파일 없음 |
| 검색 | [retrieval.py](rag3/retrieval.py), [retrieval-worker.py](scripts/retrieval-worker.py) | 의미·어휘·BM25 검색과 결과 결합; 작업자는 UI에서 호출 |
| 컬렉션·문서 | [collections.py](rag3/collections.py), [collection-worker.py](scripts/collection-worker.py) | Qdrant 컬렉션 생성·목록·삭제, 문서별 청크 조회·삭제 |
| MCP | [rag3-mcp-server.py](scripts/rag3-mcp-server.py), [mcp_jobs.py](rag3/mcp_jobs.py) | 19개 도구의 입력·응답 처리, 파일 수집 작업과 상태 제어. MCP 파일 수집에는 현재 `.txt`가 빠져 있음 |
| 원본 파일 | [attachments.py](rag3/attachments.py), [sources.py](rag3/sources.py), [source-viewer.html](rag3/source-viewer.html) | 첨부파일 다운로드 검증, 저장된 원본의 식별·분할 전송, PDF 페이지 표시 |
| 실행·연결 | [Program.cs](launcher/Program.cs), [rag3-mcp-tray.py](scripts/rag3-mcp-tray.py), [ensure-infra.ps1](scripts/ensure-infra.ps1) | Windows 실행기, MCP 트레이 프로세스, Qdrant 기동 |

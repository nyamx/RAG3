# Stage D verified evidence and open acceptance gaps

Recorded: 2026-10-24 13:25:28 KST. Git branch `codex/shared-backend`, HEAD `f23a83d` (working tree may be dirty). This is a **partial Stage D acceptance record, not a Stage D completion certificate**.

## Data protection at record time
- Operational SQLite `PRAGMA quick_check`: `ok`; jobs: **82**; active: **0**. Status counts: `{"CANCELLED": 5, "COMPLETE": 72, "FAILED": 5}`.
- Existing Qdrant user collections were used for read-only parity checks. Stage D test DB/virtual Qdrant tests were separate. Runtime shutdown/restart checks reported the same 82 operational jobs.
- Installed plugin and scheduler registration were changed earlier in Stage D, with backups under `artifacts/backups/`. This record does not change either installation.

## Verified evidence
| Area | Actual observation | Evidence (Bridge output ZIP) |
| --- | --- | --- |
| Data backup | Online SQLite backup `integrity_check=ok`, operational jobs 82 | `run_output_20261024_112701_992.zip` |
| UI/client contracts | Common search, error resilience, plugin/login bootstrap, RRF ties **5/5 PASS**; compile and diff checks PASS | `run_output_20261024_132107_964.zip` |
| HTTP MCP server | `/health`, authenticated UI API, MCP initialization, 22 tools and real `rag3_list_collections` all PASS | `run_output_20261024_122343_463.zip` |
| Actual installed Codex plugin | HTTP plugin **0.1.2** activated; new Codex read-only task made `rag3_list_collections` call | `run_output_20261024_123410_130.zip`, `run_output_20261024_123709_733.zip` |
| Qt UI | Actual UI ↔ backend read-only API verified; launcher release build & UI lifecycle PASS, same backend PID | `run_output_20261024_125503_165.zip`, `run_output_20261024_130353_292.zip` |
| Model and search parity | Post-restart semantic 1/1, RRF 4/4, non-fused hybrid 2/2 same ordered UI/MCP IDs; BGE service load count **1** | `run_output_20261024_131824_366.zip` |
| Backend restart | PID `18140` → `22688`; auth token rotated, lazy GPU load, existing jobs protected | `run_output_20261024_131635_863.zip` |
| Cold/warm BM25 | 12 post-cutover samples, median 173.0 ms, nearest-rank p95 22215.7 ms (cold outlier included). No comparable precutover sample | `run_output_20261024_130650_860.zip` |

## Unverified / not yet satisfied Stage D acceptance
1. **Existing remote Secure MCP Tunnel actual external endpoint**: runtime `healthy`/`ready` and DPAPI profile are verified; public remote MCP URL was not established, and remote MCP `initialize` + read tool through that tunnel was **not tested**. ChatGPT-connected RAG3 job-list call succeeded, but its transport path is not proven to traverse this saved tunnel.
2. **Real Windows logoff/relogin** and unattended reconnect: scheduled task registration/running and manual restart are verified; a genuine new login session is **not tested**.
3. **End-to-end fault recovery matrix**: isolated active ingestion crash/resume, injected Qdrant outage/recovery, token refresh under live Qt UI and late-write ambiguity require acceptance proof for the final deployed UI path.
4. **Performance comparisons and resources**: prior UI baseline, sufficient p95 sample size, ingest throughput comparison, process/RAM/VRAM peaks all **not measured** to the specified acceptance standard. Observed latencies above are reference samples only.
5. **Final deployment cleanliness/documentation**: verify no legacy tray/worker/manager entrypoints remain in normal startup, update user-facing docs and produce install/rollback audit and functional commits.

No `Stage D PASS` claim until each required external and operational test is verified or an explicit scoped exception is agreed.

# Install on Windows

This is the setup used for RAG3's local Windows build. The commands describe a fresh, empty Qdrant installation. Migrating existing collections also requires Qdrant snapshots and the corresponding `data/sources/` files. A complete second-PC installation has not yet been verified.

## Tools and services

| Item | Why it is needed | Working setup / check |
| --- | --- | --- |
| Windows x64 and PowerShell | UI, backend, launcher, and setup scripts run on Windows | Windows x64 |
| Git | Clone the repository | `git --version` |
| Python 3.11 | Backend, UI, Docling, and BGE-M3 | 3.11.3; `py -3.11 --version` |
| NVIDIA GPU and driver | CUDA PDF parsing and BGE-M3 embeddings | Quadro T1000, 4 GiB, driver 573.22; `nvidia-smi` |
| WSL2 with Ubuntu | Hosts the Docker Engine used by the startup script | `wsl --list --verbose` |
| Docker Engine and Compose plugin **inside Ubuntu** | Runs Qdrant | `wsl -d Ubuntu -u root -- docker info` and `docker compose version` |
| Qdrant | Stores dense and BM25 vectors | Image `qdrant/qdrant:v1.19.2` in `compose.yaml` |
| BGE-M3 and Docling model files | Local embedding and parsing | Stored below `data/models/` |
| .NET 8 | Runs `RAG3.exe`; the SDK is needed to build it | `dotnet --version`; skip the launcher when using Python directly |

Microsoft Word is needed only for legacy `.doc` files. PostgreSQL is defined in `compose.yaml` but the normal indexing and search path starts Qdrant only. Codex and an external MCP tunnel are optional clients, not prerequisites for the desktop UI.

Download installers from the [Python](https://www.python.org/downloads/), [NVIDIA driver](https://www.nvidia.com/Download/index.aspx), and [.NET 8](https://dotnet.microsoft.com/en-us/download/dotnet/8.0) sites. Use Microsoft's [WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install) and Docker's [Ubuntu Engine guide](https://docs.docker.com/engine/install/ubuntu/) for those two installations. RAG3's Python requirements select the CUDA 12.8 PyTorch wheels; check that your installed driver can run them before continuing.

## 1. Set up WSL and Docker

Open an administrator PowerShell to install WSL. Reboot if requested, complete Ubuntu's first-run setup, and make Ubuntu the default distribution:

```powershell
wsl --install -d Ubuntu
wsl --set-default Ubuntu
wsl --list --verbose
```

Install Docker Engine and the Compose plugin **in Ubuntu** using the linked Docker instructions. Check them from PowerShell:

```powershell
wsl -d Ubuntu -u root -- docker info
wsl -d Ubuntu -u root -- docker compose version
```

If `docker info` cannot reach the daemon, start it with `wsl -d Ubuntu -u root -- service docker start` and repeat the check.

The startup script invokes Docker through `wsl.exe -u root`; it does not call a Windows `docker.exe` client.

## 2. Clone and install Python packages

The current infrastructure script names `C:\python_program\RAG3\compose.yaml` and its WSL equivalent explicitly. Use that checkout path for this version.

```powershell
New-Item -ItemType Directory -Force C:\python_program | Out-Null
git clone https://github.com/nyamx/RAG3.git C:\python_program\RAG3
cd C:\python_program\RAG3
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m pip check
```

The requirements pin the main Python packages and the CUDA PyTorch build. They are not a complete transitive lockfile. Installation downloads packages from PyPI and the PyTorch wheel index.

## 3. Download models

Model weights are not committed to Git. Download them into the paths read by RAG3, or copy an existing `data/models/` directory into the same location:

```powershell
New-Item -ItemType Directory -Force data\models\huggingface, data\models\docling | Out-Null
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', cache_folder=r'data\models\huggingface')"
.\.venv\Scripts\docling-tools.exe models download -o data\models\docling
```

The BGE lookup scans `data/models/` for a local BGE-M3 checkpoint. Docling is configured to use `data/models/docling/`. Keep both directories when moving the installation to another PC.

## 4. Start Qdrant

Copy the Compose variable template and edit the password before starting PostgreSQL. Compose reads these variables even when only Qdrant is selected.

```powershell
Copy-Item .env.example .env
wsl -d Ubuntu -u root -- docker volume create rag3_qdrant_data
wsl -d Ubuntu -u root -- docker volume create rag3_postgres_data
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\ensure-infra.ps1
Invoke-RestMethod 'http://127.0.0.1:16333/collections'
```

The two Docker volumes are declared `external` in `compose.yaml`, so both must exist. The second volume does not mean PostgreSQL is started. Qdrant data lives in its Docker volume; copying the Git checkout alone does not copy indexed documents.

`compose.yaml` currently publishes Qdrant ports `16333` and `16334` without a host IP. Docker [binds that form to all host interfaces](https://docs.docker.com/reference/compose-file/services/#ports). Check the host's network exposure before putting real documents in Qdrant.

## 5. Check the environment and run RAG3

```powershell
.\.venv\Scripts\python.exe scripts\verify-dev-env.py
```

Look for `BGE_M3_DIM=1024` and `DEV_ENV_CORE_VERIFICATION=OK`. This check imports the parsing and retrieval packages, tests CUDA availability, and loads the local BGE checkpoint on CPU. It is not a full ingestion test.

Start the backend in its own PowerShell window:

```powershell
cd C:\python_program\RAG3
.\.venv\Scripts\python.exe scripts\rag3-mcp-server.py
```

In a second window, check health and open the UI:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health'
.\.venv\Scripts\python.exe scripts\ingest-ui.py
```

The backend owns jobs and the GPU model service. Keep its window open while using the UI. For a disposable end-to-end PDF check, see [verification.md](verification.md).

To connect Codex or ChatGPT on the web after the backend is healthy, follow [MCP clients](usage.md#mcp-clients). Those integrations are optional for local desktop use.

### Optional: build the launcher

With the .NET 8 SDK installed, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-launcher.ps1
.\RAG3.exe
```

`RAG3.exe` opens the UI; it does not replace the backend. Running the UI directly with Python avoids the .NET dependency.

### Optional: start the backend at Windows login

The scheduled-task setup is separate from the first-run steps above. Inspect the proposed action before registering it:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\manage-rag3-login.ps1 -Operation Inspect
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\manage-rag3-login.ps1 -Operation Plan
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\manage-rag3-login.ps1 -Operation Install
```

`Install` registers the task for a future login; it does not launch another backend immediately. The task's settings are saved in `data/backend-login-config.json`. The current local MCP plugin connects by HTTP to `http://127.0.0.1:8000/mcp`; see [Using RAG3](usage.md).

## If a step fails

- **No CUDA device:** check `nvidia-smi`, the driver, and the installed `torch` build with `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
- **Docker command works in Windows but Qdrant does not start:** run the `wsl -d Ubuntu -u root -- docker ...` checks above. The script needs Docker inside Ubuntu.
- **Model directory not found:** repeat the downloads into `data/models/`; Python packages do not include these model weights.
- **Health responds but the UI cannot connect:** verify that the running backend is the one from this checkout and that it listens on port 8000. The UI uses a local token issued by that backend.
- **Checkout moved to another path:** update both path literals in `scripts/ensure-infra.ps1` before starting Qdrant.

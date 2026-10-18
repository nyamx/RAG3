@echo off
setlocal
if not defined RAG3_PROJECT_ROOT (
  echo RAG3_PROJECT_ROOT is not configured. 1>&2
  exit /b 1
)
set "RAG3_PYTHON=%RAG3_PROJECT_ROOT%\.venv\Scripts\python.exe"
set "RAG3_SERVER=%RAG3_PROJECT_ROOT%\scripts\rag3-mcp-server.py"
if not exist "%RAG3_PYTHON%" (
  echo RAG3 virtual environment was not found. 1>&2
  exit /b 1
)
if not exist "%RAG3_SERVER%" (
  echo RAG3 MCP server was not found. 1>&2
  exit /b 1
)
"%RAG3_PYTHON%" "%RAG3_SERVER%"
exit /b %errorlevel%

$ErrorActionPreference = "Stop"

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$project = Join-Path $root "launcher\RAG3.Launcher.csproj"
$publish = Join-Path $root "launcher\publish"
$target = Join-Path $root "RAG3.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "RAG3 virtual environment is missing: $python"
}

& $python (Join-Path $root "launcher\make_icon.py")
if ($LASTEXITCODE -ne 0) { throw "Icon generation failed" }

& dotnet publish $project -c Release -o $publish --nologo
if ($LASTEXITCODE -ne 0) { throw "Launcher build failed" }

$built = Join-Path $publish "RAG3.exe"
if (-not (Test-Path -LiteralPath $built)) {
    throw "Published launcher is missing: $built"
}
Copy-Item -LiteralPath $built -Destination $target -Force
Write-Output "RAG3_LAUNCHER_READY $target"

# Serve the fine-tuned NABARD QA model on CPU with llama.cpp's OpenAI-compatible server.
# Base: Qwen2.5-1.5B-Instruct Q4_K_M; the LoRA adapter (training/ollama/nabard-qa-lora-f16.gguf)
# is applied at load time. Matches the app's local provider defaults (port 8001, model
# nabard-qa-lora), so use it with  NABARD_QA_LLM_PROVIDER=local  or  NABARD_QA_LLM_CHAIN=local,groq,gemini
#
#   powershell -ExecutionPolicy Bypass -File scripts\serve_local_model.ps1
param(
    [int]$Port = 8001,
    [int]$Context = 8192,
    [int]$Threads = [Math]::Max(1, [Environment]::ProcessorCount - 1)
)
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "runtime\llama-bin\llama-server.exe"
$base = Join-Path $root "runtime\models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$lora = Join-Path $root "training\ollama\nabard-qa-lora-f16.gguf"
foreach ($f in $server, $base, $lora) {
    if (-not (Test-Path $f)) { Write-Error "missing $f"; exit 1 }
}
& $server -m $base --lora $lora --alias nabard-qa-lora --host 127.0.0.1 --port $Port `
    -c $Context -t $Threads --temp 0.1 --jinja

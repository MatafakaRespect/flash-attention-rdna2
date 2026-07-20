# install.ps1 — поставить в указанное окружение обычным pip.
#
#   .\install.ps1 <путь к python вашего окружения>
#   .\install.ps1 <питон> -Editable      # ссылкой на репозиторий, без копирования
#   .\install.ps1 <питон> -NoCompat      # только движок, без имён оригинала
#
# Ставятся ДВА дистрибутива:
#   fable_attn  — движок: HIP-ядра, весь код, гейты, PLAN.md. Имя наше.
#   flash_attn  — ноль строк логики, только имена оригинала, ведущие в fable_attn.
#                 Нужен, чтобы чужой код шёл БЕЗ ЕДИНОЙ ПРАВКИ: `import flash_attn` работает
#                 сразу, а transformers находит пакет через importlib.metadata (ему мало
#                 __version__ внутри модуля — он спрашивает именно дистрибутив).
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [switch]$Editable,
    [switch]$NoCompat
)
$ErrorActionPreference = 'Stop'

# vcvars нужен для сборки расширения (MSVC + hipcc)
$vs = & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
& "$env:WINDIR\System32\cmd.exe" /c "`"$(Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat')`" >nul 2>&1 && set" |
  ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }

$env:MAX_JOBS = '5'
$env:DISTUTILS_USE_SDK = '1'
$env:PYTORCH_ROCM_ARCH = 'gfx1030'
Set-Location $PSScriptRoot

$ErrorActionPreference = 'Continue'

$base = @('-m', 'pip', 'install', '--no-deps', '--no-build-isolation')
if ($Editable) { $base += '-e' }

& $Python @base '.'
$e1 = $LASTEXITCODE
Write-Output "PIP_EXIT_fable_attn=$e1"

if (-not $NoCompat) {
    $compat = Join-Path (Split-Path $PSScriptRoot -Parent) 'compat'
    & $Python @base $compat
    Write-Output "PIP_EXIT_flash_attn=$LASTEXITCODE"
}

# Сборка fable_attn._C (torch ROCm-расширение). CPU-задача, очередь GPU не нужна.
$ErrorActionPreference = 'Stop'
$vs = & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
& "$env:WINDIR\System32\cmd.exe" /c "`"$(Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat')`" >nul 2>&1 && set" |
  ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
$env:MAX_JOBS = '5'
$env:DISTUTILS_USE_SDK = '1'
$env:PYTORCH_ROCM_ARCH = 'gfx1030'
Set-Location $PSScriptRoot

# Шимы (_fwd/_bwd/_dec_shim.cu) — три строки, весь код лежит в ..\kernels\*.hip, который они
# включают. ninja этой зависимости НЕ ВИДИТ: депфайла для hipcc-компиляции .cu нет, и правка
# ядра не пересобирает шим. Один раз это уже дало собранный .pyd со СТАРЫМИ ядрами и новым
# dispatch.obj — то есть разъехавшийся ABI структуры параметров и молча неверные результаты.
# Поэтому сверяем время руками и сносим устаревшие объектники.
$objDir = "build\temp.win-amd64-cpython-312\Release\csrc"
if (Test-Path $objDir) {
    $newestKernel = (Get-ChildItem "..\kernels\*.hip", "..\kernels\*.hpp", "..\kernels\*.inc" -EA SilentlyContinue |
                     Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
    if ($newestKernel) {
        foreach ($o in @("_fwd_shim.obj", "_bwd_shim.obj", "_dec_shim.obj")) {
            $p = Join-Path $objDir $o
            if ((Test-Path $p) -and ((Get-Item $p).LastWriteTime -lt $newestKernel)) {
                Write-Output "устарел (ядра новее): $o -> пересборка"
                Remove-Item $p -Force
            }
        }
    }
}

# Каким питоном собирать: FA_PY, иначе тот, что в PATH.
$py = if ($env:FA_PY) { $env:FA_PY } else { 'python' }
$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot "build.log"
& $py setup.py build_ext --inplace *>&1 | Out-File -FilePath $log -Encoding utf8
Write-Output "BUILD_EXIT=$LASTEXITCODE  log=$log"

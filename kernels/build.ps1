# build.ps1 — сборка гейтов fable_attn под gfx1030 (RX 6800 XT).
# Хост-лаунчеры (fa_forward_launch / fa_decode_launch, extern "C") компилируются без -DFA_STANDALONE
# для линковки во второй агент (torch-обёртка). Гейты — с -DFA_STANDALONE (свой main()).
$arch = "gfx1030"
$flags = "--offload-arch=$arch -O3 -ffast-math -std=c++17"
Write-Host "== forward gate =="
hipcc fa_forward.hip -o fa_forward.exe $flags.Split(' ') -DFA_STANDALONE
Write-Host "== decode gate =="
hipcc fa_decode.hip  -o fa_decode.exe  $flags.Split(' ') -DFA_STANDALONE
Write-Host "== запуск гейтов =="
./fa_forward.exe
./fa_decode.exe

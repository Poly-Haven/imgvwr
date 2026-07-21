# Generate the image-sequence test material for manual playback verification
# (plans/image_sequence_playback.md §12). The 150-frame base sequence at
# C:\tmp\imgvwr_test_files\sequence is assumed to already exist; everything else
# is derived from it (or from a committed fixture) so the whole set is
# reproducible on any machine with oiiotool + ImageMagick.
#
# Usage:  pwsh scripts\make_test_sequences.ps1 [-Root C:\tmp\imgvwr_test_files]
param(
    [string]$Root = 'C:\tmp\imgvwr_test_files'
)

$ErrorActionPreference = 'Stop'
$src = Join-Path $Root 'sequence'
$fixture = Join-Path $PSScriptRoot '..\tests\fixtures\tiny_rgb.png' | Resolve-Path

function Reset-Dir($name) {
    $d = Join-Path $Root $name
    New-Item -ItemType Directory -Force $d | Out-Null
    Get-ChildItem -LiteralPath $d -File | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    $d
}

function Frame($n) { Join-Path $src ("LS_Steam_01_Cam01.{0:d4}.png" -f $n) }

Write-Host 'seq_exr4k    — 4K float EXR, frames 1001-1030 (the memory-bound / eviction path)'
$d = Reset-Dir 'seq_exr4k'
foreach ($i in 0..29) {
    & oiiotool (Frame $i) --resize 4096x2304 -d float --compression zip `
        -o (Join-Path $d ("shot.{0:d4}.exr" -f (1001 + $i))) | Out-Null
}

Write-Host 'seq_holes    — 100 frames with a 20-frame gap, two lone holes, one corrupt frame'
$d = Reset-Dir 'seq_holes'
foreach ($i in 0..99) {
    if (($i -ge 40 -and $i -lt 60) -or $i -eq 12 -or $i -eq 83) { continue }
    Copy-Item -LiteralPath (Frame $i) -Destination (Join-Path $d ("hole.{0:d4}.png" -f $i))
}
[byte[]](1..64) | Set-Content -LiteralPath (Join-Path $d 'hole.0070.png') -AsByteStream

Write-Host 'seq_unpadded — unpadded, opened at frame 10 (must still find frames 1-9)'
$d = Reset-Dir 'seq_unpadded'
foreach ($i in 1..15) { Copy-Item -LiteralPath (Frame $i) -Destination (Join-Path $d ("f{0}.png" -f $i)) }

Write-Host 'seq_overflow — 4-padded, running 9997-10004 (padding overflow, one sequence)'
$d = Reset-Dir 'seq_overflow'
foreach ($i in 0..7) {
    $n = 9997 + $i
    $name = if ($n -lt 10000) { 'ov.{0:d4}.png' -f $n } else { "ov.$n.png" }
    Copy-Item -LiteralPath (Frame $i) -Destination (Join-Path $d $name)
}

Write-Host 'seq_long_vis — 800 frames at 640x360 with a lone hole (cache-bar bucketing)'
$d = Reset-Dir 'seq_long_vis'
$tmp = Join-Path $env:TEMP 'imgvwr_base360.png'
& oiiotool (Frame 0) --resize 640x360 -o $tmp | Out-Null
$bytes = [System.IO.File]::ReadAllBytes($tmp)
foreach ($n in 0..799) { if ($n -eq 400) { continue }; [System.IO.File]::WriteAllBytes((Join-Path $d ('v.{0:d4}.png' -f $n)), $bytes) }

Write-Host 'seq_pano     — 12-frame 2:1 HDR panorama sequence (varying exposure)'
$d = Reset-Dir 'seq_pano'
$panosrc = Join-Path $Root 'brown_photostudio_02\brown_photostudio_02_2k.exr'
foreach ($i in 0..11) {
    & oiiotool $panosrc --mulc ([string](0.6 + $i * 0.08)) -d half `
        -o (Join-Path $d ('pano.{0:d4}.exr' -f (1 + $i))) | Out-Null
}

Write-Host 'anim_test.gif        — a 6-frame animated GIF (in-memory frame source)'
& magick -delay 12 -loop 0 -size 320x240 xc:red xc:green xc:blue xc:yellow xc:magenta xc:cyan `
    (Join-Path $Root 'anim_test.gif')
Write-Host 'anim_vardelay.gif    — a variable-rate GIF (50/500/50 ms), for source-timed playback'
& magick -size 64x48 -delay 5 xc:red -delay 50 xc:green -delay 5 xc:blue -loop 0 `
    (Join-Path $Root 'anim_vardelay.gif')

Write-Host ''
Write-Host 'Done. Not covered here (needs the caller): a budget-starved run — set'
Write-Host 'IMGVWR_DEBUG_CACHE_MB low, e.g.  IMGVWR_DEBUG_CACHE_MB=400 on seq_exr4k.'

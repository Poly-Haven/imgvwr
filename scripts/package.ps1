# Package a self-contained Windows distribution: imgvwr.exe + the full runtime
# DLL closure (staged beside the exe by build.rs) + resources + licenses, zipped.
# See plans/rewrite.md §19/§20.
#
# Usage:  pwsh scripts/package.ps1 [-Profile release] [-OutDir dist]
[CmdletBinding()]
param(
    [string]$Profile = 'release',
    [string]$OutDir = 'dist'
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "target\$Profile"
$exe = Join-Path $target 'imgvwr.exe'
if (-not (Test-Path $exe)) {
    throw "imgvwr.exe not found at $exe - run `cargo build --$Profile` first."
}

# Version from Cargo.toml ([package] version = "x.y.z").
$cargo = Get-Content (Join-Path $root 'Cargo.toml')
$version = ($cargo | Select-String '^\s*version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
$name = "imgvwr-$version-windows-x64"

$stage = Join-Path $root "$OutDir\$name"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# 1. Executable.
Copy-Item $exe (Join-Path $stage 'imgvwr.exe')

# 2. Runtime DLL closure (build.rs staged the vcpkg bin DLLs beside the exe).
$dlls = Get-ChildItem (Join-Path $target '*.dll') -ErrorAction SilentlyContinue
if (-not $dlls) {
    Write-Warning "No DLLs found beside the exe; the OCIO/OpenEXR closure may be missing."
}
$dlls | ForEach-Object { Copy-Item $_.FullName $stage }

# 3. MSVC runtime (redistributable) for clean machines without the VC++ redist.
foreach ($rt in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') {
    $sys = Join-Path $env:SystemRoot "System32\$rt"
    if ((Test-Path $sys) -and -not (Test-Path (Join-Path $stage $rt))) {
        Copy-Item $sys $stage
    }
}

# 4. Resources + licenses + readme.
Copy-Item (Join-Path $root 'resources') (Join-Path $stage 'resources') -Recurse
foreach ($doc in 'THIRD_PARTY_LICENSES.md', 'LICENSE', 'README.md') {
    Copy-Item (Join-Path $root $doc) $stage
}

# 5. Zip.
$zip = Join-Path $root "$OutDir\$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal

$sizeMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Output "Packaged $zip ($sizeMb MB)"
Write-Output "Staged $((Get-ChildItem $stage -Recurse -File).Count) files; $($dlls.Count) bundled DLLs."

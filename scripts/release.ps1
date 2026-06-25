# Build, package, and (optionally) publish a GitHub release for the version in
# Cargo.toml. Releases are built locally because the native deps (OpenColorIO /
# OpenEXR) compile from source via vcpkg, which is too slow for CI.
#
# Run from a shell with VCPKG_ROOT + LIBCLANG_PATH set (see the README
# "Building from source" prerequisites).
#
# Usage:
#   pwsh scripts/release.ps1            # build + package only -> dist\
#   pwsh scripts/release.ps1 -Publish   # also tag vX.Y.Z and create the GitHub release
[CmdletBinding()]
param(
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot

# Version (and therefore the tag) come from Cargo.toml -- the single source of
# truth. Bump it before cutting a release.
$version = ((Get-Content (Join-Path $root 'Cargo.toml')) |
    Select-String '^\s*version\s*=\s*"([^"]+)"' |
    Select-Object -First 1).Matches.Groups[1].Value
$tag = "v$version"
$zip = Join-Path $root "dist\imgvwr-$version-windows-x64.zip"

Write-Host "Releasing imgvwr $tag" -ForegroundColor Cyan

Push-Location $root
try {
    # 1. Build the optimised binary (build.rs stages the runtime DLL closure).
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw 'cargo build --release failed' }

    # 2. Bundle the self-contained zip.
    & (Join-Path $PSScriptRoot 'package.ps1') -Profile release -OutDir dist
    if (-not (Test-Path $zip)) { throw "package.ps1 did not produce $zip" }

    if (-not $Publish) {
        Write-Host "Built $zip" -ForegroundColor Green
        Write-Host 'Re-run with -Publish to tag and create the GitHub release.' -ForegroundColor Yellow
        return
    }

    # 3. Tag the current commit (skip if the tag already exists) and push it.
    if (-not (git tag --list $tag)) {
        git tag -a $tag -m "imgvwr $tag"
        if ($LASTEXITCODE -ne 0) { throw "git tag $tag failed" }
    }
    git push origin $tag
    if ($LASTEXITCODE -ne 0) { throw "git push origin $tag failed" }

    # 4. Create the GitHub release with the zip attached.
    $notes = @'
Self-contained Windows x64 build of imgvwr. Extract the zip and run `imgvwr.exe` - no prerequisites required.

Because the binary is unsigned, Windows SmartScreen may prompt on first launch (More info -> Run anyway).
'@
    gh release create $tag $zip --title "imgvwr $tag" --notes $notes
    if ($LASTEXITCODE -ne 0) { throw 'gh release create failed' }
    Write-Host "Published $tag" -ForegroundColor Green
}
finally {
    Pop-Location
}

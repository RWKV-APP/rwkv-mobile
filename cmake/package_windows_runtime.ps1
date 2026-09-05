param([Parameter(Mandatory=$true)][string]$Library)
$ErrorActionPreference = 'Stop'
$libraryPath = (Resolve-Path -LiteralPath $Library).Path
$compilerBin = Split-Path (Get-Command clang).Source
$readobj = Join-Path $compilerBin 'llvm-readobj.exe'
$imports = & $readobj --coff-imports $libraryPath
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the native DLL dependencies' }
$runtimeNames = @($imports | ForEach-Object {
    if ($_ -match 'Name: ((?:libomp|libiomp)[\w.\-]*\.dll)') { $Matches[1] }
} | Sort-Object -Unique)
foreach ($name in $runtimeNames) {
    $runtime = Join-Path $compilerBin $name
    if (!(Test-Path -LiteralPath $runtime)) { throw "Missing compiler runtime: $runtime" }
    Copy-Item -LiteralPath $runtime -Destination (Split-Path $libraryPath)
}
Write-Output "Packaged OpenMP dependencies: $($runtimeNames -join ', ')"

param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = $PSScriptRoot
$Tag = 'madhav-rag:mc3-v3'
$TranscriptStarted = $false
Set-Location -LiteralPath $Root
function Assert-Exit([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed (exit $LASTEXITCODE). Send the output; do not continue manually." }
}
function Verify-ModelFile([string]$Path, $Entry) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Entry.bytes) { return $false }
    if ($Entry.PSObject.Properties.Name -contains 'sha256') {
        Write-Host "Checking SHA256: $($Entry.name)"
        $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $Entry.sha256) { throw "Checksum mismatch: $Path. Send this error; no other model files were touched." }
    }
    return $true
}
try {
    Start-Transcript -Path (Join-Path $Root 'BUILD_LOG.txt') -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host 'MADHAV RAG Qwen3 - WINDOWS LAPTOP build'
    Write-Host 'New weights: 8.887 GB (8.277 GiB), downloaded on THIS LAPTOP.'
    Write-Host 'Extra package download: approximately 50-150 MB if not cached.'
    Write-Host 'No Torch install, no Docker push, no Challenge 2 modifications.'
    Get-Command docker.exe -ErrorAction Stop | Out-Null
    Get-Command curl.exe -ErrorAction Stop | Out-Null
    $osType = (& docker.exe info --format '{{.OSType}}').Trim()
    Assert-Exit 'Docker Desktop check'
    if ($osType -ne 'linux') { throw 'Docker Desktop must use Linux containers.' }
    $hashes = Get-Content -LiteralPath (Join-Path $Root 'code-hashes.json') -Raw | ConvertFrom-Json
    foreach ($property in $hashes.PSObject.Properties) {
        $p = Join-Path $Root $property.Name
        if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) {
            throw "Code integrity check failed: $($property.Name). Extract a fresh copy of this package."
        }
    }
    Write-Host 'Tested code integrity: PASS'
    $available = @(& docker.exe image ls --format '{{.Repository}}:{{.Tag}}')
    Assert-Exit 'Local image list'
    $known = @('madhav-rag:mc3-v2','jiokoi8/madhav-rag:mc3-v2','madhav-ocr:mc2-v1','jiokoi8/madhav-ocr:mc2-v1','rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0')
    if (-not ($known | Where-Object { $available -contains $_ })) {
        throw 'No known cached AMD image found. Stop here and send docker image ls output, to avoid an unexpected large base download.'
    }
    Write-Host 'Existing AMD image found. Base-layer cache reuse expected; Docker still checks registry metadata.'
    $manifest = Get-Content -LiteralPath (Join-Path $Root 'model-download.json') -Raw | ConvertFrom-Json
    if ($manifest.repo -ne 'Qwen/Qwen3-VL-4B-Instruct' -or $manifest.revision -ne 'ebb281ec70b05090aa6165b016eac8ec08e71b17') {
        throw 'Unexpected model manifest.'
    }
    $modelDir = Join-Path $Root 'models\qwen'
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    if (-not $CheckOnly) {
        $drive = (Get-Item -LiteralPath $Root).PSDrive
        if ($null -eq $drive.Free) { throw 'Could not check free disk space. Use a local Windows drive.' }
        $freeGiB = [math]::Round($drive.Free / 1GB, 2)
        Write-Host "Project drive free space: $freeGiB GiB"
        if ($drive.Free -lt 35GB) {
            throw 'Keep at least 35 GiB free on the project drive for weights, build context and new Docker layers. Send this output; do not delete old challenge images.'
        }
        Write-Host 'Check Docker Desktop disk-image location also has enough free space if it is on another drive.'
        foreach ($entry in $manifest.files) {
            if ($entry.name -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid model filename.' }
            $destination = Join-Path $modelDir $entry.name
            if (Verify-ModelFile $destination $entry) {
                Write-Host "Already complete: $($entry.name)"
                continue
            }
            if (Test-Path -LiteralPath $destination) {
                throw "Existing incomplete final file: $destination. Send this output before overwriting it."
            }
            $partial = "$destination.part"
            $completePartial = $false
            if (Test-Path -LiteralPath $partial) {
                $length = (Get-Item -LiteralPath $partial).Length
                if ($length -gt [long]$entry.bytes) { throw "Oversized partial file: $partial" }
                if ($length -eq [long]$entry.bytes) { $completePartial = Verify-ModelFile $partial $entry }
            }
            if (-not $completePartial) {
                $url = "https://huggingface.co/$($manifest.repo)/resolve/$($manifest.revision)/$($entry.name)"
                Write-Host "Downloading/resuming: $($entry.name)"
                & curl.exe --location --fail --retry 4 --retry-delay 5 --connect-timeout 30 --speed-limit 1024 --speed-time 120 --continue-at - --output $partial $url
                Assert-Exit "Download $($entry.name)"
                if (-not (Verify-ModelFile $partial $entry)) { throw "Incomplete download: $($entry.name). Rerun this script to resume." }
            }
            Move-Item -LiteralPath $partial -Destination $destination
        }
        Write-Host 'MODEL FILES + WEIGHT CHECKSUMS: PASS'
        Write-Host 'Building madhav-rag:mc3-v3. Old tags stay unchanged.'
        & docker.exe build --platform linux/amd64 --progress plain -t $Tag $Root
        Assert-Exit 'Docker build'
    }
    $history = @(& docker.exe image history --no-trunc --human=false --format '{{.Size}}' $Tag)
    Assert-Exit 'Uncompressed layer size check'
    [decimal]$total = 0
    foreach ($line in $history) {
        $number = $line.Trim()
        if ($number -notmatch '^\d+$') { throw "Unexpected Docker layer size: $number" }
        $total += [decimal]$number
    }
    $gib = [math]::Round($total / 1GB, 3)
    Write-Host "UNCOMPRESSED LAYER TOTAL: $gib GiB | LIMIT: 60 GiB"
    if ($total -le 0 -or $total -gt 60GB) { throw 'Image size gate failed.' }
    Write-Host 'Running offline packaging check (CPU host; GPU inference is not tested here)...'
    Get-Content -LiteralPath (Join-Path $Root 'offline_check.py') -Raw | & docker.exe run --rm -i --network none --cap-drop DAC_OVERRIDE --entrypoint python3 $Tag -
    Assert-Exit 'Offline packaging check'
    $id = (& docker.exe image inspect --format '{{.Id}}' $Tag).Trim()
    Assert-Exit 'Image ID check'
    $report = [ordered]@{
        status = 'LOCAL_BUILD_AND_OFFLINE_PACKAGING_PASS'
        tag = $Tag
        image_id = $id
        uncompressed_layer_GiB = $gib
        model = $manifest.repo
        revision = $manifest.revision
        docker_gpu_inference_tested = $false
        published = $false
        checked_at = (Get-Date).ToString('o')
    }
    $report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'BUILD_REPORT.json') -Encoding UTF8
    Write-Host ''
    Write-Host 'LOCAL BUILD + OFFLINE PACKAGING: PASS'
    Write-Host 'Docker GPU inference: NOT TESTED. Nothing was pushed or submitted.'
    Write-Host 'Upload BUILD_REPORT.json and BUILD_LOG.txt to this chat.'
} catch {
    Write-Host ''
    Write-Host ('STOPPED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Send the last output or BUILD_LOG.txt. Do not reinstall Torch or delete old challenge files.'
    exit 1
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}

# OCR every page image (*.png) in a folder with the OCR engine built into Windows 10/11
# (no Tesseract or model download). Writes <page>.json next to each image:
#   [{"text": "...", "x": .., "y": .., "w": .., "h": ..}, ...]   one entry per line, pixel units.
# Called by scripts/make_searchable_pdf.py; usable on its own: .\scripts\ocr_pages.ps1 <dir>
param([Parameter(Mandatory)][string]$Dir)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) {
    $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result
}
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $engine) { throw 'No Windows OCR language installed (Settings > Time & language > Language)' }

Get-ChildItem $Dir -Filter *.png | Sort-Object Name | ForEach-Object {
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($_.FullName)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bmp = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $res = Await ($engine.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
    } finally { $stream.Dispose() }
    $lines = @(foreach ($l in $res.Lines) {
        $r = $l.Words | ForEach-Object { $_.BoundingRect }
        $x0 = ($r | Measure-Object X -Minimum).Minimum
        $y0 = ($r | Measure-Object Y -Minimum).Minimum
        $x1 = ($r | ForEach-Object { $_.X + $_.Width } | Measure-Object -Maximum).Maximum
        $y1 = ($r | ForEach-Object { $_.Y + $_.Height } | Measure-Object -Maximum).Maximum
        [ordered]@{ text = $l.Text; x = $x0; y = $y0; w = $x1 - $x0; h = $y1 - $y0 }
    })
    $out = [IO.Path]::ChangeExtension($_.FullName, '.json')
    [IO.File]::WriteAllText($out, (ConvertTo-Json -InputObject $lines -Depth 3 -Compress), [Text.UTF8Encoding]::new($false))
    Write-Output "$($_.Name): $($lines.Count) lines"
}

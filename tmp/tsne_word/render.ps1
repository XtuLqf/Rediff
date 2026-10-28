$ErrorActionPreference = 'Stop'
$app = $null
$doc = $null
$created = $false
try {
    try { $app = [Runtime.InteropServices.Marshal]::GetActiveObject('kwps.Application') }
    catch { $app = New-Object -ComObject Word.Application; $created = $true }
    $missing = [Type]::Missing
    $doc = $app.Documents.Open('D:\GitCode\Rediff\tmp\tsne_word\edited.docx', $false, $true, $false, $missing, $missing, $missing, $missing, $missing, $missing, $missing, $false)
    $doc.ExportAsFixedFormat('D:\GitCode\Rediff\tmp\tsne_word\edited.pdf', 17)
    Write-Output ('Pages: ' + $doc.ComputeStatistics(2))
} finally {
    if ($null -ne $doc) { try { $doc.Close(0) } catch {} }
    if ($created -and $null -ne $app) { try { $app.Quit() } catch {} }
}

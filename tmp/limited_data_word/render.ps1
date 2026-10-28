$ErrorActionPreference = 'Stop'
$app = $null
$doc = $null
try {
    $app = New-Object -ComObject Word.Application
    $app.Visible = $false
    $app.DisplayAlerts = 0
    $app.AutomationSecurity = 3
    $doc = $app.Documents.Open('D:\GitCode\Rediff\tmp\limited_data_word\edited.docx', $false, $true, $false)
    $doc.ExportAsFixedFormat('D:\GitCode\Rediff\tmp\limited_data_word\edited.pdf', 17)
    Write-Output ('Pages: ' + $doc.ComputeStatistics(2))
} finally {
    if ($null -ne $doc) { try { $doc.Close(0) } catch {} }
    if ($null -ne $app) { try { $app.Quit() } catch {} }
}

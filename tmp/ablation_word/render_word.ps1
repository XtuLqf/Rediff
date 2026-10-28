$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $doc = $word.Documents.Open('D:\GitCode\Rediff\tmp\ablation_word\edited.docx', $false, $true, $false)
    $doc.ExportAsFixedFormat('D:\GitCode\Rediff\tmp\ablation_word\render\edited.pdf', 17)
    Write-Output ('Pages: ' + $doc.ComputeStatistics(2))
} finally {
    if ($null -ne $doc) { $doc.Close(0) }
    if ($null -ne $word) { $word.Quit() }
}

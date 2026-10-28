$ErrorActionPreference = 'Stop'
$targetPath = 'D:\.湘潭大学学习文档\小论文\初稿-中文.docx'
$app = [Runtime.InteropServices.Marshal]::GetActiveObject('kwps.Application')
$target = $null
foreach ($document in $app.Documents) {
    if ($document.FullName -eq $targetPath) { $target = $document; break }
}
if ($null -eq $target) { throw 'The open draft was not found.' }
if (-not $target.Saved) { throw 'The draft has unsaved edits; insertion stopped.' }
if ($target.Tables.Count -ne 1) { throw 'Unexpected number of existing tables; insertion stopped.' }
$heading = $null
for ($i = 1; $i -le $target.Paragraphs.Count; $i++) {
    if ($target.Paragraphs.Item($i).Range.Text.Trim() -eq 'Ablation Study') { $heading = $i; break }
}
if ($null -eq $heading) { throw 'Ablation heading not found.' }
$placeholder = $target.Paragraphs.Item($heading + 1)
$insertion = $target.Paragraphs.Item($heading + 2).Range.Duplicate
if ($insertion.Text.Trim().Length -ne 0) { throw 'The insertion paragraph is no longer blank.' }
$staged = $null
try {
    $missing = [Type]::Missing
    $staged = $app.Documents.Open('D:\GitCode\Rediff\tmp\ablation_word\edited.docx', $false, $true, $false, $missing, $missing, $missing, $missing, $missing, $missing, $missing, $false)
    $insertion.FormattedText = $staged.Tables.Item(2).Range.FormattedText
    $placeholder.Format.KeepWithNext = -1
    if ($target.Tables.Count -ne 2) { throw 'Table insertion verification failed.' }
    $target.Save()
    $target.ExportAsFixedFormat('D:\GitCode\Rediff\tmp\ablation_word\render\saved.pdf', 17)
    Write-Output ('SAVED ' + $target.FullName + ' TABLES ' + $target.Tables.Count)
} finally {
    if ($null -ne $staged) { $staged.Close(0) }
}

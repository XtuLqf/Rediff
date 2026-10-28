$ErrorActionPreference = 'Continue'
foreach ($progid in @('Word.Application','kwps.Application','wps.Application')) {
    try {
        $app = [Runtime.InteropServices.Marshal]::GetActiveObject($progid)
        Write-Output ('APP ' + $progid)
        foreach ($d in $app.Documents) { Write-Output ('DOC ' + $d.FullName + ' SAVED ' + $d.Saved) }
    } catch { Write-Output ($progid + ': ' + $_.Exception.Message) }
}

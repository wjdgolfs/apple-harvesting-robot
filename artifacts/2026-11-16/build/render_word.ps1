param([string]$InputDocx,[string]$PdfPath,[string]$PageTextPath)
$ErrorActionPreference = 'Stop'
$wordApp = $null
$wordDoc = $null
try {
 $wordApp = New-Object -ComObject Word.Application
 $wordApp.Visible = $false
 $wordApp.DisplayAlerts = 0
 $wordDoc = $wordApp.Documents.Open($InputDocx,$false,$true)
 $wordDoc.Repaginate()
 $count = $wordDoc.ComputeStatistics(2)
 $wordDoc.ExportAsFixedFormat($PdfPath,17)
 $pageRecords = @()
 for ($pageIndex=1; $pageIndex -le $count; $pageIndex++) {
  $pageRange = $wordDoc.GoTo(1,1,$pageIndex)
  $pageStart = $pageRange.Start
  if ($pageIndex -lt $count) { $pageEnd = $wordDoc.GoTo(1,1,($pageIndex+1)).Start } else { $pageEnd = $wordDoc.Content.End }
  $pageRecords += @{ page=$pageIndex; text=$wordDoc.Range($pageStart,$pageEnd).Text }
 }
 $pageRecords | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PageTextPath -Encoding UTF8
 Write-Output "Rendered pages: $count"
} finally {
 if ($null -ne $wordDoc) { $wordDoc.Close(0) }
 if ($null -ne $wordApp) { $wordApp.Quit() }
}

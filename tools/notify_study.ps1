param(
    [Parameter(Mandatory=$true)][string]$PayloadPath
)

# Receive text through JSON, never through a constructed shell command.
$ErrorActionPreference = 'Stop'
$payload = Get-Content -LiteralPath $PayloadPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $payload.title -or -not $payload.message) { throw 'Notification title and message are required.' }
$notificationTitle = [string]$payload.title
$notificationMessage = [string]$payload.message
$notificationTag = [string]$payload.tag
if ($notificationTag.Length -gt 16) { throw 'Notification tag must contain at most 16 characters.' }
$toastError = $null

try {
    [Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime] > $null
    [Windows.UI.Notifications.ToastNotification,Windows.UI.Notifications,ContentType=WindowsRuntime] > $null
    [Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime] > $null
    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    $toastXml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $toastXml.LoadXml('<toast duration="long"><visual><binding template="ToastGeneric"><text/><text/><text/></binding></visual></toast>')
    $textNodes = $toastXml.GetElementsByTagName('text')
    $null = $textNodes.Item(0).AppendChild($toastXml.CreateTextNode($notificationTitle))
    $null = $textNodes.Item(1).AppendChild($toastXml.CreateTextNode($notificationMessage))
    $null = $textNodes.Item(2).AppendChild($toastXml.CreateTextNode([string]$payload.report_path))
    $toast = [Windows.UI.Notifications.ToastNotification]::new($toastXml)
    $toast.Tag = $notificationTag
    $toast.Group = 'StockTwitsStudy'
    $toast.ExpirationTime = [DateTimeOffset]::Now.AddDays(3)
    $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId)
    $notifier.Show($toast)
    # Check history to avoid claiming delivery when an unregistered app silently fails.
    $accepted = $false
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Start-Sleep -Milliseconds 200
        $history = [Windows.UI.Notifications.ToastNotificationManager]::History.GetHistory($appId)
        if (@($history | Where-Object { $_.Tag -eq $notificationTag -and $_.Group -eq 'StockTwitsStudy' }).Count -gt 0) {
            $accepted = $true
            break
        }
    }
    if (-not $accepted) { throw 'Toast was not found in Windows notification history.' }
    @{accepted=$true; method='toast'; tag=$notificationTag} | ConvertTo-Json -Compress
    exit 0
} catch {
    $toastError = $_.Exception.Message
}

# Shell notifications also work when this Windows installation has no registered
# PowerShell toast identity. Keep the icon alive while the shell raises the alert.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$icon = New-Object System.Windows.Forms.NotifyIcon
$shown = $false
$handler = [System.EventHandler]{ $script:shown = $true }
try {
    $icon.Icon = [System.Drawing.SystemIcons]::Information
    $icon.Text = 'StockTwits research'
    $icon.BalloonTipTitle = $notificationTitle
    $icon.BalloonTipText = $notificationMessage + "`n" + [string]$payload.report_path
    $icon.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
    $icon.add_BalloonTipShown($handler)
    $icon.Visible = $true
    $icon.ShowBalloonTip(30000)
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while (-not $shown -and [DateTime]::UtcNow -lt $deadline) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 100
    }
    if (-not $shown) { throw "Windows did not acknowledge the notification. Toast error: $toastError" }
    # Give the user time to read a transient shell banner before disposing the icon.
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    while ([DateTime]::UtcNow -lt $deadline) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 100
    }
    @{accepted=$true; method='shell_balloon'; tag=$notificationTag; toast_error=$toastError} | ConvertTo-Json -Compress
} finally {
    $icon.Visible = $false
    $icon.remove_BalloonTipShown($handler)
    $icon.Dispose()
}

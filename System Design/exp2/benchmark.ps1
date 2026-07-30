# quick benchmark script for the url shortener
# just runs each endpoint 100 times and prints the average response time
# run this in a normal VS Code/PowerShell terminal: .\benchmark.ps1
# (make sure the app is already running first - mvn spring-boot:run)

$BaseUrl = "http://localhost:8080"
$Runs = 100

# quick check the app is even up before wasting 200 requests on nothing
# just checks the port is open, doesnt care about http status codes at all - simpler and reliable
$tcp = New-Object System.Net.Sockets.TcpClient
try {
    $tcp.Connect("localhost", 8080)
} catch {
    Write-Host "cant reach $BaseUrl - is the app running? (mvn spring-boot:run)"
    exit 1
} finally {
    $tcp.Close()
}

# times one request and returns elapsed ms, or $null if the request failed
# (keeping failures instead of crashing the whole script on one bad request)
function Time-Request {
    param([scriptblock]$Request)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Request | Out-Null
        $sw.Stop()
        return $sw.Elapsed.TotalMilliseconds
    } catch {
        $sw.Stop()
        return $null
    }
}

Write-Host "running $Runs x POST /shorten ..."
$postTimes = @()
$postFailed = 0
for ($i = 1; $i -le $Runs; $i++) {
    # unique url every time so nothing collides with a previous run's data
    $body = @{ longUrl = "https://example.com/benchmark-$(Get-Random)-$i" } | ConvertTo-Json
    $ms = Time-Request { Invoke-RestMethod -Uri "$BaseUrl/shorten" -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop }
    if ($null -ne $ms) { $postTimes += $ms } else { $postFailed++ }
}

# need one real shortcode to hammer with GET requests
$shortCode = $null
$shortenBody = @{ longUrl = "https://example.com/get-benchmark-target-$(Get-Random)" } | ConvertTo-Json
try {
    $shortenResponse = Invoke-RestMethod -Uri "$BaseUrl/shorten" -Method Post -Body $shortenBody -ContentType "application/json" -ErrorAction Stop
    $shortCode = $shortenResponse.shortUrl.Split("/")[-1]
} catch {
    Write-Host "couldn't create a shortcode to test GET with - stopping here"
    exit 1
}

Write-Host "running $Runs x GET /$shortCode ..."
$getTimes = @()
$getFailed = 0
for ($i = 1; $i -le $Runs; $i++) {
    # using raw HttpWebRequest here instead of Invoke-WebRequest - Invoke-WebRequest gets
    # inconsistent about whether a 302 counts as an "error" depending on PowerShell version,
    # this way just gives us the status code directly with no surprises
    $req = [System.Net.HttpWebRequest]::Create("$BaseUrl/$shortCode")
    $req.AllowAutoRedirect = $false
    $req.Method = "GET"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = $req.GetResponse()
        $sw.Stop()
        $resp.Close()
        $getTimes += $sw.Elapsed.TotalMilliseconds
    } catch [System.Net.WebException] {
        $sw.Stop()
        # a 302 still lands here since its not a 2xx, but it IS the success case for this endpoint
        if ($_.Exception.Response.StatusCode.value__ -eq 302) {
            $getTimes += $sw.Elapsed.TotalMilliseconds
        } else {
            $getFailed++
        }
    }
}

Write-Host ""
Write-Host "===== results ====="
if ($postTimes.Count -gt 0) {
    $postAvg = ($postTimes | Measure-Object -Average).Average
    Write-Host ("POST /shorten      average: {0:N2} ms  ({1}/{2} requests succeeded)" -f $postAvg, $postTimes.Count, $Runs)
} else {
    Write-Host "POST /shorten - all requests failed, no average to show"
}
if ($getTimes.Count -gt 0) {
    $getAvg = ($getTimes | Measure-Object -Average).Average
    Write-Host ("GET  /{{shortCode}}  average: {0:N2} ms  ({1}/{2} requests succeeded)" -f $getAvg, $getTimes.Count, $Runs)
} else {
    Write-Host "GET /{shortCode} - all requests failed, no average to show"
}
Write-Host "===================="

if ($postFailed -gt 0 -or $getFailed -gt 0) {
    Write-Host ""
    Write-Host "note: $postFailed POST(s) and $getFailed GET(s) failed and were excluded from the average."
    Write-Host "if a lot failed, check the app's console for errors (e.g. duplicate short codes after many runs without restarting the app)."
}

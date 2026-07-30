# quick benchmark script for the url shortener + redis cache
# runs POST 100 times, and GET 100 times on the SAME shortcode, then prints averages
# plus the real cache hit rate / miss rate, read straight from redis's own stats
# run this in a normal VS Code/PowerShell terminal: .\benchmark.ps1
# (make sure the app is already running first - mvn spring-boot:run - and mongo + redis/memurai are both up)
# (redis is set to a 20mb maxmemory cap with allkeys-lru eviction, see UrlShortenerController.java
# bottom-of-file comments or README.md for how that's configured)

# unlike exp2, GET here is not "just a db read" every time - its cache-aside.
# so hitting the SAME shortcode 100 times in a row means:
#   request 1   = cache MISS (goes to mongo, then writes into redis)
#   requests 2-100 = cache HITS (comes straight from redis)
# reporting those separately is way more useful here than one blended average,
# since that miss-vs-hit gap is literally the whole point of this experiment

$BaseUrl = "http://localhost:8080"
$Runs = 100
$MemuraiCli = "C:\Program Files\Memurai\memurai-cli.exe"

# reads redis's own hit/miss counters (INFO stats) - these count every GET .get() call
# our controller ever makes, so this is real measured data, not just us guessing from
# "request 1 must be the miss" - redis is literally counting this itself
function Get-RedisKeyspaceStats {
    $lines = & $MemuraiCli INFO stats
    $hits = ($lines | Select-String "^keyspace_hits:(\d+)").Matches.Groups[1].Value
    $misses = ($lines | Select-String "^keyspace_misses:(\d+)").Matches.Groups[1].Value
    return @{ Hits = [int]$hits; Misses = [int]$misses }
}

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

Write-Host "running $Runs x GET /$shortCode (request 1 = cache miss, rest = cache hits) ..."
$getTimes = @()
$getFailed = 0
$statsBefore = Get-RedisKeyspaceStats
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
$statsAfter = Get-RedisKeyspaceStats

# only count the hits/misses that happened DURING this run, not redis's all-time totals
# (redis keeps these counters running since the last restart, not per-script-run)
$hitsThisRun = $statsAfter.Hits - $statsBefore.Hits
$missesThisRun = $statsAfter.Misses - $statsBefore.Misses
$totalLookups = $hitsThisRun + $missesThisRun

Write-Host ""
Write-Host "===== results ====="
if ($postTimes.Count -gt 0) {
    $postAvg = ($postTimes | Measure-Object -Average).Average
    Write-Host ("POST /shorten            average: {0:N2} ms  ({1}/{2} requests succeeded)" -f $postAvg, $postTimes.Count, $Runs)
} else {
    Write-Host "POST /shorten - all requests failed, no average to show"
}

if ($getTimes.Count -gt 0) {
    # first successful GET is the cache miss, everything after is a cache hit
    $missTime = $getTimes[0]
    $hitTimes = $getTimes | Select-Object -Skip 1
    $overallAvg = ($getTimes | Measure-Object -Average).Average

    Write-Host ("GET  /{{shortCode}} overall  average: {0:N2} ms  ({1}/{2} requests succeeded)" -f $overallAvg, $getTimes.Count, $Runs)
    Write-Host ("  - request 1 (cache MISS, from mongo):  {0:N2} ms" -f $missTime)
    if ($hitTimes.Count -gt 0) {
        $hitAvg = ($hitTimes | Measure-Object -Average).Average
        Write-Host ("  - requests 2-$($getTimes.Count) (cache HITS, from redis) average: {0:N2} ms" -f $hitAvg)
        Write-Host ("  - cache made it about {0:N1}x faster" -f ($missTime / $hitAvg))
    }

    Write-Host ""
    if ($totalLookups -gt 0) {
        # this comes straight from redis's own INFO stats counters, not from us assuming
        # "1 miss + 99 hits" - so it double-checks our request-1-vs-rest assumption above
        $hitRate = ($hitsThisRun / $totalLookups) * 100
        $missRate = ($missesThisRun / $totalLookups) * 100
        Write-Host ("cache hit rate:  {0:N1}%  ({1} hits out of {2} redis lookups)" -f $hitRate, $hitsThisRun, $totalLookups)
        Write-Host ("cache miss rate: {0:N1}%  ({1} misses out of {2} redis lookups)" -f $missRate, $missesThisRun, $totalLookups)
    } else {
        Write-Host "couldn't read redis hit/miss stats (is memurai-cli.exe at $MemuraiCli ?)"
    }
} else {
    Write-Host "GET /{shortCode} - all requests failed, no average to show"
}
Write-Host "===================="

if ($postFailed -gt 0 -or $getFailed -gt 0) {
    Write-Host ""
    Write-Host "note: $postFailed POST(s) and $getFailed GET(s) failed and were excluded from the average."
    Write-Host "if a lot failed, check the app's console for errors (e.g. duplicate short codes after many runs without restarting the app)."
}

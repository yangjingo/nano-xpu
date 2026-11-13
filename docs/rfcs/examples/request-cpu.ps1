# Call the explicitly test-only server; optionally request a clean shutdown.
param(
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [switch]$Shutdown
)
$ErrorActionPreference = 'Stop'
$BaseUrl = $BaseUrl.TrimEnd('/')
Invoke-RestMethod "$BaseUrl/ready"
$body = @{
    model = 'qwen3-0.6b'
    messages = @(@{ role = 'user'; content = '你好' })
    temperature = 0.6
    max_tokens = 16
} | ConvertTo-Json -Depth 5
$result = Invoke-RestMethod "$BaseUrl/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
if ($result.choices[0].message.content -notlike '[[]CPU TEST ENGINE:*') {
    throw 'Expected an explicitly marked CPU test engine response'
}
$result | ConvertTo-Json -Depth 8
if ($Shutdown) {
    Invoke-RestMethod "$BaseUrl/__test__/shutdown" -Method Post
}

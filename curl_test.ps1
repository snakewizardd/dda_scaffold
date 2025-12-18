$ErrorActionPreference = "Stop"

$url = "http://127.0.0.1:1234/v1/chat/completions"
$body = @{
    model = "openai/gpt-oss-20b"
    messages = @(
        @{ role = "user"; content = "Say: HELLO WORLD" }
    )
    temperature = 0.7
    max_tokens = 20
    stream = $false
} | ConvertTo-Json

Write-Host "Sending HTTP POST to $url..." -ForegroundColor Cyan
Write-Host "Please wait (up to 60 seconds)..." -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
    
    Write-Host "`nSUCCESS!" -ForegroundColor Green
    Write-Host "Response Content:" -ForegroundColor White
    Write-Host $response.choices[0].message.content
}
catch {
    Write-Host "`nFAILED!" -ForegroundColor Red
    Write-Host $_.Exception.Message
}

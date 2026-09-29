# Needle-in-a-haystack probe for long-context KV-cache validation.
# Compares recall across KV cache types (e.g. f16 vs q4_0) at a given context length.
#
# Usage:
#   .\needle.ps1 -Base http://127.0.0.1:8082 -TargetTokens 131072 -Depth 0.5
#
# Start one llama-server per KV configuration on its own port first, e.g.:
#   q4_0 : --port 8082 -ctk q4_0 -ctv q4_0
#   f16  : --port 8083                      (control)
#
# Pass = the answer contains the code. Restart the server (or use distinct depths)
# between runs so prefix caching does not short-circuit the prompt.

param(
    [int]$TargetTokens = 131072,
    [double]$Depth = 0.5,
    [string]$Base = 'http://127.0.0.1:8082',
    [string]$Needle = 'The access code for the Zephyr vault is 7741-KESTREL.',
    [string]$Question = 'What is the access code for the Zephyr vault? Reply with only the code.'
)

# ~4 chars/token heuristic, matching the existing bench/depth-test.ps1 approach.
$CharsPerToken = 4
$fillerUnit = 'The archive room held shelves of ledgers, each entry catalogued by hand in fading ink. '
$fillerCount = [math]::Ceiling(($TargetTokens * $CharsPerToken) / $fillerUnit.Length)
$pre = [math]::Floor($fillerCount * $Depth)
$haystack = ($fillerUnit * $pre) + $Needle + "`n" + ($fillerUnit * ($fillerCount - $pre))

$body = @{
    messages = @(@{ role = 'user'; content = "$haystack`n`n$Question" })
    max_tokens = 32
    temperature = 0
    stream = $false
    chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 8 -Compress

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r  = Invoke-RestMethod -Uri "$Base/v1/chat/completions" -Method Post `
        -ContentType 'application/json' -Body $body -TimeoutSec 7200
$sw.Stop()

$answer = $r.choices[0].message.content.Trim()
$found  = $answer -match '7741'

Write-Output ("base={0} depth={1} target={2} actual_prompt={3} wall={4:N0}s" -f `
    $Base, $Depth, $TargetTokens, $r.usage.prompt_tokens, $sw.Elapsed.TotalSeconds)
Write-Output ("RESULT: {0}" -f $(if ($found) { 'FOUND' } else { 'MISSED' }))
Write-Output ("ANSWER: {0}" -f $answer)

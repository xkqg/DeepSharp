# Opens a notebook's socket on a running deepsharp-serve, as a program does — the token in the address, no page named —
# and checks what a page is told: its first frame is the notebook as it stands, holding the cell named, and running that
# cell is answered with the cell as it ran; or, for a name the server does not serve, that it is not found, in words, on
# the socket. Prints what it found; exits 1 when anything is otherwise.
#
#   pwsh tools/serve/socket.ps1 -Address http://127.0.0.1:5000 -Token <token> -Notebook titanic.verso -Cell <id>
#   pwsh tools/serve/socket.ps1 -Address http://127.0.0.1:5000 -Token <token> -Notebook private.txt -NotServed

param(
    [Parameter(Mandatory)] [string]$Address,
    [Parameter(Mandatory)] [string]$Token,
    [Parameter(Mandatory)] [string]$Notebook,
    [string]$Cell,
    [switch]$NotServed
)

$ErrorActionPreference = 'Stop'

function Fail([string]$message) {
    Write-Host $message
    exit 1
}

# One frame, however many pieces it arrives in; nothing when the server closed the socket, or the patience ran out.
function Receive([System.Net.WebSockets.ClientWebSocket]$socket, [int]$seconds) {
    $buffer = [byte[]]::new(64 * 1024)
    $frame = [System.IO.MemoryStream]::new()
    $patience = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($seconds))

    try {
        do {
            $received = $socket.ReceiveAsync([ArraySegment[byte]]::new($buffer), $patience.Token).GetAwaiter().GetResult()

            if ($received.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
                return $null
            }

            $frame.Write($buffer, 0, $received.Count)
        } while (-not $received.EndOfMessage)
    }
    catch [System.OperationCanceledException] {
        return $null
    }

    return [System.Text.Encoding]::UTF8.GetString($frame.ToArray())
}

$socket = [System.Net.WebSockets.ClientWebSocket]::new()
$uri = [Uri]"$($Address -replace '^http', 'ws')/api/notebooks/$Notebook/socket?token=$Token"

try {
    $null = $socket.ConnectAsync($uri, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult()
}
catch {
    Fail "the socket did not open: $($_.Exception.Message)"
}

# The notebook as it stands, first.
$first = Receive $socket 30

if ($null -eq $first) {
    Fail 'the socket told nothing before it closed'
}

$told = $first | ConvertFrom-Json

# A name the server does not serve is refused on the socket, which it then closes.
if ($NotServed) {
    if ($told.type -ne 'refused' -or $told.status -ne 404 -or $null -ne (Receive $socket 10) -or [int]$socket.CloseStatus -ne 4404) {
        Fail "the socket said: $first, and closed with $([int]$socket.CloseStatus)"
    }

    Write-Host 'not served'
    exit 0
}

if ($told.type -ne 'snapshot' -or -not ($told.version.cells | Where-Object { $_.id -eq $Cell })) {
    Fail "the socket began with: $first"
}

# The cell run, answered by the id it was asked with; the changes it made come first.
$ask = [System.Text.Encoding]::UTF8.GetBytes((@{ id = 1; ask = 'run'; cell = $Cell } | ConvertTo-Json -Compress))
$null = $socket.SendAsync([ArraySegment[byte]]::new($ask), [System.Net.WebSockets.WebSocketMessageType]::Text, $true, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult()

do {
    $frame = Receive $socket 120

    if ($null -eq $frame) {
        Fail 'the run was not answered'
    }

    $told = $frame | ConvertFrom-Json
} while ($told.type -ne 'answer' -and $told.type -ne 'refused')

if ($told.type -ne 'answer' -or $told.id -ne 1 -or $told.result.lastStatus -ne 'Success' -or $told.result.outputs.Count -eq 0) {
    Fail "the run was answered with: $frame"
}

$null = $socket.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, 'done', [System.Threading.CancellationToken]::None).GetAwaiter().GetResult()
Write-Host 'the notebook as it stands, then the step run'

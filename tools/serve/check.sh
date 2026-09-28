#!/usr/bin/env bash
# Starts deepsharp-serve from its package, installed the way a person installs it, in a folder of its own, and asks it
# what a browser would. What a person runs is the package, and a package can lack what the build had while every suite
# -- which runs the build -- stays green: a file left out, or the build for one runtime finding a dependency only on the
# other. So each way a person comes to run it is started:
#
#   the command `dotnet tool install` makes     the newest build, on the newest runtime
#   the .NET 8 build on .NET 8                    what an SDK for .NET 8 installs
#   the .NET 8 build on the newest runtime        where that build goes once .NET 8 is removed
#
# and each is asked for its page, with and without the token in the address it names, for a file beside the notebook,
# for the folder's notebooks, and — over the notebook's socket, as its page asks — for a file beside the notebook asked
# for as a notebook, for the notebook, and to run the notebook's step.
#
#   bash tools/serve/check.sh nupkgs     the folder `dotnet pack` wrote the packages to

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
packages=$(cd "${1:?name the folder dotnet pack wrote the packages to}" && pwd)
work=$(mktemp -d)
folder="$work/notebooks"
log="$work/serve.log"
cell=5d1b7a52-3c2e-4f0a-9d6b-2f4e8c9a1b07
server=

# The notebook's socket is spoken to by PowerShell, which every runner carries; a Windows without PowerShell 7 has its
# own, which runs a script only when told to.
powershell=$(command -v pwsh || command -v powershell || true)
[ -n "$powershell" ] || { echo '::error::no PowerShell to speak to the notebook socket with'; exit 1; }

trap 'if [ -n "$server" ]; then kill "$server" 2> /dev/null || true; wait "$server" 2> /dev/null || true; fi; rm -rf "$work"' EXIT

fail() {
  echo "::error::$1"
  echo '--- what the server said:'
  cat "$log" 2> /dev/null || true
  exit 1
}

tool=("$packages"/DeepSharp.Verso.Serve.*.nupkg)
[ ${#tool[@]} -eq 1 ] || fail "$packages holds ${#tool[@]} packages of deepsharp-serve, and this starts one"

# From these packages alone: with the feed as a second source, a version already published would install as well, and
# this would start that one.
dotnet tool install DeepSharp.Verso.Serve --tool-path "$work/tool" --source "$packages"
unzip -q "${tool[0]}" -d "$work/package"

# A notebook with one step reading the sample's passengers, the file it reads, and beside them a file that is not a
# notebook and is not to be served.
mkdir "$folder"
cp "$root/Samples/data/titanic.csv" "$folder/"
echo 'not a notebook' > "$folder/private.txt"
cat > "$folder/titanic.verso" << NOTEBOOK
{"formatVersion": "1.0", "cells": [{"id": "$cell", "type": "deepsharp.step", "source": "{\"step\": \"read.csv\", \"path\": \"titanic.csv\"}"}]}
NOTEBOOK
cd "$folder"

# The first argument says what is started; the rest is the command that starts it, in the folder, as a person would.
check() {
  local what=$1 address= base token status notebook stream ran
  shift
  echo "==> $what"

  "$@" --no-browser > "$log" 2>&1 &
  server=$!
  for _ in $(seq 1 120); do
    address=$(grep -o 'http://[^ ]*/?token=[0-9a-f]*' "$log" || true)
    if [ -n "$address" ] || ! kill -0 "$server" 2> /dev/null; then break; fi
    sleep 1
  done
  [ -n "$address" ] || fail "$what named no address to open"
  base=${address%/?token=*}
  token=${address#*token=}

  status=$(curl -s -o "$work/page.html" -w '%{http_code}' "$address")
  [ "$status" = 200 ] && grep -q 'id="notebook-name"' "$work/page.html" || fail "$what answered $status for the address it named"
  status=$(curl -s -o /dev/null -w '%{http_code}' "$base/")
  [ "$status" = 401 ] || fail "$what answered $status for its page without the token"
  status=$(curl -s -o /dev/null -w '%{http_code}' "$base/private.txt?token=$token")
  [ "$status" = 404 ] || fail "$what answered $status for the file beside the notebook"
  [ "$(curl -s "$base/api/notebooks?token=$token")" = '["titanic.verso"]' ] || fail "$what listed the folder otherwise"

  # The notebook is asked over its socket, as its page asks, by a client every runner carries: the server writes no log,
  # since a running cell takes over the console it would write to, so what the socket was told is what is said.
  said=$("$powershell" -NoProfile -ExecutionPolicy Bypass -File "$root/tools/serve/socket.ps1" -Address "$base" -Token "$token" -Notebook private.txt -NotServed) \
    || fail "$what, asked for the file beside the notebook as a notebook: $said"
  said=$("$powershell" -NoProfile -ExecutionPolicy Bypass -File "$root/tools/serve/socket.ps1" -Address "$base" -Token "$token" -Notebook titanic.verso -Cell "$cell") \
    || fail "$what, asked for the notebook over its socket: $said"

  kill "$server"
  wait "$server" 2> /dev/null || true
  server=
  echo "    its page, refused without the token, nothing beside the notebook, the list, over its socket the notebook and a run"
}

check 'the installed command' "$work/tool/deepsharp-serve"
check 'the .NET 8 build on .NET 8' dotnet "$work/package/tools/net8.0/any/DeepSharp.Verso.Serve.dll"
check 'the .NET 8 build on the newest runtime' dotnet exec --roll-forward LatestMajor "$work/package/tools/net8.0/any/DeepSharp.Verso.Serve.dll"

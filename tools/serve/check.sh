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
# for the folder's notebooks, for the notebook and a stream of it, and to run the notebook's step.
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
  status=$(curl -s -o /dev/null -w '%{http_code}' "$base/api/notebooks/private.txt?token=$token")
  [ "$status" = 404 ] || fail "$what answered $status for the file beside the notebook, asked for as a notebook"

  [ "$(curl -s "$base/api/notebooks?token=$token")" = '["titanic.verso"]' ] || fail "$what listed the folder otherwise"
  notebook=$(curl -s -D - "$base/api/notebooks/titanic.verso?token=$token")
  grep -q "\"id\":\"$cell\"" <<< "$notebook" || fail "$what served the notebook as $notebook"

  # A stream does not end: whatever arrived in the seconds given is what it began with, and the notebook is open by now.
  # The status line is kept with each answer read, so a refusal says what it was: the server writes no log, since a
  # running cell takes over the console it would write to.
  stream=$(curl -s -N -D - --max-time 5 "$base/api/notebooks/titanic.verso/updates?token=$token" || true)
  grep -q '^event: snapshot' <<< "$stream" && grep -q "$cell" <<< "$stream" || fail "$what began its stream with: $stream"

  ran=$(curl -s -D - --max-time 120 -X POST "$base/api/notebooks/titanic.verso/cells/$cell/run?token=$token")
  grep -q '"lastStatus":"Success"' <<< "$ran" && grep -q '"outputs":\[{' <<< "$ran" || fail "$what ran the step to: $ran"

  kill "$server"
  wait "$server" 2> /dev/null || true
  server=
  echo "    its page, refused without the token, nothing beside the notebook, the list, the notebook, its stream, a run"
}

check 'the installed command' "$work/tool/deepsharp-serve"
check 'the .NET 8 build on .NET 8' dotnet "$work/package/tools/net8.0/any/DeepSharp.Verso.Serve.dll"
check 'the .NET 8 build on the newest runtime' dotnet exec --roll-forward LatestMajor "$work/package/tools/net8.0/any/DeepSharp.Verso.Serve.dll"

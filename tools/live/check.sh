#!/usr/bin/env bash
# Builds an application against the Binance landing's package from the packages just made, and nothing of DeepSharp's
# source, and starts it twice: once bringing that package, where it lands a window of candles from a stand-in venue on this
# machine -- the stand-in the tests use, never the real one -- and writes the pipeline that reads the landing; and once
# bringing only the pipeline library, where that pipeline is read back with the library's own catalog and run over the
# landing, and the application carries nothing of the package and nothing of Polly at all. What a person runs is the
# package, and a package can lack what the build had while every suite -- which runs the build -- stays green.
#
#   bash tools/live/check.sh nupkgs     the folder `dotnet pack` wrote the packages to

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
packages=$(cd "${1:?name the folder dotnet pack wrote the packages to}" && pwd)
host="$root/tools/live/LiveCheckHost"
venue="$root/Tst/DeepSharp/Pipelines/FakeBinanceVenue.cs"
socket="$root/Tst/DeepSharp/Pipelines/StandIn.cs"
work=$(mktemp -d)

trap 'rm -rf "$work"' EXIT

fail() {
  echo "::error::$1"
  exit 1
}

# A path as the programs started here read it: Windows' own, under the bash that comes with Git.
native() {
  if command -v cygpath > /dev/null; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

for id in DeepSharp.Pipelines DeepSharp.Pipelines.Binance; do
  found=("$packages"/$id.[0-9]*.nupkg)
  [ ${#found[@]} -eq 1 ] && [ -f "${found[0]}" ] || fail "$packages holds ${#found[@]} $id packages, and this checks one"
done

[ -f "$venue" ] || fail "the stand-in venue the tests use is not at $venue"
[ -f "$socket" ] || fail "the socket the tests put that venue on is not at $socket"

mkdir -p "$work/landed"

# The first argument says what is checked, the second which packages are brought, the third what the application is told to
# do: "lands", which lands a window and writes the pipeline, or "reads", which reads that pipeline back and runs it over the
# landing without the landing's package anywhere.
check() {
  local what=$1 landing=$2 does=$3 run="$work/$2" config temp

  echo "==> $what"

  mkdir -p "$run/temp"
  # A copy of its own for each build: one project folder built twice would hold one obj, and the second build would meet
  # the first one's assembly attributes.
  cp -r "$host" "$run/host"
  rm -rf "$run/host/obj" "$run/host/bin"
  cat > "$run/nuget.config" << CONFIG
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="packages" value="$(native "$packages")" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
CONFIG

  config=$(native "$run/nuget.config")
  temp=$(native "$run/temp")

  TMP="$temp" TEMP="$temp" TMPDIR="$temp" NUGET_HTTP_CACHE_PATH="$(native "$run/temp/http")" \
    NUGET_PACKAGES="$(native "$run/temp/packages")" \
    dotnet restore "$(native "$run/host/LiveCheckHost.csproj")" -p:RestoreConfigFile="$config" -p:Landing="$landing" -p:VenueSource="$(native "$venue")" -p:StandInSource="$(native "$socket")" \
    || fail "$what: could not restore the application against the packages just made"

  TMP="$temp" TEMP="$temp" TMPDIR="$temp" NUGET_HTTP_CACHE_PATH="$(native "$run/temp/http")" \
    NUGET_PACKAGES="$(native "$run/temp/packages")" \
    dotnet run --project "$(native "$run/host/LiveCheckHost.csproj")" --configuration Release --no-restore -p:Landing="$landing" -p:VenueSource="$(native "$venue")" -p:StandInSource="$(native "$socket")" -- \
      "$does" "$(native "$work/landed")" \
    || fail "$what: the packages, referenced as an application references them, did not do what the lines above say"
}

check 'with the package, a window landed from a stand-in venue on this machine, and the pipeline that reads it written' binance lands
check 'with the pipeline library alone, that pipeline read back and run over the landing, and nothing of Binance or Polly anywhere' none reads

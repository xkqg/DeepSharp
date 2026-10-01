#!/usr/bin/env bash
# Builds an application against DeepSharp.Backends.TorchSharp from the package just made, and nothing of DeepSharp's
# source, and starts it twice: once bringing no libtorch, where TorchBackend.OnCpu() is refused, naming every package
# that brings one, and once bringing the processor's, where it fits a step of the networks sample's Titanic pipeline.
# What a person runs is the package, and a package can lack what the build had while every suite -- which runs the
# build -- stays green, so this is what an application that brought only the package finds.
#
#   bash tools/torch/check.sh nupkgs     the folder `dotnet pack` wrote the packages to

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
packages=$(cd "${1:?name the folder dotnet pack wrote the packages to}" && pwd)
host="$root/tools/torch/TorchCheckHost/TorchCheckHost.csproj"
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

engine=("$packages"/DeepSharp.Backends.TorchSharp.*.nupkg)
[ ${#engine[@]} -eq 1 ] && [ -f "${engine[0]}" ] || fail "$packages holds ${#engine[@]} engine packages, and this checks one"

# The first argument says what is checked, the second whether libtorch is brought, the third what the host is told to
# expect: "refused", with nothing else to run, or "titanic", with the passenger list beside it.
check() {
  local what=$1 libtorch=$2 expect=$3 run="$work/$2" config temp

  echo "==> $what"

  mkdir -p "$run/temp"
  cp "$root/Samples/data/titanic.csv" "$run/"
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
    dotnet restore "$(native "$host")" -p:RestoreConfigFile="$config" -p:Libtorch="$libtorch" \
    || fail "$what: could not restore the application against the package just made"

  TMP="$temp" TEMP="$temp" TMPDIR="$temp" NUGET_HTTP_CACHE_PATH="$(native "$run/temp/http")" \
    NUGET_PACKAGES="$(native "$run/temp/packages")" \
    dotnet run --project "$(native "$host")" --configuration Release --no-restore -p:Libtorch="$libtorch" -- \
      "$expect" $([ "$expect" = titanic ] && echo "$(native "$run/titanic.csv")") \
    || fail "$what: the engine package, referenced as an application references it, did not do what the lines above say"
}

check 'without libtorch, refused by name' none refused
check "with the processor's libtorch, a Titanic step" cpu titanic

#!/usr/bin/env bash
# Builds an application against the ML.NET learner's packages from the packages just made, and nothing of DeepSharp's
# source, and starts it twice: once bringing the package that trains, where it fits a tree of the Titanic pipeline and
# writes the model beside its pipeline as one file, and once bringing only the package that declares and reads, where
# that file is read back and the application carries nothing of ML.NET at all. What a person runs is the package, and a
# package can lack what the build had while every suite -- which runs the build -- stays green.
#
#   bash tools/ml/check.sh nupkgs     the folder `dotnet pack` wrote the packages to

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
packages=$(cd "${1:?name the folder dotnet pack wrote the packages to}" && pwd)
host="$root/tools/ml/MLCheckHost"
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

for id in DeepSharp.Learners.ML DeepSharp.Learners.MLNet; do
  found=("$packages"/$id.[0-9]*.nupkg)
  [ ${#found[@]} -eq 1 ] && [ -f "${found[0]}" ] || fail "$packages holds ${#found[@]} $id packages, and this checks one"
done

cp "$root/Samples/data/titanic.csv" "$work/"

# The first argument says what is checked, the second which packages are brought, the third what the application is
# told to do: "trains", which writes the model file, or "reads", which reads it back without ML.NET anywhere.
check() {
  local what=$1 trainer=$2 does=$3 run="$work/$2" config temp

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
    dotnet restore "$(native "$run/host/MLCheckHost.csproj")" -p:RestoreConfigFile="$config" -p:Trainer="$trainer" \
    || fail "$what: could not restore the application against the packages just made"

  TMP="$temp" TEMP="$temp" TMPDIR="$temp" NUGET_HTTP_CACHE_PATH="$(native "$run/temp/http")" \
    NUGET_PACKAGES="$(native "$run/temp/packages")" \
    dotnet run --project "$(native "$run/host/MLCheckHost.csproj")" --configuration Release --no-restore -p:Trainer="$trainer" -- \
      "$does" "$(native "$work/titanic.csv")" "$(native "$work/titanic.mlmodel.json")" \
    || fail "$what: the packages, referenced as an application references them, did not do what the lines above say"
}

check 'with the trainer, a tree of the Titanic pipeline written as one file' mlnet trains
check 'with the declaration alone, the same file read back and no ML.NET anywhere' none reads

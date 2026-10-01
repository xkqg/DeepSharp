#!/usr/bin/env bash
# Installs the notebook package the way Verso installs it and runs a notebook in it. Verso installs a package into a folder
# of its own and loads it apart from everything else, and every suite runs the build, where a dependency the package never
# declared is found all the same: so the package just made is installed by Verso's own installer, from these packages and
# nuget.org, with a download cache of its own, and loaded by Verso's own loader, in a host of Verso's engine that holds
# nothing of DeepSharp. It is started as each of Verso's hosts starts:
#
#   the host's .NET 8 build on .NET 8               the browser editor `verso serve` opens, for as long as .NET 8 is there
#   the host's .NET 8 build on the newest runtime   Verso's VS Code extension, which takes the newest .NET installed
#
# and each time checks that Verso lays out the files tools/verso/installed.<runtime>.txt names — the list the notebook's
# PackageTests hold to what the build resolves — and that its loader loads every one; opens the sample notebook, whose
# report block shows the data, whose C# cell trains a network on DeepSharp's packages named by their NuGet ids and draws
# its loss, after which the report block draws the measures handed back; and checks that everything the notebook refers
# to is found in the folder Verso installed it to, and every method of DeepSharp's compiles against it.
#
#   bash tools/verso/check.sh nupkgs     the folder `dotnet pack` wrote the packages to

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
packages=$(cd "${1:?name the folder dotnet pack wrote the packages to}" && pwd)
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

notebook=("$packages"/DeepSharp.Verso.Notebooks.*.nupkg)
[ ${#notebook[@]} -eq 1 ] && [ -f "${notebook[0]}" ] || fail "$packages holds ${#notebook[@]} notebook packages, and this installs one"

dotnet build "$root/tools/verso/VersoHost/VersoHost.csproj" --configuration Release --nologo --verbosity quiet
host="$root/tools/verso/VersoHost/bin/Release/net8.0/VersoHost.dll"

# The first argument says what is started, the second names its folder; the rest is the command that starts the host.
check() {
  local what=$1 run="$work/$2"
  shift 2
  echo "==> $what"

  # The sample notebook, the passenger list beside it, and the sources its C# cell's packages come from: the packages just
  # made, ahead of nuget.org, and no other a machine names.
  mkdir -p "$run/temp" "$run/notebook/data"
  cp "$root/Samples/titanic.verso" "$run/notebook/"
  cp "$root/Samples/data/titanic.csv" "$run/notebook/data/"
  cat > "$run/notebook/nuget.config" << CONFIG
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="packages" value="$(native "$packages")" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
CONFIG

  # A temporary folder of its own, where Verso keeps what it downloads and the host keeps its install, and NuGet caches of
  # its own: nothing a run before this one fetched is found again, and nothing this run extracts is left in the machine's
  # cache, where a later check of the same version would find this build rather than the one it was handed.
  local temp
  temp=$(native "$run/temp")
  (cd "$run/notebook" && TMP="$temp" TEMP="$temp" TMPDIR="$temp" NUGET_HTTP_CACHE_PATH="$(native "$run/temp/http")" \
    NUGET_PACKAGES="$(native "$run/temp/packages")" "$@" "$(native "$host")" "$(native "${notebook[0]}")" titanic.verso "$(native "$root/tools/verso")") \
    || fail "$what: the notebook package, installed as Verso installs it, did not do what the lines above say"
}

check 'the .NET 8 build on .NET 8' net8 dotnet
check 'the .NET 8 build on the newest runtime' newest dotnet exec --roll-forward LatestMajor

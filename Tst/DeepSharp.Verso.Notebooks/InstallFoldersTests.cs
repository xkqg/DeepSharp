// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A package loaded into a context of its own keeps its files locked until its process ends, so a suite that lays a package out
/// beside itself cannot lay it out again while another process of the same suite holds it. Each process therefore has a folder
/// of its own, and the folders of processes that have ended are taken away by the next one.
/// </summary>
public sealed class InstallFoldersTests : IDisposable
{
    // A process id no process has: the largest ids a system hands out are far below it.
    private const int NoProcess = int.MaxValue - 1;

    private readonly string _root = Directory.CreateTempSubdirectory("deepsharp-installs-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void TwoProcesses_LayTheirInstallsOutInFoldersOfTheirOwn_BesideTheSuite()
    {
        var one = VersoInstall.InstallsOf(4001);
        var other = VersoInstall.InstallsOf(4002);

        Assert.NotEqual(one, other);
        Assert.Equal(Path.Join(AppContext.BaseDirectory, "installed"), Path.GetDirectoryName(one));
        Assert.Equal("4001", Path.GetFileName(one));
    }

    [Fact]
    public void ThisProcess_HasTheFolderItsId_Names()
    {
        Assert.Equal(VersoInstall.InstallsOf(Environment.ProcessId), VersoInstall.Installs);
        Assert.True(Directory.Exists(VersoInstall.Installs));
    }

    [Fact]
    public void TheFolderOfAProcessThatHasEnded_IsTakenAway_AndTheFolderOfOneThatRuns_Stays()
    {
        var ended = Directory.CreateDirectory(Path.Join(_root, NoProcess.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        var running = Directory.CreateDirectory(Path.Join(_root, Environment.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture)));

        File.WriteAllText(Path.Join(ended.FullName, "DeepSharp.Verso.Notebooks.dll"), "left by a process that ended");
        File.WriteAllText(Path.Join(running.FullName, "DeepSharp.Verso.Notebooks.dll"), "held by a process that runs");

        VersoInstall.ClearAfterTheDead(_root);

        Assert.False(Directory.Exists(ended.FullName));
        Assert.True(File.Exists(Path.Join(running.FullName, "DeepSharp.Verso.Notebooks.dll")));
    }

    [Fact]
    public void WhatAnEarlierLayoutLeft_UnderNamesThatAreNoProcess_IsTakenAway()
    {
        foreach (var name in new[] { "package", "top", "managed", "listed" })
        {
            Directory.CreateDirectory(Path.Join(_root, name));
            File.WriteAllText(Path.Join(_root, name, "DeepSharp.Verso.Notebooks.dll"), "the old layout");
        }

        VersoInstall.ClearAfterTheDead(_root);

        Assert.Empty(Directory.GetDirectories(_root));
    }

    [Fact]
    public void AFolderThatIsNotThere_IsNothingToClear()
    {
        VersoInstall.ClearAfterTheDead(Path.Join(_root, "nowhere"));

        Assert.False(Directory.Exists(Path.Join(_root, "nowhere")));
    }
}

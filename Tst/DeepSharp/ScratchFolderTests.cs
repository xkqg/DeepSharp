// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests;

/// <summary>
/// A folder one test writes into is looked at, and taken away, by programs that are not the test: a scanner that opens each
/// new file leaves a copy of its own beside it for a moment and holds it open. The folder must not fail the test for that,
/// and must still fail it for what stays.
/// </summary>
public sealed class ScratchFolderTests
{
    private static readonly TimeSpan Moment = TimeSpan.FromMilliseconds(150);

    [Fact]
    public void ACopyAScannerLeavesBesideAFile_AndTakesAwayAgain_IsNotWhatStands()
    {
        using var folder = new ScratchFolder("deepsharp-scratch-");

        File.WriteAllText(folder.At("rows.csv"), "x\n");

        using var scanner = new Scanner(folder, "rows.csv");

        Assert.Contains("ROWS.CSV.tmp", folder.Standing());

        scanner.LeavesAfter(Moment);

        Assert.Equal(["rows.csv"], folder.Standing(["rows.csv"], within: TimeSpan.FromSeconds(10)));
    }

    [Fact]
    public void AnythingThatStays_IsWhatStands_WhenTheTimeIsUp()
    {
        using var folder = new ScratchFolder("deepsharp-scratch-");

        File.WriteAllText(folder.At("rows.csv"), "x\n");
        File.WriteAllText(folder.At(".rows.csv.left-behind.tmp"), "x\n");

        Assert.Equal([".rows.csv.left-behind.tmp", "rows.csv"], folder.Standing(["rows.csv"], within: Moment));
    }

    [Fact]
    public void AFileAScannerHoldsForAMoment_DoesNotStopTheFolderFromBeingRemoved()
    {
        var folder = new ScratchFolder("deepsharp-scratch-");

        File.WriteAllText(folder.At("rows.csv"), "x\n");

        using var scanner = new Scanner(folder, "rows.csv");

        scanner.LeavesAfter(Moment);
        folder.Dispose(TimeSpan.FromSeconds(10));

        Assert.False(Directory.Exists(folder.Path));
    }

    [Fact]
    public void AFolderAlreadyGone_IsRemovedWithoutAFuss()
    {
        var folder = new ScratchFolder("deepsharp-scratch-");

        Directory.Delete(folder.Path, recursive: true);

        folder.Dispose();

        Assert.False(Directory.Exists(folder.Path));
    }

    [Fact]
    public void AFileThatStaysHeld_StopsTheFolderFromBeingRemoved_AndSaysSo()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Only Windows refuses to remove a file that is held open; elsewhere the removal succeeds, and there is nothing held to wait for.");

        var folder = new ScratchFolder("deepsharp-scratch-");

        using (new FileStream(folder.At("held.tmp"), FileMode.CreateNew, FileAccess.Write, FileShare.None))
        {
            Assert.Throws<IOException>(() => folder.Dispose(patience: TimeSpan.FromMilliseconds(200)));
        }

        folder.Dispose();
    }

    // What a scanner does to a file it has just seen: a copy in capitals beside it, held, and given up when it is done. The copy
    // is there when the constructor returns, and it goes only when told to, on a thread of its own, so that neither the clock
    // nor a busy thread pool decides whether a test finds it.
    private sealed class Scanner : IDisposable
    {
        private readonly string _copy;
        private readonly FileStream _held;
        private Thread? _leaving;

        public Scanner(ScratchFolder folder, string name)
        {
            _copy = folder.At(name.ToUpperInvariant() + ".tmp");
            _held = new FileStream(_copy, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        }

        public void LeavesAfter(TimeSpan delay)
        {
            _leaving = new Thread(() =>
            {
                Thread.Sleep(delay);
                _held.Dispose();

                try
                {
                    File.Delete(_copy);
                }
                catch (Exception left) when (left is IOException or UnauthorizedAccessException)
                {
                    // The folder was removed first, with the copy in it; or another scanner has the copy open, and the folder takes it.
                }
            })
            { IsBackground = true };
            _leaving.Start();
        }

        public void Dispose()
        {
            _leaving?.Join();
            _held.Dispose();
        }
    }
}

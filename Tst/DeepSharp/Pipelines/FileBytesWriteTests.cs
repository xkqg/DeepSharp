// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A file written whole: its bytes go to a name of their own beside it and are moved into place in one step, so whoever
/// looks finds nothing or the whole of it, and whatever stops a write leaves nothing of its own behind. A file that is
/// written once is never replaced: the same bytes already standing is the same file, other bytes are refused, and of many
/// that race for one place exactly one stays.
/// </summary>
public sealed class FileBytesWriteTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-whole-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private string[] Standing() => [.. Directory.GetFileSystemEntries(_folder).Select(entry => Path.GetFileName(entry)).Order(StringComparer.Ordinal)];

    private static byte[] Text(string text) => Encoding.UTF8.GetBytes(text);

    // Moving a file onto a folder is refused as the system says it: an IOException where the file system answers "is a
    // directory", an UnauthorizedAccessException where it answers "access denied".
    private static void TheMoveWasRefused(Exception? failure) =>
        Assert.True(failure is IOException or UnauthorizedAccessException, $"The move was not refused as a move is: {failure?.ToString() ?? "nothing was thrown"}");

    [Fact]
    public void WritingWhole_PutsTheBytesWhereTheyAreMeant_AndNothingElseStandsBeside()
    {
        Text("one\n").WriteWhole(At("one.txt"));

        Assert.Equal(Text("one\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public void WritingWhole_ReplacesWhatStood()
    {
        File.WriteAllText(At("one.txt"), "old\n");

        Text("new\n").WriteWhole(At("one.txt"));

        Assert.Equal(Text("new\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public async Task WritingWholeAsync_ReplacesWhatStood_AndNothingElseStandsBeside()
    {
        File.WriteAllText(At("one.txt"), "old\n");

        await Text("new\n").WriteWholeAsync(At("one.txt"), TestContext.Current.CancellationToken);

        Assert.Equal(Text("new\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public void WritingWhole_WhereTheMoveIsRefused_LeavesNoNameOfItsOwnBehind()
    {
        Directory.CreateDirectory(At("taken"));

        TheMoveWasRefused(Record.Exception(() => Text("new\n").WriteWhole(At("taken"))));
        Assert.Equal(["taken"], Standing());
    }

    [Fact]
    public async Task WritingWholeAsync_WhereTheMoveIsRefused_LeavesNoNameOfItsOwnBehind()
    {
        Directory.CreateDirectory(At("taken"));

        TheMoveWasRefused(await Record.ExceptionAsync(() => Text("new\n").WriteWholeAsync(At("taken"), TestContext.Current.CancellationToken)));
        Assert.Equal(["taken"], Standing());
    }

    [Fact]
    public async Task WritingWholeAsync_AfterItWasCancelled_WritesNothing()
    {
        using var cancelled = new CancellationTokenSource();

        await cancelled.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Text("new\n").WriteWholeAsync(At("one.txt"), cancelled.Token));
        Assert.Empty(Standing());
    }

    [Fact]
    public async Task WritingWholeOnce_WhereNothingStands_Writes()
    {
        await Text("one\n").WriteWholeOnceAsync(At("one.txt"), TestContext.Current.CancellationToken);

        Assert.Equal(Text("one\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public async Task WritingWholeOnce_WhereTheSameBytesStand_IsTheSameFile_AndNothingStandsBeside()
    {
        File.WriteAllBytes(At("one.txt"), Text("one\n"));

        await Text("one\n").WriteWholeOnceAsync(At("one.txt"), TestContext.Current.CancellationToken);

        Assert.Equal(Text("one\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public async Task WritingWholeOnce_WhereOtherBytesStand_IsRefused_AndTheFirstIsKept()
    {
        File.WriteAllBytes(At("one.txt"), Text("first\n"));

        var refused = await Assert.ThrowsAsync<IOException>(() => Text("second\n").WriteWholeOnceAsync(At("one.txt"), TestContext.Current.CancellationToken));

        Assert.Contains("one.txt", refused.Message, StringComparison.Ordinal);
        Assert.Equal(Text("first\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public async Task WritingWholeOnce_AfterItWasCancelled_WritesNothing()
    {
        using var cancelled = new CancellationTokenSource();

        await cancelled.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Text("one\n").WriteWholeOnceAsync(At("one.txt"), cancelled.Token));
        Assert.Empty(Standing());
    }

    [Fact]
    public async Task WritingWholeOnce_WhereTheFolderIsMissing_IsNotSwallowed()
    {
        await Assert.ThrowsAnyAsync<IOException>(() => Text("one\n").WriteWholeOnceAsync(At(Path.Join("nowhere", "one.txt")), TestContext.Current.CancellationToken));
        Assert.Empty(Standing());
    }

    [Fact]
    public async Task ManyWritingWholeOnce_OfTheSameBytes_AllSucceed_AndOneFileStands()
    {
        var bytes = Text("the same answer\n");

        await Task.WhenAll(Enumerable.Range(0, 16).Select(_ => Task.Run(() => bytes.WriteWholeOnceAsync(At("one.txt"), TestContext.Current.CancellationToken), TestContext.Current.CancellationToken)));

        Assert.Equal(bytes, File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public async Task ManyWritingWholeOnce_OfOtherBytes_LetExactlyOneStand_AndRefuseTheRest()
    {
        var attempts = Enumerable.Range(0, 16).Select(at => Text($"answer {at}\n")).ToArray();

        var outcomes = await Task.WhenAll(attempts.Select(bytes => Task.Run(async () =>
        {
            try
            {
                await bytes.WriteWholeOnceAsync(At("one.txt"), TestContext.Current.CancellationToken);

                return true;
            }
            catch (IOException)
            {
                return false;
            }
        }, TestContext.Current.CancellationToken)));

        Assert.Single(outcomes, won => won);
        Assert.Contains(File.ReadAllBytes(At("one.txt")), attempts);
        Assert.Equal(["one.txt"], Standing());
    }

    // A move as the system makes it for a while: refused so many times, as it refuses one it will allow a moment later, and then made.
    private sealed class Refusing(int times, Exception refusal, Action? meanwhile = null)
    {
        public int Tries { get; private set; }

        public void Move(Placing placing)
        {
            Tries++;
            meanwhile?.Invoke();

            if (Tries <= times)
            {
                throw refusal;
            }

            placing.Done();
        }
    }

    // A name of its own with bytes in it, as a write leaves it just before the move.
    private string Written(string bytes)
    {
        var own = At($".own.{Guid.NewGuid():N}.tmp");

        File.WriteAllBytes(own, Text(bytes));

        return own;
    }

    private static readonly TimeSpan Plenty = TimeSpan.FromSeconds(30);

    [Fact]
    public void AMoveRefusedAsAccessDenied_IsMadeAgain_UntilItGoesThrough()
    {
        var refusing = new Refusing(2, new UnauthorizedAccessException("Access to the path is denied."));

        WholeFile.MoveInto(new Placing(Written("new\n"), At("one.txt"), Replaces: true), Plenty, refusing.Move);

        Assert.Equal(3, refusing.Tries);
        Assert.Equal(Text("new\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Theory]
    [InlineData(0x80070020)]
    [InlineData(0x80070021)]
    public void AMoveRefusedAsASharingOrLockViolation_IsMadeAgain(uint code)
    {
        var refusing = new Refusing(1, new IOException("being used by another process", unchecked((int)code)));

        WholeFile.MoveInto(new Placing(Written("new\n"), At("one.txt"), Replaces: true), Plenty, refusing.Move);

        Assert.Equal(2, refusing.Tries);
        Assert.Equal(Text("new\n"), File.ReadAllBytes(At("one.txt")));
    }

    [Theory]
    [InlineData(0x80070050)]
    [InlineData(0x80070003)]
    [InlineData(0x80004005)]
    public void AMoveRefusedForAnotherReason_IsNotMadeAgain(uint code)
    {
        var refusing = new Refusing(5, new IOException("another reason", unchecked((int)code)));

        Assert.Throws<IOException>(() => WholeFile.MoveInto(new Placing(Written("new\n"), At("one.txt"), Replaces: false), Plenty, refusing.Move));
        Assert.Equal(1, refusing.Tries);
    }

    [Fact]
    public void AMoveRefusedWhereAFolderStands_IsNotMadeAgain()
    {
        Directory.CreateDirectory(At("taken"));

        var refusing = new Refusing(5, new UnauthorizedAccessException("Access to the path is denied."));

        Assert.Throws<UnauthorizedAccessException>(() => WholeFile.MoveInto(new Placing(Written("new\n"), At("taken"), Replaces: true), Plenty, refusing.Move));
        Assert.Equal(1, refusing.Tries);
    }

    [Fact]
    public void AMoveRefusedWhereAReadOnlyFileStands_IsNotMadeAgain()
    {
        File.WriteAllText(At("one.txt"), "old\n");
        File.SetAttributes(At("one.txt"), FileAttributes.ReadOnly);

        try
        {
            var refusing = new Refusing(5, new UnauthorizedAccessException("Access to the path is denied."));

            Assert.Throws<UnauthorizedAccessException>(() => WholeFile.MoveInto(new Placing(Written("new\n"), At("one.txt"), Replaces: true), Plenty, refusing.Move));
            Assert.Equal(1, refusing.Tries);
        }
        finally
        {
            File.SetAttributes(At("one.txt"), FileAttributes.Normal);
        }
    }

    [Fact]
    public void AMoveRefusedForGood_IsTheAnswerOnceThePatienceIsSpent_AsTheSameRefusal()
    {
        var refusal = new UnauthorizedAccessException("Access to the path is denied.");
        var refusing = new Refusing(int.MaxValue, refusal);

        var thrown = Assert.Throws<UnauthorizedAccessException>(() => WholeFile.MoveInto(new Placing(Written("new\n"), At("one.txt"), Replaces: true), TimeSpan.FromMilliseconds(300), refusing.Move));

        Assert.Same(refusal, thrown);
        Assert.True(refusing.Tries > 1, $"it was made {refusing.Tries} time");
    }

    [Fact]
    public async Task AnAsyncMoveRefusedAsAccessDenied_IsMadeAgain_UntilItGoesThrough()
    {
        var refusing = new Refusing(2, new UnauthorizedAccessException("Access to the path is denied."));

        await WholeFile.MoveIntoAsync(new Placing(Written("new\n"), At("one.txt"), Replaces: true), Plenty, refusing.Move, TestContext.Current.CancellationToken);

        Assert.Equal(3, refusing.Tries);
        Assert.Equal(Text("new\n"), File.ReadAllBytes(At("one.txt")));
    }

    [Fact]
    public async Task AnAsyncMoveRefusedForGood_IsTheAnswerOnceThePatienceIsSpent()
    {
        var refusing = new Refusing(int.MaxValue, new UnauthorizedAccessException("Access to the path is denied."));

        await Assert.ThrowsAsync<UnauthorizedAccessException>(
            () => WholeFile.MoveIntoAsync(new Placing(Written("new\n"), At("one.txt"), Replaces: true), TimeSpan.FromMilliseconds(300), refusing.Move, TestContext.Current.CancellationToken));
        Assert.True(refusing.Tries > 1, $"it was made {refusing.Tries} time");
    }

    [Fact]
    public async Task AnAsyncMove_WaitingBetweenTries_EndsWhenItIsCancelled()
    {
        using var cancelled = new CancellationTokenSource();
        var refusing = new Refusing(int.MaxValue, new UnauthorizedAccessException("Access to the path is denied."), meanwhile: cancelled.Cancel);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => WholeFile.MoveIntoAsync(new Placing(Written("new\n"), At("one.txt"), Replaces: true), Plenty, refusing.Move, cancelled.Token));
        Assert.Equal(1, refusing.Tries);
    }

    // Windows refuses a file to be replaced while another handle to it does not share deletion; whatever holds it for a moment
    // lets go, and the write goes through. Where a file does not need that, the refusal is not there to be had.
    [Fact]
    public async Task WritingWhole_WhereSomethingHoldsTheFileForAMoment_WaitsForItAndWrites()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Only Windows refuses a file to be replaced while it is open without sharing deletion.");
        File.WriteAllText(At("one.txt"), "old\n");

        var holding = new FileStream(At("one.txt"), FileMode.Open, FileAccess.Read, FileShare.Read);
        var writing = Task.Run(() => Text("new\n").WriteWhole(At("one.txt")), TestContext.Current.CancellationToken);

        await Task.Delay(150, TestContext.Current.CancellationToken);
        await holding.DisposeAsync();
        await writing.WaitAsync(Plenty, TestContext.Current.CancellationToken);

        Assert.Equal(Text("new\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public async Task WritingWholeAsync_WhereSomethingHoldsTheFileForAMoment_WaitsForItAndWrites()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Only Windows refuses a file to be replaced while it is open without sharing deletion.");
        File.WriteAllText(At("one.txt"), "old\n");

        var holding = new FileStream(At("one.txt"), FileMode.Open, FileAccess.Read, FileShare.Read);
        var writing = Text("new\n").WriteWholeAsync(At("one.txt"), TestContext.Current.CancellationToken);

        await Task.Delay(150, TestContext.Current.CancellationToken);
        await holding.DisposeAsync();
        await writing.WaitAsync(Plenty, TestContext.Current.CancellationToken);

        Assert.Equal(Text("new\n"), File.ReadAllBytes(At("one.txt")));
        Assert.Equal(["one.txt"], Standing());
    }

    [Fact]
    public void NoBytes_AreWrittenWhole()
    {
        Assert.Throws<ArgumentNullException>(() => ((byte[])null!).WriteWhole(At("one.txt")));
        Assert.Throws<ArgumentNullException>(() => Text("one\n").WriteWhole(null!));
    }

    [Fact]
    public async Task NoBytes_AreWrittenWholeOnce_AndNoPlaceIsNamedEither()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() => ((byte[])null!).WriteWholeOnceAsync(At("one.txt"), TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ArgumentNullException>(() => Text("one\n").WriteWholeOnceAsync(null!, TestContext.Current.CancellationToken));
    }
}

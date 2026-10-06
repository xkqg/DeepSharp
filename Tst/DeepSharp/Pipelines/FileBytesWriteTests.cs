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

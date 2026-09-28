// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Channels;
using Microsoft.Playwright;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// What a person types reaches the notebook as it is typed, each keystroke at once, as Verso's editor sends each change —
// even when the page is reloaded at once — and waits its turn there. What the notebook tells of a cell while an edit of
// it is on its way is older than what was typed since, and writes over none of it; once every edit of the cell is
// answered, the cell shows what the notebook holds. What is typed while the page has no socket, or sent on one that went
// before it was answered, is kept, marked on its cell as not sent yet, and sent once the page has a socket again. When the
// connection goes while the notebook has changes not saved, the page says they are lost if the server has stopped.
public sealed partial class PageTests
{
    [Fact]
    public async Task TypingDuringARun_ThenAReloadAtOnce_ReachesEveryTab_OnceTheRunIsStopped()
    {
        await SaveAsync("typing.verso", EndlessCell(), new CellModel { Type = "code", Language = "csharp", Source = "var kept = 1;" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var first = await context.NewPageAsync();
        var second = await context.NewPageAsync();

        foreach (var page in new[] { first, second })
        {
            await page.GotoAsync($"{served.Address}?token={Token}&notebook=typing.verso");
            await Expect(Cell(page, 1).Locator("textarea.source")).ToHaveValueAsync("var kept = 1;");
        }

        await Cell(first, 0).Locator("button.run").ClickAsync();
        await Expect(second.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

        // Typed, and the page reloaded at once: what was typed left as it was typed.
        await Cell(first, 1).Locator("textarea.source").FillAsync("var typed = 2;");
        await first.ReloadAsync();

        await second.Locator("#stop").ClickAsync();

        foreach (var page in new[] { first, second })
        {
            await Expect(Cell(page, 1).Locator("textarea.source")).ToHaveValueAsync("var typed = 2;", new() { Timeout = 30_000 });
        }
    }

    [Fact]
    public async Task EachKeystroke_LeavesThePageAsItIsTyped_WithNoWaitForTheTypingToSettle()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var edited = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

        page.WebSocket += (_, socket) => socket.FrameSent += (_, frame) =>
        {
            if (frame.Text?.Contains("\"ask\":\"edit\"", StringComparison.Ordinal) == true)
            {
                edited.TrySetResult(frame.Text);
            }
        };

        // The page's clock stands still once the notebook is drawn: nothing that waits on it ever comes.
        await page.Clock.InstallAsync(new() { TimeDate = DateTime.UnixEpoch });
        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");
        await Expect(Cell(page, 0).Locator("textarea.source")).ToHaveValueAsync("var a = 1;");
        await page.Clock.PauseAtAsync(DateTime.UnixEpoch.AddHours(1));

        await Cell(page, 0).Locator("textarea.source").FillAsync("var b = 2;");

        var sent = await edited.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

        Assert.Contains("\"source\":\"var b = 2;\"", sent, StringComparison.Ordinal);
    }

    [Fact]
    public async Task WhatTheNotebookTellsOfACell_WhileAnEditOfItIsOnItsWay_NeverWritesOverWhatWasTypedSince()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "x" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        var text = Cell(page, 0).Locator("textarea.source");

        await Expect(text).ToHaveValueAsync("x");

        // The first keystroke leaves, and every one after it.
        await text.ClickAsync();
        await page.Keyboard.PressAsync("End");
        await page.Keyboard.TypeAsync("1");

        var first = await held.EditAsync();

        await page.Keyboard.PressAsync("Enter");
        await page.Keyboard.TypeAsync("2");
        await page.Keyboard.PressAsync("Enter");
        await page.Keyboard.TypeAsync("3");

        // What the notebook says of the first keystroke reaches the page: older than what is typed now, it writes over
        // none of it, and the text keeps the height of what the person has.
        await held.LetThroughAsync(frame => HeldSocket.Answers(frame, first));
        await Expect(page.Locator("#save .unsaved")).ToBeVisibleAsync();
        await Expect(text).ToHaveValueAsync("x1\n2\n3");
        await Expect(text).ToHaveAttributeAsync("rows", "3");

        // Typed on, and everything the server told let through: the notebook holds what was typed, and so does the page.
        await page.Keyboard.TypeAsync("4");

        var flowing = held.FlowAsync();

        for (var waited = 0; (await CurrentAsync(served, "code.verso")).Cells[0].Source != "x1\n2\n34"; waited += 50)
        {
            Assert.True(waited < 10_000, "the typing never reached the notebook whole");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        await Expect(text).ToHaveValueAsync("x1\n2\n34");

        held.Stop();
        await flowing;
    }

    [Fact]
    public async Task AnotherPagesText_ToldBeforeThisPagesOwnEditIsAnswered_IsShownOnceItIs()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "x" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        var text = Cell(page, 0).Locator("textarea.source");

        await Expect(text).ToHaveValueAsync("x");

        // This page's keystroke is made, and what the server tells of it waits in line.
        await text.ClickAsync();
        await page.Keyboard.PressAsync("End");
        await page.Keyboard.TypeAsync("1");

        var edit = await held.EditAsync();
        var before = await held.TakeUntilAsync(frame => HeldSocket.Answers(frame, edit));
        var answer = before[^1];

        // Another page types into the cell after it. The server tells the change before it answers this page's edit, as
        // it does whenever that change is made before it gets to the answer: every change waiting goes first.
        await using var other = await SocketAsync(served, "code.verso");
        var cell = (await other.SnapshotAsync()).Version.Cells[0].Id;

        await other.AskAsync("edit", new { cell, source = "from another page" });

        var after = await held.TakeUntilAsync(frame => HeldSocket.Type(frame) == "change" && frame.GetRawText().Contains("from another page", StringComparison.Ordinal));

        foreach (var frame in before[..^1].Concat(after).Append(answer))
        {
            held.Tell(frame);
        }

        // Once its own edit is answered, the page shows what the notebook holds.
        await Expect(text).ToHaveValueAsync("from another page");
        Assert.Equal("from another page", (await CurrentAsync(served, "code.verso")).Cells[0].Source);
    }

    [Fact]
    public async Task WhatIsTypedWhileThePageHasNoSocket_IsKept_AndSentOnceItHasOneAgain()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        var text = Cell(page, 0).Locator("textarea.source");

        await Expect(text).ToHaveValueAsync("var a = 1;");

        // The socket goes, and the page tries in vain to open another while the person types.
        held.Refusing = true;
        await held.CloseAsync();
        await Expect(page.Locator("#status")).ToHaveTextAsync("Reconnecting…");

        await text.FillAsync("var kept = 3;");

        var tries = held.Refused;

        for (var waited = 0; held.Refused < tries + 2; waited += 50)
        {
            Assert.True(waited < 30_000, "the page stopped opening its socket again");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        // The page reaches the server again: the notebook is drawn afresh with what was typed, before the notebook has
        // answered it, and the notebook is sent it.
        held.Refusing = false;
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");
        await Expect(text).ToHaveValueAsync("var kept = 3;");

        for (var waited = 0; (await CurrentAsync(served, "code.verso")).Cells[0].Source != "var kept = 3;"; waited += 50)
        {
            Assert.True(waited < 30_000, "what was typed while the page had no socket never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        var flowing = held.FlowAsync();

        await Expect(text).ToHaveValueAsync("var kept = 3;");

        held.Stop();
        await flowing;
    }

    [Fact]
    public async Task WhatWasTypedOnASocketThatWentBeforeItWasAnswered_IsSentAgainOnTheNextOne_AndNotSaidToBeLost()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var sockets = Channel.CreateUnbounded<IWebSocketRoute>();
        var lost = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var firstSocket = 1;

        // The page's first socket loses every edit sent on it: none reaches the server, and none is answered.
        await page.RouteWebSocketAsync("**/socket", socket =>
        {
            var server = socket.ConnectToServer();
            var losing = Interlocked.Exchange(ref firstSocket, 0) == 1;

            socket.OnMessage(frame =>
            {
                if (frame.Text is not { } asked)
                {
                    return;
                }

                if (losing && asked.Contains("\"ask\":\"edit\"", StringComparison.Ordinal))
                {
                    lost.TrySetResult();

                    return;
                }

                server.Send(asked);
            });
            sockets.Writer.TryWrite(socket);
        });

        // The page's clock stands still once the notebook is drawn, so the page opens no socket again until it runs on.
        await page.Clock.InstallAsync(new() { TimeDate = DateTime.UnixEpoch });
        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");

        var text = Cell(page, 0).Locator("textarea.source");

        await Expect(text).ToHaveValueAsync("var a = 1;");
        await page.Clock.PauseAtAsync(DateTime.UnixEpoch.AddHours(1));

        await text.FillAsync("var again = 4;");
        await lost.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

        // The socket goes before the edit is answered: the page says it reconnects, and says nothing of the edit, which it
        // sends again.
        await (await sockets.Reader.ReadAsync(TestContext.Current.CancellationToken)).CloseAsync(new() { Code = 3000, Reason = "the server went" });
        await Expect(page.Locator("#status")).ToHaveTextAsync("Reconnecting…");

        await page.Clock.ResumeAsync();

        for (var waited = 0; (await CurrentAsync(served, "code.verso")).Cells[0].Source != "var again = 4;"; waited += 50)
        {
            Assert.True(waited < 30_000, "what was typed on the socket that went never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        await Expect(text).ToHaveValueAsync("var again = 4;");
    }

    [Fact]
    public async Task WhatWasTypedOnASocketThatWent_IntoACellAnotherPageTookAway_IsNotSentAgain()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" }, new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var sockets = Channel.CreateUnbounded<IWebSocketRoute>();
        var lost = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var edits = 0;
        var firstSocket = 1;

        // The page's first socket loses every edit sent on it; every edit on another reaches the server, and is counted.
        await page.RouteWebSocketAsync("**/socket", socket =>
        {
            var server = socket.ConnectToServer();
            var losing = Interlocked.Exchange(ref firstSocket, 0) == 1;

            socket.OnMessage(frame =>
            {
                if (frame.Text is not { } asked)
                {
                    return;
                }

                if (asked.Contains("\"ask\":\"edit\"", StringComparison.Ordinal))
                {
                    if (losing)
                    {
                        lost.TrySetResult();

                        return;
                    }

                    Interlocked.Increment(ref edits);
                }

                server.Send(asked);
            });
            sockets.Writer.TryWrite(socket);
        });
        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");

        var text = Cell(page, 0).Locator("textarea.source");

        await Expect(text).ToHaveValueAsync("var a = 1;");
        await text.FillAsync("var gone = 5;");
        await lost.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

        // Another page takes the cell away, and this page draws it gone before its socket goes.
        await using var other = await SocketAsync(served, "code.verso");
        var cell = (await other.SnapshotAsync()).Version.Cells[0].Id;

        await other.AskAsync("remove", new { cell });
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(1);

        await (await sockets.Reader.ReadAsync(TestContext.Current.CancellationToken)).CloseAsync(new() { Code = 3000, Reason = "the server went" });
        await sockets.Reader.ReadAsync(TestContext.Current.CancellationToken).AsTask().WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        await Expect(page.Locator("#status")).ToHaveTextAsync(string.Empty, new() { Timeout = 10_000 });

        // Nothing of the cell that went is sent again, and nothing is said of it.
        await page.Locator("#save").ClickAsync();
        await Expect(page.Locator("#notice")).ToHaveTextAsync("Saved to code.verso");
        await Expect(page.Locator("#error")).ToBeHiddenAsync();
        Assert.Equal(0, Volatile.Read(ref edits));
    }

    [Fact]
    public async Task AReconnect_LeavesTheCellBeingWrittenAsItWas_ItsTextFocusAndSelection_AndRunsNothing()
    {
        await SaveAsync("notes.verso", new CellModel { Type = "markdown", Source = "# Notes" }, new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=notes.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        var flowing = held.FlowAsync();
        var text = Cell(page, 0).Locator("textarea.source");

        // The Markdown cell is selected, shows its text, and is written in.
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await text.ClickAsync();
        await page.Keyboard.PressAsync("End");
        await page.Keyboard.TypeAsync(" and more");

        for (var waited = 0; (await CurrentAsync(served, "notes.verso")).Cells[0].Source != "# Notes and more"; waited += 50)
        {
            Assert.True(waited < 10_000, "the typing never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        var asked = held.Asked.Count;

        // The socket goes, and the page opens another and draws the notebook from it.
        await held.CloseAsync();
        await Expect(page.Locator("#status")).ToHaveTextAsync(new Regex("^Reconnecting…"));
        await Expect(page.Locator("#status")).ToHaveTextAsync(string.Empty, new() { Timeout = 10_000 });

        // The cell is as the person left it: selected, its text theirs, the cursor still in it, and nothing was run.
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(text).ToHaveValueAsync("# Notes and more");
        await Expect(text).ToBeFocusedAsync();
        Assert.DoesNotContain("run", held.Asked.Skip(asked));

        await page.Keyboard.TypeAsync("!");

        for (var waited = 0; (await CurrentAsync(served, "notes.verso")).Cells[0].Source != "# Notes and more!"; waited += 50)
        {
            Assert.True(waited < 10_000, "the typing after the reconnect never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        held.Stop();
        await flowing;
    }

    [Fact]
    public async Task TheConnectionGoingWhileChangesAreUnsaved_SaysTheyAreLostIfTheServerHasStopped()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        var flowing = held.FlowAsync();

        await Cell(page, 0).Locator("textarea.source").FillAsync("var b = 2;");
        await Expect(page.Locator("#save .unsaved")).ToBeVisibleAsync(new() { Timeout = 10_000 });

        held.Refusing = true;
        await held.CloseAsync();

        await Expect(page.Locator("#status")).ToHaveTextAsync("Reconnecting… What was not saved is lost if the server has stopped.");

        held.Stop();
        await flowing;
    }

    [Fact]
    public async Task TypingNotSentYet_IsMarkedOnItsCell_UntilItIsSent()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=code.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        held.Refusing = true;
        await held.CloseAsync();
        await Expect(page.Locator("#status")).ToHaveTextAsync("Reconnecting…");
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex(@"\bunsent\b"));

        await Cell(page, 0).Locator("textarea.source").FillAsync("var kept = 3;");
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\bunsent\b"));

        // Chosen meanwhile, the cell is drawn again, and stays marked.
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\bunsent\b"));

        // The page reaches the server again, and sends it: the mark goes as it is sent, before it is answered, and stays
        // gone once it is.
        held.Refusing = false;
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");
        await held.EditAsync();
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex(@"\bunsent\b"));

        var flowing = held.FlowAsync();

        for (var waited = 0; (await CurrentAsync(served, "code.verso")).Cells[0].Source != "var kept = 3;"; waited += 50)
        {
            Assert.True(waited < 10_000, "what was kept never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex(@"\bunsent\b"));

        held.Stop();
        await flowing;
    }

    [Fact]
    public async Task AnotherPageAddingACellAboveTheOneBeingTypedIn_LeavesTheTypingWhereItIs()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" }, new CellModel { Type = "code", Language = "csharp", Source = "x" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "code.verso");
        var text = Cell(page, 1).Locator("textarea.source");

        await Expect(text).ToHaveValueAsync("x");
        await text.ClickAsync();
        await page.Keyboard.PressAsync("End");
        await page.Keyboard.TypeAsync("1");

        // Another page adds a cell between the two.
        await using var other = await SocketAsync(served, "code.verso");
        var first = (await other.SnapshotAsync()).Version.Cells[0].Id;

        await other.AskAsync("add", new { after = first, type = "code", language = "csharp" });
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(3);

        // The person types on, in the cell they were in.
        await Expect(Cell(page, 2).Locator("textarea.source")).ToBeFocusedAsync();
        await page.Keyboard.TypeAsync("2");

        for (var waited = 0; (await CurrentAsync(served, "code.verso")).Cells[2].Source != "x12"; waited += 50)
        {
            Assert.True(waited < 10_000, "the typing after the cell was added never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task AnotherPageTakingAwayTheCellBeingWritten_RunsNothing_AndSaysNoRefusal()
    {
        await SaveAsync("notes.verso", new CellModel { Type = "markdown", Source = "# Notes" }, new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=notes.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        var flowing = held.FlowAsync();

        // The Markdown cell is selected, shows its text, and has the cursor.
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Cell(page, 0).Locator("textarea.source").ClickAsync();

        var asked = held.Asked.Count;

        // Another page takes it away: the page draws it gone, and neither runs it nor says it was refused.
        await using var other = await SocketAsync(served, "notes.verso");
        var cell = (await other.SnapshotAsync()).Version.Cells[0].Id;

        await other.AskAsync("remove", new { cell });
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(1);
        await Expect(page.Locator("#status")).ToHaveTextAsync(string.Empty);
        Assert.DoesNotContain("run", held.Asked.Skip(asked));

        held.Stop();
        await flowing;
    }

    // A page's socket to the server, what the server tells the page held in line until the test lets it through, in the
    // order it was told; what the page asks goes on at once, and the id of each edit it sends is kept. While the test
    // keeps the page from the server, each socket the page opens is closed at once, as it is while a server cannot be
    // reached.
    private sealed class HeldSocket
    {
        private readonly Channel<string> _told = Channel.CreateUnbounded<string>();
        private readonly Channel<long> _edits = Channel.CreateUnbounded<long>();
        private readonly ConcurrentQueue<string> _asked = new();
        private IWebSocketRoute? _page;
        private int _refusing;
        private int _refused;

        public bool Refusing
        {
            set => Volatile.Write(ref _refusing, value ? 1 : 0);
        }

        public int Refused => Volatile.Read(ref _refused);

        // What the page asked, in the order it asked it.
        public IReadOnlyCollection<string> Asked => _asked;

        public static async Task<HeldSocket> RouteAsync(IPage page)
        {
            var held = new HeldSocket();

            await page.RouteWebSocketAsync("**/socket", socket =>
            {
                if (Volatile.Read(ref held._refusing) == 1)
                {
                    Interlocked.Increment(ref held._refused);
                    _ = socket.CloseAsync(new() { Code = 3000, Reason = "the server cannot be reached" });

                    return;
                }

                var server = socket.ConnectToServer();

                held._page = socket;
                socket.OnMessage(frame =>
                {
                    if (frame.Text is { } asked)
                    {
                        using var ask = JsonDocument.Parse(asked);

                        held._asked.Enqueue(ask.RootElement.GetProperty("ask").GetString()!);

                        if (ask.RootElement.GetProperty("ask").GetString() == "edit")
                        {
                            held._edits.Writer.TryWrite(ask.RootElement.GetProperty("id").GetInt64());
                        }

                        server.Send(asked);
                    }
                });
                server.OnMessage(frame =>
                {
                    if (frame.Text is { } said)
                    {
                        held._told.Writer.TryWrite(said);
                    }
                });
            });

            return held;
        }

        public static string? Type(JsonElement frame) => frame.GetProperty("type").GetString();

        public static bool Answers(JsonElement frame, long ask) => Type(frame) == "answer" && frame.GetProperty("id").GetInt64() == ask;

        // The id of the next edit the page sends.
        public async Task<long> EditAsync() =>
            await _edits.Reader.ReadAsync(TestContext.Current.CancellationToken).AsTask().WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

        // What the server told, held back, up to and with the first frame that is the last one meant.
        public async Task<List<JsonElement>> TakeUntilAsync(Func<JsonElement, bool> last)
        {
            var taken = new List<JsonElement>();

            while (true)
            {
                var said = await _told.Reader.ReadAsync(TestContext.Current.CancellationToken).AsTask().WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

                using var frame = JsonDocument.Parse(said);

                taken.Add(frame.RootElement.Clone());

                if (last(taken[^1]))
                {
                    return taken;
                }
            }
        }

        // Lets through to the page what the server told, up to and with the first frame that is the last one meant.
        public async Task LetThroughAsync(Func<JsonElement, bool> last)
        {
            foreach (var frame in await TakeUntilAsync(last))
            {
                Tell(frame);
            }
        }

        // Lets through everything the server tells from now on, until the test stops it.
        public Task FlowAsync() => Task.Run(
            async () =>
            {
                await foreach (var said in _told.Reader.ReadAllAsync(TestContext.Current.CancellationToken))
                {
                    _page!.Send(said);
                }
            },
            TestContext.Current.CancellationToken);

        public void Stop() => _told.Writer.TryComplete();

        // The page's socket goes, as it goes when the server does.
        public Task CloseAsync() => _page!.CloseAsync(new() { Code = 3000, Reason = "the server went" });

        public void Tell(JsonElement frame) => _page!.Send(frame.GetRawText());
    }
}

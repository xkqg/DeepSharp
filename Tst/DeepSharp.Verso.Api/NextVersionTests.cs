// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;

namespace DeepSharp.Tests.Api;

/// <summary>
/// The version a turn publishes next, as whoever asked for it waits for it: everyone who asked before it is made is told
/// it, however many asked at once, and whoever asks while it is made is told the one after.
/// </summary>
public sealed class NextVersionTests
{
    private static readonly HostedLayout InTheNotebook = new("notebook", (LayoutAllows)255, HasPropertiesPanel: true);

    private static NotebookVersion Version(long number) => new(number, [], null, [], InTheNotebook, [], new HostedKernels(0, Restarting: false, Fault: null), Unsaved: false, ThemeId: null, Arrangement: default, Metadata: default);

    [Fact]
    public async Task TwoWhoAskBeforeAVersionIsMade_AreBothToldIt()
    {
        var next = new NextVersion();
        var one = next.AskAsync();
        var two = next.AskAsync();

        Assert.Equal(Version(1), await next.TellAsync(() => Task.FromResult(Version(1))));
        Assert.Equal(Version(1), await one.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));
        Assert.Equal(Version(1), await two.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task WhoAsksWhileAVersionIsMade_IsToldTheNext()
    {
        var next = new NextVersion();
        Task<NotebookVersion>? meanwhile = null;

        await next.TellAsync(() =>
        {
            meanwhile = next.AskAsync();

            return Task.FromResult(Version(1));
        });

        Assert.False(meanwhile!.IsCompleted);

        await next.TellAsync(() => Task.FromResult(Version(2)));

        Assert.Equal(Version(2), await meanwhile.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task AVersionNobodyAskedFor_IsMadeAllTheSame()
    {
        var next = new NextVersion();

        Assert.Equal(Version(3), await next.TellAsync(() => Task.FromResult(Version(3))));
    }
}

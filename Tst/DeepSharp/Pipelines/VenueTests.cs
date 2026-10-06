// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The venue for the length of one landing: made when it begins and let go when it ends. A client it made itself is let go
/// with it; a handler a caller handed in is theirs, and is never disposed here.
/// </summary>
public sealed class VenueTests
{
    [Fact]
    public async Task AVenueOpenedWithNoHandlerHandedIn_AsksThroughAClientOfItsOwn_OverARealSocket()
    {
        // The one place the package's own client is used for real: a stand-in that listens on this machine, which the
        // suite's dead proxy lets through because it is this machine's own address.
        using var fake = new FakeBinanceVenue(TimeProvider.System);
        using var socket = new StandIn(fake);
        using var venue = Venue.Open(null, new Uri($"http://127.0.0.1:{socket.Port}"), TimeProvider.System);

        var said = await venue.ServerTimeAsync(TestContext.Current.CancellationToken);

        Assert.InRange(said, DateTime.UtcNow.AddMinutes(-1), DateTime.UtcNow.AddMinutes(1));
        Assert.Equal(DateTimeKind.Utc, said.Kind);
        Assert.Equal(["/api/v3/time"], fake.Seen.Select(request => request.Path));
    }

    [Fact]
    public async Task AHandlerHandedIn_IsUsed_AndIsKept_WhenTheVenueIsLetGo()
    {
        using var fake = new FakeBinanceVenue(TimeProvider.System);
        using var spy = new Spy(fake);

        using (var venue = Venue.Open(spy, new Uri("https://venue.test"), TimeProvider.System))
        {
            await venue.ServerTimeAsync(TestContext.Current.CancellationToken);
        }

        Assert.Single(fake.Seen);
        Assert.False(spy.LetGo, "A handler a caller handed in was let go by the venue that used it.");
    }

    private sealed class Spy(HttpMessageHandler inner) : DelegatingHandler(inner)
    {
        public bool LetGo { get; private set; }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                LetGo = true;
            }

            base.Dispose(disposing);
        }
    }
}

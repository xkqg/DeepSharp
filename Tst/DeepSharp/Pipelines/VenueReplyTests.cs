// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One answer of the venue before anybody has decided what it means: its clock, as its <c>Date</c> header says it, and what
/// it said of a refusal in its own words — each read as far as the answer goes and never further, since a refusal that
/// could not be read still has to be told as a refusal.
/// </summary>
public sealed class VenueReplyTests
{
    private static VenueReply Said(string body, string? date = null) =>
        new(HttpStatusCode.BadRequest, Encoding.UTF8.GetBytes(body), Used: null, RetryAfter: null, Date: date, Uuid: null);

    [Fact]
    public void TheVenuesDateHeader_IsItsOwnClock_ToTheSecond_InUniversalTime()
    {
        var clock = Said("{}", "Tue, 06 Oct 2026 08:00:01 GMT").DateValue;

        Assert.Equal(new DateTimeOffset(2026, 10, 6, 8, 0, 1, TimeSpan.Zero), clock);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("sometime after lunch")]
    public void ADateHeaderThatIsNoMoment_SaysNothingOfTheVenuesClock(string? date)
    {
        Assert.Null(Said("{}", date).DateValue);
    }

    [Fact]
    public void ARefusal_IsReadAsTheCodeAndTheMessageTheVenueGave()
    {
        var words = Said("{\"code\":-1121,\"msg\":\"Invalid symbol.\"}").Words();

        Assert.Equal(new VenueWords(-1121, "Invalid symbol."), words);
    }

    [Theory]
    [InlineData("{\"msg\":\"Too much.\"}", null, "Too much.")]
    [InlineData("{\"code\":-1003}", -1003, null)]
    [InlineData("{\"code\":\"-1003\",\"msg\":\"As text.\"}", null, "As text.")]
    [InlineData("{\"code\":99999999999,\"msg\":\"Beyond a whole number.\"}", null, "Beyond a whole number.")]
    [InlineData("{\"code\":-1,\"msg\":7}", -1, null)]
    [InlineData("{}", null, null)]
    [InlineData("[1,2]", null, null)]
    [InlineData("\"words\"", null, null)]
    [InlineData("<html>Bad Gateway</html>", null, null)]
    [InlineData("", null, null)]
    public void ARefusal_IsReadAsFarAsItGoes_AndNothingItDoesNotSay(string body, int? code, string? message)
    {
        Assert.Equal(new VenueWords(code, message), Said(body).Words());
    }
}

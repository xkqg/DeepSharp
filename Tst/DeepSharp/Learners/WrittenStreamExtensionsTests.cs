// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The text of what a writer wrote into a stream, each line ended with a line feed alone, whatever the machine ends its
/// lines with: made once, at the file's own length, so the longest file one machine writes is the longest every machine
/// writes.
/// </summary>
public class WrittenStreamExtensionsTests
{
    [Fact]
    public void LinesAWriterEndedWithCarriageReturns_EndWithALineFeedAlone_AndACarriageReturnEscapedInAString_StaysEscaped()
    {
        using var stream = new MemoryStream();
        stream.Write("{\r\n  \"größe €\": \"a\\r\\nb 😀\",\r\n  \"n\": 1\r\n}"u8);

        Assert.Equal("{\n  \"größe €\": \"a\\r\\nb 😀\",\n  \"n\": 1\n}", stream.TextWithLineFeeds());
    }

    [Fact]
    public void LinesEndedWithALineFeedAlready_AreTheTextAsWritten()
    {
        using var stream = new MemoryStream();
        stream.Write("{\n  \"n\": 1\n}"u8);

        Assert.Equal("{\n  \"n\": 1\n}", stream.TextWithLineFeeds());
    }
}

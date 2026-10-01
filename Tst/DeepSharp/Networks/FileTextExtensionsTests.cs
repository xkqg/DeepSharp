// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What a file holds, shown in a message about it: every character that would break the message's line, hide what follows
/// it or turn it round written as its escape, the backslash too so an escape is never the file's own text, and a long text
/// cut short with how long it was, so a file can put nothing into a message but the words it holds.
/// </summary>
/// <remarks>The text of these tests is written as code points and its escapes as a backslash and letters, so this file holds none of what it tests.</remarks>
public class FileTextExtensionsTests
{
    private const char Backslash = (char)92;

    public static TheoryData<string, string> Words() => new()
    {
        { "0.weight", "0.weight" },
        { Of('e', 0xE9, ' ', 0x65E5, 0x672C, ' ', 0xD83D, 0xDE00), Of('e', 0xE9, ' ', 0x65E5, 0x672C, ' ', 0xD83D, 0xDE00) },
        { "tensor 'w'", "tensor 'w'" },
        { Of('a', 13, 10, 'b', 9, 'c'), $"a{Esc("r")}{Esc("n")}b{Esc("t")}c" },
        { Of(0x00, 0x01, 0x1B, 0x7F, 0x85, 0x9F), $"{Esc("x00")}{Esc("x01")}{Esc("x1b")}{Esc("x7f")}{Esc("x85")}{Esc("x9f")}" },
        { Of('y', 0x202E, 't', 'x', 't'), $"y{Esc("u202e")}txt" },
        { Of('z', 0x200B, 'w', 0xFEFF), $"z{Esc("u200b")}w{Esc("ufeff")}" },
        { Of('l', 0x2028, 'p', 0x2029), $"l{Esc("u2028")}p{Esc("u2029")}" },
        { Of('a', Backslash, 'b'), $"a{Esc(Backslash.ToString())}b" },
        { Of(Backslash, 'n'), $"{Esc(Backslash.ToString())}n" },
    };

    [Theory]
    [MemberData(nameof(Words))]
    public void TheWordsAFileHolds_AreShownAsTheyAre_AndEverythingElseAsItsEscape(string held, string shown) =>
        Assert.Equal(shown, held.Quoted());

    [Fact]
    public void HalfOfASurrogatePair_StandingAlone_IsShownAsItsEscape_AndAWholePairAsItIs()
    {
        Assert.Equal($"extra{Esc("udc80")}", Of('e', 'x', 't', 'r', 'a', 0xDC80).Quoted());
        Assert.Equal($"{Esc("ud800")}x", Of(0xD800, 'x').Quoted());
        Assert.Equal($"end{Esc("ud83d")}", Of('e', 'n', 'd', 0xD83D).Quoted());
        Assert.Equal($"{Esc("udc80")}{Esc("ud800")}", Of(0xDC80, 0xD800).Quoted());
        Assert.Equal(Of(0xD83D, 0xDE00), Of(0xD83D, 0xDE00).Quoted());
    }

    [Fact]
    public void TextPastTwoHundredCharacters_IsCutShort_SayingHowLongItWas()
    {
        Assert.Equal(new string('m', 200), new string('m', 200).Quoted());
        Assert.Equal($"{new string('m', 200)}… (201 characters)", new string('m', 201).Quoted());
        Assert.Equal($"{new string('m', 200)}… (1000000 characters)", new string('m', 1_000_000).Quoted());
    }

    [Fact]
    public void EscapesCountTowardsTheTwoHundred_AndAreNeverCutInHalf()
    {
        // A hundred and fifty line breaks: a hundred of them fill the two hundred characters shown.
        Assert.Equal($"{string.Concat(Enumerable.Repeat(Esc("n"), 100))}… (150 characters)", new string((char)10, 150).Quoted());

        // One letter, then escapes of six: the one that would pass two hundred is left out whole.
        Assert.Equal($"m{string.Concat(Enumerable.Repeat(Esc("u202e"), 33))}… (41 characters)", ("m" + new string((char)0x202E, 40)).Quoted());

        // A whole surrogate pair is never parted: the pair that would pass two hundred is left out whole.
        Assert.Equal($"{new string('m', 199)}… (201 characters)", (new string('m', 199) + Of(0xD83D, 0xDE00)).Quoted());
    }

    [Fact]
    public void NothingIsQuotedOfNoText() => Assert.Throws<ArgumentNullException>(() => ((string)null!).Quoted());

    // Text of the given code units.
    private static string Of(params int[] units) => new([.. units.Select(unit => (char)unit)]);

    // An escape as a message shows it: a backslash, then what follows it.
    private static string Esc(string after) => Backslash + after;
}

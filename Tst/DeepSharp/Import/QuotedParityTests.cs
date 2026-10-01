// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests.Import;

/// <summary>
/// A file's text is shown in a message by one rule, kept twice: in the deep-learning core, for a network's file and every
/// importer, and in the pipelines package, for a pipeline's file, since a pipeline never carries the core. A rule kept
/// twice holds only while the two are compared: this is that comparison, over every kind of text the rule treats apart.
/// </summary>
public class QuotedParityTests
{
    public static TheoryData<string> Texts => new()
    {
        string.Empty,
        "survived",
        "Größe \U0001F600",
        "name\r\nFAKE LINE",
        "tab\tand\u0085next line",
        @"back\slash",
        "\u200Bformat\u202Eright-to-left and\u2028a line separator",
        "lone \uD800 high and \uDC00 low",
        new string('x', 199),
        new string('x', 200),
        new string('x', 201),
        new string('x', 199) + "\U0001F600" + "tail",
        new string('y', 10_000),
    };

    [Theory]
    [MemberData(nameof(Texts))]
    public void ANetworksFileAndAPipelinesFile_ShowTheSameTextTheSameWay(string text) =>
        Assert.Equal(DeepSharp.Networks.FileTextExtensions.Quoted(text), DeepSharp.Pipelines.FileTextExtensions.Quoted(text));
}

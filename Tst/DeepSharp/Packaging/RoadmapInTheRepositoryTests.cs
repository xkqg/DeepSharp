// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// The roadmap is the wiki's page, and the repository carries its table too, so a reader of the code sees what is done and
/// what comes next without leaving it. Two copies of one table drift apart, so the repository's is held to the wiki's: the
/// same rows in the same order, a link to a wiki page written as the address GitHub shows it at, and every row's section
/// left to the wiki.
/// </summary>
public sealed partial class RoadmapInTheRepositoryTests
{
    [GeneratedRegex("^\\| \\*\\*(?:Done|Later)\\*\\* \\|.*$", RegexOptions.Multiline)]
    private static partial Regex Row();

    [GeneratedRegex("\\[\\[(?<page>[^\\]]+)\\]\\]")]
    private static partial Regex WikiLink();

    [Fact]
    public void TheRepositorysRoadmap_HoldsTheWikisTable_RowForRow_AndSendsTheReaderThereForTheRest()
    {
        var wiki = File.ReadAllText(Path.Join(Wiki.Folder, "Roadmap.md")).ReplaceLineEndings("\n");
        var here = File.ReadAllText(Path.Join(Repository.Root, "ROADMAP.md")).ReplaceLineEndings("\n");
        string[] expected = [.. Row().Matches(wiki).Select(row => WikiLink().Replace(row.Value, link => $"[{link.Groups["page"].Value}](https://github.com/xkqg/DeepSharp/wiki/{link.Groups["page"].Value.Replace(' ', '-')})"))];

        Assert.NotEmpty(expected);
        Assert.Equal(expected, Row().Matches(here).Select(row => row.Value));
        Assert.Contains("https://github.com/xkqg/DeepSharp/wiki/Roadmap", here, StringComparison.Ordinal);
    }
}

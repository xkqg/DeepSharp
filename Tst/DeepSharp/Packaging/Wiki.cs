// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.RegularExpressions;

namespace DeepSharp.Tests;

/// <summary>
/// The wiki's pages and the C# blocks on them, read by one rule for every suite that reads them.
/// </summary>
/// <remarks>
/// A block is a program of its own unless the line above it says otherwise, in a comment the page does not show:
/// <c>&lt;!-- example: name --&gt;</c> names a block others go on from, <c>&lt;!-- continues: name --&gt;</c> goes on from
/// that block and every block that went on from it before, and <c>&lt;!-- a C# cell of a notebook --&gt;</c> is a notebook's
/// cell. A comment of any other words, an example named twice and one nobody named are refused, so no block is passed over.
/// The wiki is found beside the repository, where GitHub keeps it as a repository of its own and the build machine clones it.
/// </remarks>
internal static partial class Wiki
{
    /// <summary>The comment above a block that is a notebook's C# cell.</summary>
    public const string Cell = "a C# cell of a notebook";

    /// <summary>The wiki's folder.</summary>
    public static string Folder => Path.GetFullPath(Path.Join(Repository.Root, "..", "DeepSharp.wiki"));

    [GeneratedRegex("^(?:<!-- (?<marker>[^\\n]*?) -->\\n)?```csharp\\n(?<code>.*?)^```", RegexOptions.Singleline | RegexOptions.Multiline)]
    private static partial Regex CSharpBlock();

    [GeneratedRegex("^(?<kind>example|continues): (?<name>[a-z-]+)$")]
    private static partial Regex Marker();

    [GeneratedRegex("^\\| (?<at>\\d+) \\| \\[\\[(?<title>[^\\]]+)\\]\\] \\|", RegexOptions.Multiline)]
    private static partial Regex TutorialRow();

    /// <summary>Every page of the wiki, its blocks read into programs and cells.</summary>
    /// <returns>The pages, by name.</returns>
    public static IReadOnlyList<WikiPage> Pages()
    {
        Assert.True(
            File.Exists(Path.Join(Folder, "Home.md")),
            $"The wiki is not beside the repository, at {Folder}: clone it there — git clone https://github.com/xkqg/DeepSharp.wiki.git — as the build machine does.");

        return [.. Directory.GetFiles(Folder, "*.md").Order(StringComparer.Ordinal).Select(path => Read(Path.GetFileName(path), File.ReadAllText(path).ReplaceLineEndings("\n")))];
    }

    /// <summary>The steps of the tutorial, in the order the table on its start page gives them, each with the page it names.</summary>
    /// <returns>The steps, numbered as the table numbers them.</returns>
    public static IReadOnlyList<TutorialStep> Tutorial()
    {
        var pages = Pages().ToDictionary(page => page.Name, StringComparer.Ordinal);
        var table = File.ReadAllText(Path.Join(Folder, "Getting-Started.md")).ReplaceLineEndings("\n");

        return [.. TutorialRow().Matches(table).Select(row =>
        {
            var title = row.Groups["title"].Value;

            return new TutorialStep(int.Parse(row.Groups["at"].Value, CultureInfo.InvariantCulture), title, pages[$"{title.Replace(' ', '-')}.md"]);
        })];
    }

    /// <summary>A page's C# blocks: each a program of its own, a program going on from an example, or a notebook's cell.</summary>
    /// <param name="name">The page's name.</param>
    /// <param name="text">Its text, a line feed between lines.</param>
    /// <returns>The page.</returns>
    /// <exception cref="InvalidOperationException">A comment above a block is none of the three, or names an example twice or one nobody named.</exception>
    public static WikiPage Read(string name, string text)
    {
        var examples = new Dictionary<string, List<WikiBlock>>(StringComparer.Ordinal);
        var programs = new List<WikiProgram>();
        var cells = new List<WikiBlock>();

        foreach (Match match in CSharpBlock().Matches(text))
        {
            var block = new WikiBlock(text[..match.Groups["code"].Index].Count(letter => letter == '\n') + 1, match.Groups["code"].Value);
            var marker = match.Groups["marker"];

            if (!marker.Success)
            {
                programs.Add(new WikiProgram(block.Line, [block]));
            }
            else if (marker.Value == Cell)
            {
                cells.Add(block);
            }
            else if (Marker().Match(marker.Value) is { Success: true } said)
            {
                var example = said.Groups["name"].Value;

                if (said.Groups["kind"].Value == "example")
                {
                    if (!examples.TryAdd(example, [block]))
                    {
                        throw new InvalidOperationException($"{name}, line {block.Line}: the example '{example}' is named twice.");
                    }
                }
                else
                {
                    var before = examples.GetValueOrDefault(example)
                                 ?? throw new InvalidOperationException($"{name}, line {block.Line}: it continues '{example}', and no example above it has that name.");

                    before.Add(block);
                }

                programs.Add(new WikiProgram(block.Line, [.. examples[example]]));
            }
            else
            {
                throw new InvalidOperationException($"{name}, line {block.Line}: the comment '{marker.Value}' above the block is one this test does not know.");
            }
        }

        return new WikiPage(name, programs, cells);
    }
}

/// <summary>A C# block of a page: the line its code starts on, and the code.</summary>
/// <param name="Line">The line its code starts on, from one.</param>
/// <param name="Code">The code.</param>
internal readonly record struct WikiBlock(int Line, string Code);

/// <summary>What is compiled as one program: the line of the block it is compiled for, and the blocks it is made of, in order.</summary>
/// <param name="Line">The line of the block it is compiled for.</param>
/// <param name="Blocks">The blocks, in the order they stand.</param>
internal readonly record struct WikiProgram(int Line, IReadOnlyList<WikiBlock> Blocks);

/// <summary>A page: its name, the programs its blocks make, and the notebook cells it holds.</summary>
/// <param name="Name">The page's file name.</param>
/// <param name="Programs">The programs its blocks make.</param>
/// <param name="Cells">The notebook cells it holds.</param>
internal readonly record struct WikiPage(string Name, IReadOnlyList<WikiProgram> Programs, IReadOnlyList<WikiBlock> Cells);

/// <summary>A step of the tutorial: its number in the table, the title the table links to, and the page.</summary>
/// <param name="At">The step's number.</param>
/// <param name="Title">The title the table links to, as a link writes it.</param>
/// <param name="Page">The page.</param>
internal readonly record struct TutorialStep(int At, string Title, WikiPage Page);

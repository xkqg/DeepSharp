// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.RegularExpressions;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// The README's example is the first code anybody runs, and one the library has moved away from reads as a broken
/// library. So the example is compiled and run as it stands on the page, over a file of the shape it names: the
/// page cannot drift from the verbs, their arguments, or the rules a declaration keeps.
/// </summary>
public sealed partial class ReadmeExampleTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-readme-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [GeneratedRegex("```csharp\\n(?<code>.*?)```", RegexOptions.Singleline)]
    private static partial Regex CSharpBlock();

    [Fact]
    public void TheReadmesExample_CompilesAndRuns_OverAFileOfTheShapeItNames()
    {
        var readme = File.ReadAllText(Path.Join(Repository.Root, "README.md")).ReplaceLineEndings("\n");
        var example = CSharpBlock().Matches(readme).Select(match => match.Groups["code"].Value)
            .Single(code => code.Contains("Pdd.Create()", StringComparison.Ordinal));

        // Twenty days, a gap in every fifth count of trades.
        var file = Path.Join(_folder, "btceur-1d.csv");
        File.WriteAllText(file, "timestamp,trades,close\n" + string.Concat(Enumerable.Range(1, 20).Select(day =>
            string.Create(CultureInfo.InvariantCulture, $"2024-01-{day:00},{(day % 5 == 0 ? string.Empty : (day * 3).ToString(CultureInfo.InvariantCulture))},{day + 0.5}\n"))));

        var made = Assert.IsType<object[]>(Run(example.Replace("\"btceur-1d.csv\"", JsonSerializer.Serialize(file), StringComparison.Ordinal)));
        var trained = Assert.IsType<TrainedNetwork>(made[0]);
        var prepared = trained.Prepared;

        // Twenty days, the last of the training part kept apart as the gap the answer read from tomorrow needs.
        Assert.Equal(20, prepared.Table.RowCount);
        Assert.Equal(13, prepared.CountIn(Part.Train));
        Assert.Equal(20, trained.History!.Epochs.Count);
        Assert.Equal([Part.Validation, Part.Test], trained.Measures!.Parts.Select(part => part.Part));
        Assert.Contains("\"trainedOn\"", Assert.IsType<string>(made[1]), StringComparison.Ordinal);
    }

    [Fact]
    public void TheReadme_SaysWhereTheFileItsExampleReadsComesFrom_NamingEveryColumnTheExampleDeclares()
    {
        // The file the example reads ships nowhere, so the words under the example say whose it is and what it holds: a
        // column the example comes to declare, and the words do not name, is a file nobody can make.
        var readme = File.ReadAllText(Path.Join(Repository.Root, "README.md")).ReplaceLineEndings("\n");
        var example = CSharpBlock().Matches(readme).Select(match => match.Groups["code"].Value)
            .Single(code => code.Contains("Pdd.Create()", StringComparison.Ordinal));
        var file = ReadFile().Match(example).Groups["file"].Value;
        string[] columns = [.. DeclaredColumn().Matches(example).Select(match => match.Groups["column"].Value)];
        var below = readme[(readme.IndexOf(example, StringComparison.Ordinal) + example.Length)..];
        var said = below.Split("\n\n").FirstOrDefault(paragraph => paragraph.Contains($"`{file}`", StringComparison.Ordinal));

        Assert.Equal(["timestamp", "close", "trades"], columns);
        Assert.True(said is not null, $"Nothing below the README's example says where {file} comes from.");
        Assert.All(columns, column => Assert.Contains($"`{column}`", said, StringComparison.Ordinal));
    }

    [GeneratedRegex("\\.ReadCsv\\(\"(?<file>[^\"]+)\"\\)")]
    private static partial Regex ReadFile();

    [GeneratedRegex("\\.(?:Timestamp|Number|Integer|Category|Text|Boolean|Optional)\\(\"(?<column>[^\"]+)\"")]
    private static partial Regex DeclaredColumn();

    // The example as a method: its usings above, its statements inside, and what it trained and wrote handed back.
    private static object? Run(string example)
    {
        var lines = example.Split('\n');
        var source = $$"""
            {{string.Join('\n', lines.Where(line => line.StartsWith("using ", StringComparison.Ordinal)))}}

            public static class Example
            {
                public static object Run()
                {
            {{string.Join('\n', lines.Where(line => !line.StartsWith("using ", StringComparison.Ordinal)))}}
                    return new object[] { trained, file };
                }
            }
            """;
        var references = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!)
            .Split(Path.PathSeparator)
            .Select(path => MetadataReference.CreateFromFile(path));
        var compilation = CSharpCompilation.Create(
            "ReadmeExample",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Enable));

        using var image = new MemoryStream();
        var emitted = compilation.Emit(image);

        Assert.True(emitted.Success, string.Join('\n', emitted.Diagnostics.Where(each => each.Severity == DiagnosticSeverity.Error)));

        image.Position = 0;
        var loaded = new AssemblyLoadContext("readme", isCollectible: true).LoadFromStream(image);

        return loaded.GetType("Example")!.GetMethod("Run")!.Invoke(null, null);
    }
}

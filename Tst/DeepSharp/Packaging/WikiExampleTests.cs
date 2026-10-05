// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.Loader;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// The wiki's code is what a person copies first after the README's, so every C# block on every page is compiled against
/// the packages as built here, with the usings a console program has without asking, and fails on a warning as the build
/// does — an obsolete form included. A block is a program of its own unless the line above it says otherwise, in a comment
/// the page does not show: <c>&lt;!-- example: name --&gt;</c> names a block others go on from, <c>&lt;!-- continues: name
/// --&gt;</c> is compiled after that block and every block that went on from it before, and <c>&lt;!-- a C# cell of a
/// notebook --&gt;</c> is a notebook's cell, which the notebook's suite runs in Verso's own engine. No block is passed over
/// without one of those, and a comment the test does not know is refused.
/// </summary>
public sealed class WikiExampleTests
{
    [Fact]
    public void EveryCSharpBlockOfTheWiki_CompilesAgainstThePackagesAsBuilt_WithoutAWarning()
    {
        var pages = Wiki.Pages();
        var compiled = 0;
        var faults = new List<string>();

        foreach (var page in pages)
        {
            foreach (var program in page.Programs)
            {
                var diagnostics = program.Compiled().Diagnostics;

                faults.AddRange(diagnostics.Select(diagnostic => $"{page.Name}, the block at line {program.Line}: {diagnostic}"));
                compiled++;
            }
        }

        Assert.True(faults.Count == 0, string.Join('\n', faults));
        Assert.True(compiled >= 20, $"The wiki's pages hold {compiled} blocks that compile as programs: the wiki this reads is not the one the pages are.");
        Assert.Contains(pages, page => page.Cells.Count > 0);
    }

    [Fact]
    public void TheNetworkTheNetworksPageWritesAsCode_TrainsBehindThePipeline_AndIsSavedAndReadBack()
    {
        // The page's network written as code is saved as its one file and read back with the catalog it is registered with,
        // as the page says, and answers what it answered before it was saved.
        var page = Wiki.Pages().Single(each => each.Name == "Networks.md");
        var program = page.Programs.Single(each => each.Blocks.Any(block => block.Code.Contains(": Network,", StringComparison.Ordinal)));
        var compiled = program.Compiled();
        var context = new AssemblyLoadContext("wiki", isCollectible: true);

        Assert.Empty(compiled.Diagnostics);

        try
        {
            using var image = new MemoryStream(compiled.Image);
            var written = context.LoadFromStream(image).GetTypes().Single(type => type.IsSubclassOf(typeof(Network)));
            var network = (Network)Activator.CreateInstance(written, new RandomStream(42).Draw("initialise:wiki", 0, 0))!;
            var prepared = WikiTitanic.In(WikiTitanic.DataFolder).Run();
            var trained = network.Compile(new Adam(0.01), new BinaryCrossEntropy()).Fit(prepared, new FitOptions(seed: 7) { Epochs = 2 });
            var catalog = NetworkCatalog.BuiltIn();

            typeof(NetworkCatalog).GetMethod(nameof(NetworkCatalog.Register))!.MakeGenericMethod(written).Invoke(catalog, null);

            var read = TrainedNetwork.FromJson(trained.ToJson(), catalog, StepCatalog.BuiltIn());
            var passenger = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"]]);

            Assert.IsType(written, read.Network);
            Assert.Equal(trained.Predict(passenger).Answers, read.Predict(passenger).Answers);
        }
        finally
        {
            context.Unload();
        }
    }

    [Fact]
    public void ACommentTheTestDoesNotKnow_OrAnExampleNobodyNamed_IsRefused()
    {
        Assert.Contains("does not know", Assert.Throws<InvalidOperationException>(() => Wiki.Read("Page.md", "<!-- skip -->\n```csharp\nvar a = 1;\n```\n")).Message, StringComparison.Ordinal);
        Assert.Contains("no example", Assert.Throws<InvalidOperationException>(() => Wiki.Read("Page.md", "<!-- continues: none -->\n```csharp\nvar a = 1;\n```\n")).Message, StringComparison.Ordinal);
        Assert.Contains("twice", Assert.Throws<InvalidOperationException>(() => Wiki.Read("Page.md", "<!-- example: one -->\n```csharp\nvar a = 1;\n```\n<!-- example: one -->\n```csharp\nvar b = 1;\n```\n")).Message, StringComparison.Ordinal);

        var read = Wiki.Read("Page.md", "<!-- example: one -->\n```csharp\nvar a = 1;\n```\n\n<!-- continues: one -->\n```csharp\nvar b = a;\n```\n\n<!-- a C# cell of a notebook -->\n```csharp\n#r \"nuget: X\"\n```\n");

        Assert.Equal([1, 2], read.Programs.Select(program => program.Blocks.Count));
        Assert.Single(read.Cells);
    }
}

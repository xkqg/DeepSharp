// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// The pages of the tutorial are the code a person copies first after the README's, and a block that compiles can still
/// stop where it runs: one of them asked a model for a column of dates, and another read a fit by a place its steps no
/// longer held, and both compiled for as long as they were wrong. So the pipeline on each page is run, over the published
/// data copied into a folder of its own, and fails on whatever it throws.
/// </summary>
public sealed partial class TutorialExampleTests : IDisposable
{
    private static readonly string[] Data = ["titanic.csv", "apple.csv", "bikes.csv"];

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-tutorial-").FullName;

    public TutorialExampleTests()
    {
        foreach (var file in Data)
        {
            File.Copy(Repository.Data(file), Path.Join(_folder, file));
        }
    }

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [GeneratedRegex("\"(?<file>[A-Za-z0-9._-]+\\.(?:csv|html|svg|json))\"")]
    private static partial Regex FileName();

    // The program with every file it names — the data it reads, the pages and pictures it writes — in the folder of this test,
    // and what it prints written nowhere: it is run to see that it runs, and a runner's log is not its page.
    private WikiProgram InTheFolder(WikiProgram program) => program with
    {
        Blocks = [.. program.Blocks.Select(block => block with { Code = FileName().Replace(block.Code, name => JsonSerializer.Serialize(Path.Join(_folder, name.Groups["file"].Value))).Replace("Console.WriteLine(", "TextWriter.Null.WriteLine(", StringComparison.Ordinal) })],
    };

    [Fact]
    public async Task EveryPipelineOfTheTutorial_RunsOverThePublishedData_WithoutAFailure()
    {
        var faults = new List<string>();
        var ran = 0;

        foreach (var (page, program) in Wiki.Runnable())
        {
            var compiled = InTheFolder(program).Compiled();
            var context = new AssemblyLoadContext("tutorial", isCollectible: true);

            ran++;

            try
            {
                using var image = new MemoryStream(compiled.Image);
                var entry = context.LoadFromStream(image).EntryPoint!;

                if (entry.Invoke(null, entry.GetParameters().Length == 0 ? null : [Array.Empty<string>()]) is Task running)
                {
                    await running;
                }
            }
            catch (TargetInvocationException failed)
            {
                faults.Add($"{page}, the block at line {program.Line}: {failed.InnerException?.GetType().Name}: {failed.InnerException?.Message}");
            }
            finally
            {
                context.Unload();
            }
        }

        Assert.True(faults.Count == 0, string.Join('\n', faults));
        Assert.True(ran >= 10, $"The tutorial holds {ran} pipelines that run: what this runs is not what a person reads.");
    }
}

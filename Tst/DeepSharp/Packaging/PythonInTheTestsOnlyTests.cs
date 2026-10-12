// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// Where Python may stand in this repository, and where it may not. PyTorch, Keras and TensorFlow are what the arithmetic
/// here is held to and what writes the files the readers read, so a script of them belongs beside the tests it serves: it
/// runs once, by hand, and what it leaves behind — the numbers and the files — is what every build reads. Nothing that
/// ships may hold such a script or start one, which is what these tests hold the repository to rather than a habit.
/// </summary>
/// <remarks>
/// The boundary is the owner's, stated plainly: Python for the tests is right, Python in production is not. It is cheap to
/// keep and quiet to break — a single <c>Process.Start("python", …)</c> in a package would turn a test-time reference into a
/// thing every application of this library would have to install.
/// </remarks>
public sealed class PythonInTheTestsOnlyTests
{
    private static readonly string[] Skipped = ["bin", "obj", ".git", ".vs", "nupkgs", "out", "node_modules"];

    [Fact]
    public void EveryPythonScript_StandsBesideTheTestsItServes_OrAmongTheTools()
    {
        // A writer of fixtures lives in the Fixtures folder of the suite that reads what it wrote; the icon's own script is a
        // tool of the repository. Anywhere else is a script nobody runs by hand and everybody has to carry.
        var scripts = Scripts().Select(Relative).Order(StringComparer.Ordinal).ToArray();

        Assert.NotEmpty(scripts);
        Assert.All(
            scripts,
            script => Assert.True(
                script.StartsWith("tools/", StringComparison.Ordinal) ||
                (script.StartsWith("Tst/", StringComparison.Ordinal) && script.Contains("/Fixtures/", StringComparison.Ordinal)),
                $"'{script}' is a Python script outside a suite's Fixtures folder and outside tools."));
    }

    [Fact]
    public void NothingThatShips_HoldsAPythonScript_OrNamesPythonInAnythingItRuns()
    {
        // The prose of a reader may say what Python does with a pickle — that is what it reads. What no shipped code may do is
        // name python in a value: a path, a file name, a program to start.
        var named = new List<string>();

        foreach (var source in Directory.EnumerateFiles(Path.Join(Repository.Root, "Src"), "*.cs", SearchOption.AllDirectories).Where(Kept))
        {
            var literals = CSharpSyntaxTree.ParseText(File.ReadAllText(source), cancellationToken: TestContext.Current.CancellationToken)
                .GetRoot(TestContext.Current.CancellationToken)
                .DescendantTokens()
                .Where(token => token.IsKind(SyntaxKind.StringLiteralToken) || token.IsKind(SyntaxKind.SingleLineRawStringLiteralToken))
                .Select(token => token.ValueText)
                .Where(text => text.Contains("python", StringComparison.OrdinalIgnoreCase) || text.EndsWith(".py", StringComparison.OrdinalIgnoreCase));

            named.AddRange(literals.Select(text => $"{Relative(source)}: '{text}'"));
        }

        Assert.Empty(named);
        Assert.DoesNotContain(Scripts().Select(Relative), script => script.StartsWith("Src/", StringComparison.Ordinal));
    }

    [Fact]
    public void NoPackageThePackMakes_CarriesAPythonScript()
    {
        // A script packed as content would travel into every application: the projects say what they pack, and none of them
        // says a Python file.
        var projects = Directory.EnumerateFiles(Path.Join(Repository.Root, "Src"), "*.csproj", SearchOption.AllDirectories).Where(Kept).ToArray();

        Assert.NotEmpty(projects);
        Assert.DoesNotContain(projects, project => File.ReadAllText(project).Contains(".py\"", StringComparison.OrdinalIgnoreCase));
    }

    // Every Python script of the repository, the folders a build writes left out.
    private static IEnumerable<string> Scripts() =>
        Directory.EnumerateFiles(Repository.Root, "*.py", SearchOption.AllDirectories).Where(Kept);

    // Whether a path is the repository's own rather than something a build or a tool wrote under it.
    private static bool Kept(string path) =>
        !Relative(path).Split('/').Any(part => Skipped.Contains(part, StringComparer.OrdinalIgnoreCase));

    private static string Relative(string path) =>
        Path.GetRelativePath(Repository.Root, path).Replace('\\', '/');
}

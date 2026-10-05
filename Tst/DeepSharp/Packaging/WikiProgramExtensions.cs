// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// A program of the wiki, compiled as a person's project would compile it: its blocks put together as one, against the
/// packages as built here, with the usings a console program has without asking.
/// </summary>
internal static class WikiProgramExtensions
{
    // What a console project has without a using of its own: the SDK's implicit usings.
    private const string ImplicitUsings = """
        global using System;
        global using System.Collections.Generic;
        global using System.IO;
        global using System.Linq;
        global using System.Net.Http;
        global using System.Threading;
        global using System.Threading.Tasks;
        """;

    // What the .NET SDK leaves unsaid in every project it builds: that one version of an assembly is taken for another the
    // runtime unifies it with — on .NET 8, System.Text.Json 8 as referenced beside the 10 a package brings.
    private static readonly Dictionary<string, ReportDiagnostic> SdkQuiets = new()
    {
        ["CS1701"] = ReportDiagnostic.Suppress,
        ["CS1702"] = ReportDiagnostic.Suppress,
    };

    extension(WikiProgram program)
    {
        /// <summary>
        /// The program compiled: its blocks put together as one — every block's usings first, then every block's statements
        /// in the order they stand, then every type they declare — against every assembly this suite runs with, the
        /// packages as built among them.
        /// </summary>
        /// <returns>Every warning and error, and the image.</returns>
        public CompiledProgram Compiled()
        {
            var units = program.Blocks.Select(block => CSharpSyntaxTree.ParseText(block.Code).GetCompilationUnitRoot()).ToArray();
            var statements = units.SelectMany(unit => unit.Members.OfType<GlobalStatementSyntax>()).ToArray();
            var source = string.Join(
                '\n',
                [
                    .. units.SelectMany(unit => unit.Usings).Select(each => each.ToString()).Distinct(StringComparer.Ordinal),
                    .. statements.Select(statement => statement.ToFullString()),
                    .. units.SelectMany(unit => unit.Members.Where(member => member is not GlobalStatementSyntax)).Select(member => member.ToFullString()),
                ]);
            var references = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!)
                .Split(Path.PathSeparator)
                .Select(path => MetadataReference.CreateFromFile(path));
            var compilation = CSharpCompilation.Create(
                $"Wiki{program.Line}",
                [CSharpSyntaxTree.ParseText(ImplicitUsings), CSharpSyntaxTree.ParseText(source)],
                references,
                new CSharpCompilationOptions(
                        statements.Length > 0 ? OutputKind.ConsoleApplication : OutputKind.DynamicallyLinkedLibrary,
                        nullableContextOptions: NullableContextOptions.Enable)
                    .WithSpecificDiagnosticOptions(SdkQuiets));

            using var image = new MemoryStream();
            var emitted = compilation.Emit(image);

            return new CompiledProgram(
                [.. emitted.Diagnostics.Where(each => each.Severity >= DiagnosticSeverity.Warning).Select(each => each.ToString())],
                image.ToArray());
        }
    }
}

/// <summary>A compiled program: every warning and error, and the image.</summary>
/// <param name="Diagnostics">Every warning and error.</param>
/// <param name="Image">The assembly, as it is emitted.</param>
internal readonly record struct CompiledProgram(IReadOnlyList<string> Diagnostics, byte[] Image);

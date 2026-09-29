// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The notebook package is installed by Verso itself, into a folder of its own, beside whatever else a person
/// has installed. What it may carry and which Verso it runs in are therefore facts about the package, and each
/// is held by a test rather than remembered: the abstractions it is compiled against decide the oldest Verso
/// that loads it, and every reference it takes is copied in with it.
/// </summary>
public class PackageTests
{
    private static readonly Assembly Package = typeof(StepCellType).Assembly;

    private static XDocument Project() =>
        XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Verso.Notebooks", "DeepSharp.Verso.Notebooks.csproj"));

    [Fact]
    public void ThePackageIsCompiledAgainstTheOldestAbstractionsThatCarryWhatItCalls()
    {
        // Verso refuses an extension compiled against a newer minor than the one it runs. Compiled against
        // 1.1.0 it loads in every Verso from 1.1.0 on; a restore that quietly took a newer version would lock
        // out everyone still on an older one, and nothing else would notice.
        var abstractions = Package.GetReferencedAssemblies().Single(reference => reference.Name == "Verso.Abstractions");

        Assert.Equal(new Version(1, 1, 0, 0), abstractions.Version);
    }

    [Fact]
    public void ThePackageReferencesTheRuntimeAndAClosedListOfPackages()
    {
        // Everything the package references is installed with it into Verso's folder for it. The list is closed
        // so that a reference added for convenience is a decision somebody makes, not a surprise in every install.
        string[] allowed = ["DeepSharp.Pipelines", "DeepSharp.Pipelines.Indicators", "MatPlotLibNet", "Verso.Abstractions"];
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        var outside = Package.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(allowed)
            .ToArray();

        Assert.True(outside.Length == 0, $"The notebook package references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void NeitherTheNotebookPackageNorItsHost_TakesALock()
    {
        // What one change makes whole is a value, handed on whole; who may change it is its owner — never a lock around
        // state, which makes the collection safe and the fact it holds unsafe. Read off the compiled code, every method a
        // lock or a semaphore would sit in included, so the rule holds without anybody remembering it.
        string[] blocking = ["Monitor", "Lock", "SemaphoreSlim", "Semaphore", "Mutex", "ReaderWriterLockSlim", "ReaderWriterLock", "SpinLock"];

        var taken = new[] { Package, typeof(DeepSharp.Verso.Api.NotebookHost).Assembly }
            .SelectMany(assembly => assembly.GetTypes())
            .SelectMany(type => type.GetMethods(Every).Cast<MethodBase>().Concat(type.GetConstructors(Every)))
            .SelectMany(method => Called(method).Select(called => (Method: method, Called: called)))
            .Where(each => each.Called.DeclaringType is { Namespace: "System.Threading" } owner && blocking.Contains(owner.Name))
            .Select(each => $"{each.Method.DeclaringType!.Name}.{each.Method.Name} → {each.Called.DeclaringType!.Name}.{each.Called.Name}")
            .Distinct()
            .ToArray();

        Assert.True(taken.Length == 0, $"Taken: {string.Join("; ", taken)}");
    }

    private const BindingFlags Every = BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    private static readonly Dictionary<short, System.Reflection.Emit.OpCode> OpCodes = typeof(System.Reflection.Emit.OpCodes)
        .GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(field => (System.Reflection.Emit.OpCode)field.GetValue(null)!)
        .ToDictionary(code => code.Value);

    // Every method a body calls or constructs, read off its IL one instruction at a time.
    private static IEnumerable<MethodBase> Called(MethodBase method)
    {
        if (method.GetMethodBody()?.GetILAsByteArray() is not { } il)
        {
            yield break;
        }

        var typeArguments = method.DeclaringType!.IsGenericType ? method.DeclaringType.GetGenericArguments() : null;
        var methodArguments = method.IsGenericMethod ? method.GetGenericArguments() : null;

        for (var at = 0; at < il.Length;)
        {
            var code = il[at] == 0xFE ? OpCodes[BitConverter.ToInt16([il[at + 1], 0xFE])] : OpCodes[il[at]];
            at += code.Size;

            if (code.OperandType == System.Reflection.Emit.OperandType.InlineMethod)
            {
                yield return method.Module.ResolveMethod(BitConverter.ToInt32(il, at), typeArguments, methodArguments)!;
            }

            at += code.OperandType switch
            {
                System.Reflection.Emit.OperandType.InlineNone => 0,
                System.Reflection.Emit.OperandType.ShortInlineBrTarget or System.Reflection.Emit.OperandType.ShortInlineI or System.Reflection.Emit.OperandType.ShortInlineVar => 1,
                System.Reflection.Emit.OperandType.InlineVar => 2,
                System.Reflection.Emit.OperandType.InlineI8 or System.Reflection.Emit.OperandType.InlineR => 8,
                System.Reflection.Emit.OperandType.InlineSwitch => 4 + 4 * BitConverter.ToInt32(il, at),
                _ => 4,
            };
        }
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags_AndTakesTheAbstractionsAsARange()
    {
        var project = Project();
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');
        var abstractions = project.Descendants("PackageReference")
            .Single(reference => reference.Attribute("Include")!.Value == "Verso.Abstractions");

        Assert.Equal("DeepSharp.Verso.Notebooks", project.Descendants("PackageId").Single().Value);
        Assert.Contains("verso", tags);
        Assert.Contains("notebook", tags);
        Assert.Contains("pdd", tags);
        Assert.Equal("[1.1.0, 2.0.0)", abstractions.Attribute("Version")!.Value);
    }

    [Fact]
    public void DependabotLeavesTheAbstractionsAtTheirFloor()
    {
        // Raising the floor decides which Verso the package still loads in, and the tests above refuse it. Left to
        // itself, Dependabot proposes it for every version Verso publishes, as a pull request that can only fail.
        var lines = File.ReadLines(Path.Join(Repository.Root, ".github", "dependabot.yml"))
            .Select(line => line.Trim())
            .Where(line => !line.StartsWith('#'));

        Assert.Contains("- dependency-name: \"Verso.Abstractions\"", lines);
    }
}

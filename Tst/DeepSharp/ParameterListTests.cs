// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Runtime.CompilerServices;

namespace DeepSharp.Tests;

/// <summary>
/// No method and no constructor in a package takes more than four parameters, as CONTRIBUTING says: a fifth says some
/// of them belong together. The rule is read from the assemblies themselves, public members and private ones alike,
/// so it holds for code nobody remembered to look at.
/// </summary>
/// <remarks>
/// <para>
/// This one file is compiled into every suite, and each suite reads the packages it references: a suite cannot load
/// what it does not reference, and the notebook's packages are built against another compiler than the core suite's.
/// Every package in <c>Src</c> must be read by one of them, so a package added later is held to the rule the day it
/// appears.
/// </para>
/// <para>
/// What is not counted, and why: whatever the compiler writes on its own (a lambda, a local function, an iterator, the
/// members a record is given, such as its <c>Deconstruct</c>), whatever the runtime implements (a delegate's
/// <c>Invoke</c>, <c>BeginInvoke</c> and <c>EndInvoke</c>), and the primary constructor of a positional record, which
/// is the record's list of named members rather than a list of arguments.
/// </para>
/// </remarks>
public class ParameterListTests
{
    private const int Most = 4;

    private const BindingFlags Declared =
        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

    /// <summary>Which suite reads each package: the one that references it.</summary>
    private static readonly Dictionary<string, string> ReadBy = new()
    {
        ["DeepSharp"] = "DeepSharp.Tests",
        ["DeepSharp.Backends.TorchSharp"] = "DeepSharp.Backends.TorchSharp.Tests",
        ["DeepSharp.Charts"] = "DeepSharp.Tests",
        ["DeepSharp.Import.Keras"] = "DeepSharp.Tests",
        ["DeepSharp.Import.Onnx"] = "DeepSharp.Tests",
        ["DeepSharp.Import.PyTorch"] = "DeepSharp.Tests",
        ["DeepSharp.Learners.ML"] = "DeepSharp.Tests",
        ["DeepSharp.Learners.MLNet"] = "DeepSharp.Learners.MLNet.Tests",
        ["DeepSharp.Learners.Networks"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines.Binance"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines.DataFrame"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines.Excel"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines.Indicators"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines.Json"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines.Parquet"] = "DeepSharp.Tests",
        ["DeepSharp.Verso.Notebooks"] = "DeepSharp.Verso.Notebooks.Tests",
        ["DeepSharp.Verso.Api"] = "DeepSharp.Verso.Api.Tests",
        ["DeepSharp.Verso.Serve"] = "DeepSharp.Verso.Serve.Tests",
    };

    /// <summary>
    /// The only members allowed more than four: forms an earlier release published, kept so that code written against
    /// them still compiles, and marked obsolete in favour of the shorter form beside them.
    /// </summary>
    private static readonly string[] Kept =
    [
        "DeepSharp.Pipelines: DeepSharp.Pipelines.PreparedData..ctor(PipelineDeclaration, Table, IReadOnlyList<Part>, IReadOnlyDictionary<Int32, FittedStepValues>, IReadOnlyDictionary<Int32, Evidence>)",
        "DeepSharp.Pipelines: DeepSharp.Pipelines.ColumnParameter..ctor(String, String, String, IReadOnlyList<ColumnKind>, Boolean)",
        "DeepSharp.Pipelines: DeepSharp.Pipelines.ColumnsParameter..ctor(String, String, IReadOnlyList<String>, IReadOnlyList<ColumnKind>, Boolean, Boolean)",
        "DeepSharp.Pipelines: DeepSharp.Pipelines.NumberParameter..ctor(String, String, Double, Nullable<Double>, Nullable<Double>)",
        "DeepSharp.Pipelines: DeepSharp.Pipelines.WholeNumberParameter..ctor(String, String, Int32, Nullable<Int32>, Nullable<Int32>)",
        "DeepSharp.Pipelines: DeepSharp.Pipelines.FillStrategyParameter..ctor(String, String, FillStrategy, IReadOnlyList<String>, String)",
        "DeepSharp.Pipelines: DeepSharp.Pipelines.SchemaBuilder.Column(String, ColumnKind, Boolean, String, String)",
        "DeepSharp.Pipelines.Indicators: DeepSharp.Pipelines.IndicatorExtensions.AddIndicator(PipelineBuilder, String, Indicator, String[], Int32)",
    ];

    private static IEnumerable<Assembly> ReadHere()
    {
        var suite = typeof(ParameterListTests).Assembly.GetName().Name;

        return ReadBy.Where(package => package.Value == suite).Select(package => Assembly.Load(package.Key));
    }

    [Fact]
    public void NoMethodOrConstructorTakesMoreThanFourParameters()
    {
        var read = ReadHere().ToArray();
        var over = read.SelectMany(Over).Except(Kept).ToArray();

        Assert.NotEmpty(read);

        if (over.Length > 0)
        {
            Assert.Fail($"{over.Length} members take more than {Most} parameters:{Environment.NewLine}" +
                        string.Join(Environment.NewLine, over));
        }
    }

    [Fact]
    public void EachKeptFormIsObsoleteAndStillThere()
    {
        // An exception that no longer names a member, or names one that is not marked, would let the list grow
        // quietly; every entry is a published member that says which form to call instead.
        foreach (var assembly in ReadHere())
        {
            var name = assembly.GetName().Name;
            var over = Members(assembly).Where(member => member.GetParameters().Length > Most).ToArray();

            foreach (var kept in Kept.Where(entry => entry.StartsWith($"{name}: ", StringComparison.Ordinal)))
            {
                var member = Assert.Single(over, member => Listed(member) == kept);
                Assert.True(member.IsDefined(typeof(ObsoleteAttribute), false), $"{kept} is kept but not obsolete.");
                Assert.True(member.IsPublic || member.IsFamily || member.IsFamilyOrAssembly, $"{kept} was never published.");
            }
        }
    }

    [Fact]
    public void EveryPackageIsReadBySomeSuite()
    {
        var packages = Directory.GetDirectories(Path.Join(Repository.Root, "Src"))
            .Where(folder => File.Exists(Path.Join(folder, $"{Path.GetFileName(folder)}.csproj")))
            .Select(Path.GetFileName)
            .Order(StringComparer.Ordinal);

        Assert.Equal(packages, ReadBy.Keys.Order(StringComparer.Ordinal));
    }

    private static IEnumerable<string> Over(Assembly assembly) =>
        Members(assembly)
            .Where(member => member.GetParameters().Length > Most)
            .Where(member => !member.IsDefined(typeof(CompilerGeneratedAttribute), false))
            .Where(member => (member.MethodImplementationFlags & MethodImplAttributes.Runtime) == 0)
            .Where(member => !IsPositionalRecordConstructor(member))
            .Select(Listed);

    private static IEnumerable<MethodBase> Members(Assembly assembly) =>
        assembly.GetTypes()
            .Where(type => !WrittenByTheCompiler(type))
            .SelectMany(type => type.GetConstructors(Declared).Cast<MethodBase>().Concat(type.GetMethods(Declared)));

    private static bool WrittenByTheCompiler(Type type)
    {
        for (var at = type; at is not null; at = at.DeclaringType)
        {
            if (at.Name.Contains('<', StringComparison.Ordinal) || at.IsDefined(typeof(CompilerGeneratedAttribute), false))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// A constructor whose parameters the compiler's own <c>Deconstruct</c> hands back, name for name and type for
    /// type. Only a positional record is given that <c>Deconstruct</c>, and only for its primary constructor.
    /// </summary>
    private static bool IsPositionalRecordConstructor(MethodBase member)
    {
        if (member is not ConstructorInfo)
        {
            return false;
        }

        var parameters = member.GetParameters();

        return member.DeclaringType!.GetMethods(Declared)
            .Where(method => method.Name == "Deconstruct" && method.IsDefined(typeof(CompilerGeneratedAttribute), false))
            .Select(method => method.GetParameters())
            .Any(outs => outs.Select(parameter => parameter.Name).SequenceEqual(parameters.Select(parameter => parameter.Name)) &&
                         outs.Select(parameter => parameter.ParameterType.GetElementType())
                             .SequenceEqual(parameters.Select(parameter => parameter.ParameterType)));
    }

    private static string Listed(MethodBase member) =>
        $"{member.DeclaringType!.Assembly.GetName().Name}: {member.DeclaringType.FullName!.Replace('+', '.')}.{member.Name}" +
        $"({string.Join(", ", member.GetParameters().Select(parameter => Named(parameter.ParameterType)))})";

    private static string Named(Type type)
    {
        if (type.HasElementType)
        {
            var element = Named(type.GetElementType()!);

            return type.IsArray ? $"{element}[]" : type.IsByRef ? $"ref {element}" : element;
        }

        if (!type.IsGenericType)
        {
            return type.Name;
        }

        var name = type.Name[..type.Name.IndexOf('`', StringComparison.Ordinal)];

        return $"{name}<{string.Join(", ", type.GetGenericArguments().Select(Named))}>";
    }
}

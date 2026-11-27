// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Runtime.CompilerServices;

namespace DeepSharp.Tests;

/// <summary>
/// No method and no constructor in a package takes more than four parameters, as CONTRIBUTING says: a fifth says some
/// of them belong together. The rule is read from the assemblies themselves, public members and private ones alike,
/// so it holds for code nobody remembered to look at. The same reading holds two more of CONTRIBUTING's rules about the
/// shape of a member: several values out of a method are a record struct and never a tuple, and no type is named for a
/// drawer nobody owns (a helper, a utility, a manager).
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
        ["DeepSharp.Export.Onnx"] = "DeepSharp.Export.Onnx.Tests",
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
        ["DeepSharp.Pipelines.Images"] = "DeepSharp.Tests",
        ["DeepSharp.Pipelines.Sound"] = "DeepSharp.Tests",
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

    [Fact]
    public void NoMemberHandsBackOrTakesATuple()
    {
        var read = ReadHere().ToArray();
        var tuples = read.SelectMany(TupleMembers).Except(PublishedTuples).Except(TuplesUntilTheirStepRemovesThem).ToArray();

        Assert.NotEmpty(read);

        if (tuples.Length > 0)
        {
            Assert.Fail($"{tuples.Length} members return, take or hold a tuple, where CONTRIBUTING asks for a readonly record struct:{Environment.NewLine}" +
                        string.Join(Environment.NewLine, tuples));
        }
    }

    [Fact]
    public void EachListedTupleFormIsStillThere_AndAPublishedOneIsObsolete()
    {
        // A name that no longer matches a member would let the lists outlive what they excuse.
        foreach (var assembly in ReadHere())
        {
            var name = assembly.GetName().Name;
            var members = TupleMemberInfos(assembly).ToArray();

            foreach (var listed in PublishedTuples.Concat(TuplesUntilTheirStepRemovesThem).Where(entry => entry.StartsWith($"{name}: ", StringComparison.Ordinal)))
            {
                var found = members.Where(member => ListedTuple(member) == listed).ToArray();

                Assert.NotEmpty(found);
            }

            foreach (var listed in PublishedTuples.Where(entry => entry.StartsWith($"{name}: ", StringComparison.Ordinal)))
            {
                var member = members.First(member => ListedTuple(member) == listed);

                Assert.True(member.IsDefined(typeof(ObsoleteAttribute), false), $"{listed} is kept but not obsolete.");
            }
        }
    }

    [Fact]
    public void NoTypeIsNamedForADrawerNobodyOwns()
    {
        var read = ReadHere().ToArray();
        var drawers = read.SelectMany(assembly => assembly.GetTypes())
            .Where(type => !WrittenByTheCompiler(type))
            .Where(type => Drawers.Any(drawer => type.Name.EndsWith(drawer, StringComparison.Ordinal)))
            .Select(type => $"{type.Assembly.GetName().Name}: {type.FullName}")
            .ToArray();

        Assert.NotEmpty(read);

        if (drawers.Length > 0)
        {
            Assert.Fail($"{drawers.Length} types are named for a drawer, which CONTRIBUTING refuses:{Environment.NewLine}" + string.Join(Environment.NewLine, drawers));
        }
    }

    private static readonly string[] Drawers = ["Helper", "Helpers", "Util", "Utils", "Utility", "Utilities", "Manager"];

    [Fact]
    public void NoPackageSpellsTheWordAnEnumIsWrittenAsByHand_ButTheTensorCore()
    {
        // The word a pipeline file writes a choice as has one owner, the Word() of DeepSharp.Pipelines: its reader, its writer, a
        // message and a notebook all name the choice the same way. Lower-casing the name in place is a second owner that drifts
        // the day a value is renamed. The tensor core cannot reach that owner, since it stands under it, and keeps its own.
        const string ByHand = "ToString().ToLowerInvariant()";

        var read = ReadHere().Select(assembly => assembly.GetName().Name!).Where(name => name != "DeepSharp").ToArray();
        var spelled = read
            .SelectMany(name => Directory.EnumerateFiles(Path.Join(Repository.Root, "Src", name), "*.cs", SearchOption.AllDirectories))
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .SelectMany(file => File.ReadAllLines(file)
                .Select((line, at) => line.Contains(ByHand, StringComparison.Ordinal)
                    ? $"{Path.GetRelativePath(Repository.Root, file).Replace('\\', '/')}:{at + 1}"
                    : null)
                .OfType<string>())
            .ToArray();

        Assert.NotEmpty(read);

        if (spelled.Length > 0)
        {
            Assert.Fail($"{spelled.Length} places write an enum's word by hand, where its Word() is the one owner:{Environment.NewLine}" +
                        string.Join(Environment.NewLine, spelled));
        }
    }

    [Fact]
    public void NoPackageDecidesWhichRowsLearnByHand_ButThePartFacts()
    {
        // Which rows a fit learns from, which rows a split deals again, and which parts a fit counts are facts about a Part, and
        // they have one owner, PartFacts, beside Part: a fit that wrote them in place would drift the day a part is added. A
        // split WRITES parts and is left alone; only a read of them is held here.
        string[] byHand = ["== Part.Train", "!= Part.Train", "Part.Train or Part.Validation", "new[] { Part."];
        const string Owner = "static class PartFacts";

        var read = ReadHere().Select(assembly => assembly.GetName().Name!).Where(name => name != "DeepSharp").ToArray();
        var spelled = read
            .SelectMany(name => Directory.EnumerateFiles(Path.Join(Repository.Root, "Src", name), "*.cs", SearchOption.AllDirectories))
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .SelectMany(file => TheOwnerLeftOut(File.ReadAllLines(file), Owner)
                .Where(line => byHand.Any(form => line.Text.Contains(form, StringComparison.Ordinal)))
                .Select(line => $"{Path.GetRelativePath(Repository.Root, file).Replace('\\', '/')}:{line.Number}"))
            .ToArray();

        Assert.NotEmpty(read);

        if (spelled.Length > 0)
        {
            Assert.Fail($"{spelled.Length} places decide by hand which rows learn or which parts a fit counts, where PartFacts is the one owner:{Environment.NewLine}" +
                        string.Join(Environment.NewLine, spelled));
        }
    }

    [Fact]
    public void NoPackageFindsTheSchemaTheFirstSplitOrAnOnceOnlyListByHand_ButTheDeclarationAndItsStepPositions()
    {
        // Which step is the first of its kind, and which kinds a declaration holds one of, have one owner each: the steps' own
        // positions, the declaration's schema and the rules that limit a kind. A rule that looked for the schema or the split
        // itself would drift the day another step stood in for it, and a list of the limited kinds kept beside the rules would
        // be a second copy to forget.
        string[] byHand = ["OfType<DeclareStep>()", "FirstSplit(", "PipelineDeclaration.FaultsIn(", "OnceOnly"];

        var read = ReadHere().Select(assembly => assembly.GetName().Name!).Where(name => name != "DeepSharp").ToArray();
        var spelled = read
            .SelectMany(name => Directory.EnumerateFiles(Path.Join(Repository.Root, "Src", name), "*.cs", SearchOption.AllDirectories))
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .SelectMany(file => File.ReadAllLines(file)
                .Select((line, at) => byHand.Any(form => line.Contains(form, StringComparison.Ordinal))
                    ? $"{Path.GetRelativePath(Repository.Root, file).Replace('\\', '/')}:{at + 1}"
                    : null)
                .OfType<string>())
            .ToArray();

        Assert.NotEmpty(read);

        if (spelled.Length > 0)
        {
            Assert.Fail($"{spelled.Length} places find the schema or the first split by hand, or decide whether steps are a declaration without PipelineDeclaration.Try:{Environment.NewLine}" +
                        string.Join(Environment.NewLine, spelled));
        }
    }

    // The lines of a file, numbered, without the class that owns the facts: it runs from its declaration to the closing brace at the margin.
    private static IEnumerable<NumberedLine> TheOwnerLeftOut(string[] lines, string owner)
    {
        var inside = false;

        for (var at = 0; at < lines.Length; at++)
        {
            inside |= lines[at].Contains(owner, StringComparison.Ordinal);

            if (!inside)
            {
                yield return new NumberedLine(lines[at], at + 1);
            }
            else if (lines[at] == "}")
            {
                inside = false;
            }
        }
    }

    private readonly record struct NumberedLine(string Text, int Number);

    /// <summary>
    /// The only tuple form a package may keep: one an earlier release published, marked obsolete in favour of the record
    /// struct beside it, so that code written against it still compiles.
    /// </summary>
    private static readonly string[] PublishedTuples =
    [
        "DeepSharp: DeepSharp.Networks.TransposedConvolution.Settling(Sides, Int32)",
    ];

    /// <summary>
    /// A tuple that is on its way out in a later step of the same release, named here until the step that removes it; none
    /// is left, since the study loop the last one belonged to hands its trial and what it trained over as a record struct.
    /// </summary>
    private static readonly string[] TuplesUntilTheirStepRemovesThem = [];

    private static IEnumerable<string> TupleMembers(Assembly assembly) => TupleMemberInfos(assembly).Select(ListedTuple);

    private static IEnumerable<MemberInfo> TupleMemberInfos(Assembly assembly) =>
        assembly.GetTypes()
            .Where(type => !WrittenByTheCompiler(type))
            .SelectMany(type => type.GetConstructors(Declared).Cast<MemberInfo>()
                .Concat(type.GetMethods(Declared))
                .Concat(type.GetFields(Declared)))
            .Where(member => !member.Name.Contains('<', StringComparison.Ordinal) && !member.IsDefined(typeof(CompilerGeneratedAttribute), false))
            .Where(member => TypesOf(member).Any(IsOrHoldsATuple))
            .Where(member => member is not MethodInfo method || !IsSetByAnotherLibrary(method));

    /// <summary>
    /// A member whose shape a library outside this repository dictates: it implements a member of that library's interface,
    /// or overrides one of its methods. A package cannot hand back anything but what the member there is declared to.
    /// </summary>
    private static bool IsSetByAnotherLibrary(MethodInfo method)
    {
        var type = method.DeclaringType!;

        if (method.GetBaseDefinition().DeclaringType!.Assembly != type.Assembly && !IsOurs(method.GetBaseDefinition().DeclaringType!.Assembly))
        {
            return true;
        }

        return !type.IsInterface && type.GetInterfaces()
            .Where(contract => !IsOurs(contract.Assembly))
            .Any(contract => type.GetInterfaceMap(contract).TargetMethods.Contains(method));
    }

    private static bool IsOurs(Assembly assembly) => assembly.GetName().Name!.StartsWith("DeepSharp", StringComparison.Ordinal);

    private static IEnumerable<Type> TypesOf(MemberInfo member) => member switch
    {
        MethodInfo method => method.GetParameters().Select(parameter => parameter.ParameterType).Append(method.ReturnType),
        ConstructorInfo constructor => constructor.GetParameters().Select(parameter => parameter.ParameterType),
        FieldInfo field => [field.FieldType],
        _ => [],
    };

    private static bool IsOrHoldsATuple(Type type)
    {
        if (type.HasElementType)
        {
            return IsOrHoldsATuple(type.GetElementType()!);
        }

        if (!type.IsGenericType)
        {
            return false;
        }

        var definition = type.GetGenericTypeDefinition().FullName!;

        return definition.StartsWith("System.ValueTuple`", StringComparison.Ordinal)
               || definition.StartsWith("System.Tuple`", StringComparison.Ordinal)
               || type.GetGenericArguments().Any(IsOrHoldsATuple);
    }

    private static string ListedTuple(MemberInfo member) =>
        $"{member.DeclaringType!.Assembly.GetName().Name}: {member.DeclaringType.FullName!.Replace('+', '.')}.{member.Name}" +
        (member is MethodBase method
            ? $"({string.Join(", ", method.GetParameters().Select(parameter => Named(parameter.ParameterType)))})"
            : string.Empty);

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

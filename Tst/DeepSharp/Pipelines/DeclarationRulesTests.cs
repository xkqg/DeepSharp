// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Emit;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A declaration is the program that runs, so everything it says has to be something that can run. A
/// second source, a second schema, a second split or a second target cannot: the run used to take the first
/// of each and quietly ignore the rest, so the file said one thing and the numbers came from another. A
/// notebook makes each of these one edit away, which is why they are refused where every door passes
/// through rather than in the door that happened to be used — and why every rule here holds for every
/// prefix of a declaration, since a notebook builds one for each block.
/// </summary>
public class DeclarationRulesTests
{
    private static readonly SplitShares Shares = new(0.70, 0.15, 0.15);

    private static DeclareStep Schema(params string[] names) =>
        new([.. names.Select(name => new ColumnDeclaration(name, ColumnKind.Number, false))]);

    private static DeclarationException Refused(params IPipelineStep[] steps) =>
        Assert.Throws<DeclarationException>(() => new PipelineDeclaration(steps));

    [Fact]
    public void ADeclarationWithTwoSources_IsRefused()
    {
        var refused = Refused(new ReadCsvStep("a.csv"), new ReadCsvStep("b.csv"), Schema("fare"));

        Assert.Contains(refused.Faults, fault => fault.At == 1 && fault.Verb == "read.csv");
    }

    [Fact]
    public void ADeclarationWithTwoSchemas_IsRefused()
    {
        var refused = Refused(new ReadCsvStep("a.csv"), Schema("fare"), Schema("age"));

        var fault = Assert.Single(refused.Faults);

        Assert.Equal(2, fault.At);
        Assert.Equal("declare", fault.Verb);
    }

    [Fact]
    public void ADeclarationWithTwoSplits_IsRefused()
    {
        // The builder already refused this; a file and a notebook did not, and the second split was inert.
        var refused = Refused(
            new ReadCsvStep("a.csv"), Schema("t"), new SplitAtRandomStep(Shares, 1), new SplitByTimeStep("t", Shares));

        Assert.Equal("split.byTime", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void ADeclarationWithTwoTargets_IsRefused()
    {
        // With two targets the run predicted the last one and handed the first to the model as a feature:
        // the answer, among the inputs, with nothing going red.
        var refused = Refused(
            new ReadCsvStep("a.csv"), Schema("fare", "survived"), new SplitAtRandomStep(Shares, 1),
            new TargetStep("fare"), new TargetStep("survived"));

        Assert.Equal(4, Assert.Single(refused.Faults).At);
    }

    [Fact]
    public void TheSameRulesHoldForAFile()
    {
        const string json = """
            {"declaration":[{"step":"read.csv","path":"a.csv"},
                            {"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]},
                            {"step":"target","column":"a"},
                            {"step":"target","column":"a"}]}
            """;

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AndForTheChain()
    {
        Assert.Throws<DeclarationException>(() => Pdd.Create().ReadCsv("a.csv").ReadCsv("b.csv"));
        Assert.Throws<DeclarationException>(
            () => Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("a")).SplitAtRandom(0.70).Target("a").Target("a"));
    }

    [Fact]
    public void ASourceThatIsNotTheFirstStep_IsRefused()
    {
        // Rows come from the source, so nothing can stand before it: a schema declared above the file it
        // describes names columns of nothing.
        var refused = Refused(Schema("fare"), new ReadCsvStep("a.csv"));

        Assert.Contains(refused.Faults, fault => fault.At == 1 && fault.Verb == "read.csv");
    }

    [Fact]
    public void AStepBeforeTheColumnsAreDeclared_IsRefused()
    {
        // Every step after the source works on columns, and there are none until the schema says which.
        // This used to fail a whole run later, as a key nobody could find.
        var refused = Refused(new ReadCsvStep("a.csv"), new SplitAtRandomStep(Shares, 1), Schema("fare"));

        var fault = Assert.Single(refused.Faults, fault => fault.At == 1);

        Assert.Equal("split.atRandom", fault.Verb);
        Assert.Contains("declare", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepWithNoSchemaAnywhere_IsRefused()
    {
        var refused = Refused(new ReadCsvStep("a.csv"), new AddFeatureStep("x", "a", Arithmetic.Plus, "b"));

        Assert.Equal("feature.add", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void RowsHandedInNeedNoSource_ButTheSchemaStillComesFirst()
    {
        // Serving hands the rows in, so a declaration may start at its schema; it may not start anywhere else.
        Assert.Equal(2, new PipelineDeclaration([Schema("a"), new AddFeatureStep("b", "a", Arithmetic.Plus, "a")]).Steps.Count);

        Refused(new AddFeatureStep("b", "a", Arithmetic.Plus, "a"), Schema("a"));
    }

    [Fact]
    public void AStepThatDoesNothing_IsRefused()
    {
        // A step the run never acts on used to be carried along and ignored: in the file, in the chain, in
        // nothing the pipeline did. Saying something and doing nothing is refused where it is written.
        var refused = Refused(new ReadCsvStep("a.csv"), Schema("a"), new ClaimStep());

        var fault = Assert.Single(refused.Faults);

        Assert.Equal("claim", fault.Verb);
        Assert.Contains("does nothing", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheTarget_IsTheOneStepThatActsOnNothing()
    {
        Assert.Empty(PipelineDeclaration.FaultsIn([new ReadCsvStep("a.csv"), Schema("a"), new TargetStep("a")]));
        Assert.False(typeof(IActsInAWalk).IsAssignableFrom(typeof(TargetStep)));
    }

    [Fact]
    public void EveryStepThisLibraryShips_DoesExactlyOneThing()
    {
        // Two capabilities on one step cannot compile outside this library — the run would not know which of
        // them the step is — so what is left to pin is that every step here has one, the target aside.
        var steps = new[] { typeof(Pdd).Assembly, typeof(AddIndicatorStep).Assembly }
            .SelectMany(assembly => assembly.GetTypes())
            .Where(type => type is { IsClass: true, IsAbstract: false } && typeof(IPipelineStep).IsAssignableFrom(type))
            .ToArray();

        Assert.NotEmpty(steps);

        foreach (var step in steps)
        {
            var acts = step.GetInterfaces().Count(face => face != typeof(IActsInAWalk) && typeof(IActsInAWalk).IsAssignableFrom(face));

            Assert.True(step == typeof(TargetStep) ? acts == 0 : acts == 1, $"{step.Name} does {acts} things.");
        }
    }

    [Fact]
    public void RowsAreDroppedBeforeTheyAreDivided_NeverAfter()
    {
        // A split divides rows; dropping some afterwards changes what each part holds without the parts
        // knowing, and the run and a replay then disagree about how many rows there are.
        var refused = Refused(
            new ReadCsvStep("a.csv"), Schema("a"), new OrderByStep(["a"]), new SplitAtRandomStep(Shares, 1), new DropWarmUpStep());

        Assert.Equal("drop.warmup", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void EveryFaultIsNamedAtOnce_WithTheStepItIsAt()
    {
        // Somebody handed one fault at a time, five times over, stops using the thing. A notebook needs the
        // place as well, to put each fault on the block it belongs to.
        IPipelineStep[] steps =
        [
            new ReadCsvStep("a.csv"),
            Schema("age", "a"),
            FillMissingStep.Of("age", With.Mean),
            new TargetStep("a"),
            new TargetStep("a"),
        ];

        var faults = PipelineDeclaration.FaultsIn(steps);

        Assert.Equal([2, 4], faults.Select(fault => fault.At));
        Assert.Equal(["fill.missing", "target"], faults.Select(fault => fault.Verb));
        Assert.StartsWith("Step 3, 'fill.missing': learns from the data", faults[0].ToString(), StringComparison.Ordinal);

        var refused = Assert.Throws<DeclarationException>(() => new PipelineDeclaration(steps));

        Assert.Equal(faults, refused.Faults);
        Assert.Contains(faults[0].ToString(), refused.Message, StringComparison.Ordinal);
        Assert.Contains(faults[1].ToString(), refused.Message, StringComparison.Ordinal);
        Assert.IsAssignableFrom<InvalidOperationException>(refused);
        Assert.Throws<ArgumentNullException>(() => new DeclarationException(null!));
    }

    [Fact]
    public void StepsThatMakeADeclaration_HaveNoFaults()
    {
        Assert.Empty(PipelineDeclaration.FaultsIn(
            [new ReadCsvStep("a.csv"), Schema("a"), new SplitAtRandomStep(Shares, 1), new NormaliseStep("a")]));
        Assert.Throws<ArgumentNullException>(() => PipelineDeclaration.FaultsIn(null!));
    }

    [Fact]
    public void ToSayAStepSplitsOrLearns_IsToDoIt()
    {
        // A split used to be a marker with the dividing kept on a second interface, and a step that learns
        // the same: a step could carry the marker alone, pass every rule, and then divide nothing — every row
        // came out as training — or fit nothing at all. The marker and the capability are one thing now, so
        // saying it without doing it does not compile.
        Assert.NotNull(typeof(ISplitStep).GetMethod(nameof(ISplitStep.Assign)));
        Assert.NotNull(typeof(IFittedStep).GetMethod(nameof(IFittedStep.Fit)));
        Assert.NotNull(typeof(IFittedStep).GetMethod(nameof(IFittedStep.ApplyTo)));

        Assert.Null(typeof(Pdd).Assembly.GetType("DeepSharp.Pipelines.IAssignsParts"));
        Assert.Null(typeof(Pdd).Assembly.GetType("DeepSharp.Pipelines.ILearnsFromData"));
    }

    public static TheoryData<string> ValidDeclarations() =>
    [
        "titanic",
        "apple",
        "handed in",
        "empty",
    ];

    [Theory]
    [MemberData(nameof(ValidDeclarations))]
    public void EveryPrefixOfEveryValidDeclaration_IsValid(string which)
    {
        // A notebook builds a declaration for every block — the steps down to it — and rebuilds them on every
        // edit. So a rule a whole declaration keeps has to be kept by each of its beginnings too, or the grid
        // under a perfectly good block would refuse to show.
        IReadOnlyList<IPipelineStep> steps = which switch
        {
            "titanic" => Pdd.Create()
                .ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived", "pclass").Category("sex", "embarked").Optional("age", ColumnKind.Number).Number("fare"))
                .Reshape("fare", Maths.Log1P)
                .SplitStratified("survived", 0.70, 0.15)
                .FillMissing("age", With.Median)
                .ClipOutliers("fare")
                .Normalise("age", "fare")
                .EncodeCategories()
                .Drop("age_was_missing")
                .Target("survived")
                .Declaration.Steps,
            "apple" => Pdd.Create()
                .ReadCsv(Repository.Data("apple.csv"))
                .Declare(schema => schema.Timestamp("Date").Number("Open", "High", "Low", "Close", "Volume"))
                .OrderBy("Date")
                .AddIndicator("sma5", Indicator.Sma, ["Close"], 5)
                .DropWarmUp()
                .TimeParts("Date", TimePart.Month)
                .SplitByTime("Date", 0.70, 0.15)
                .Normalise("Close", Scale.Robust)
                .EncodeCategories()
                .Target("Close")
                .Declaration.Steps,
            "handed in" => [Schema("a", "b"), new AddFeatureStep("c", "a", Arithmetic.Plus, "b"), new SplitAtRandomStep(Shares, 1), new NormaliseStep("c")],
            _ => [],
        };

        Assert.Empty(PipelineDeclaration.FaultsIn(steps));

        for (var count = 0; count <= steps.Count; count++)
        {
            Assert.Empty(PipelineDeclaration.FaultsIn(steps.Take(count)));
        }
    }

    [Fact]
    public void ANewCapability_NeedsNoChangeToTheWalker()
    {
        // The walk hands every step to the step, and the step does what it does. A walker that asked each
        // step which capability it had would be a list that every new capability has to be added to, and a
        // list that has to agree with the rules about which steps act. So the walker is read here, instruction
        // by instruction, for any mention of a capability other than the one they all share.
        var walk = typeof(Pdd).Assembly.GetType("DeepSharp.Pipelines.Walk", throwOnError: true)!;
        var mode = typeof(Pdd).Assembly.GetType("DeepSharp.Pipelines.WalkMode", throwOnError: true)!;
        var walker = typeof(Pdd).Assembly.GetTypes().Where(type => type == walk || mode.IsAssignableFrom(type)).ToArray();

        var capabilities = typeof(Pdd).Assembly.GetTypes()
            .Where(type => type.IsInterface && type != typeof(IActsInAWalk) && typeof(IActsInAWalk).IsAssignableFrom(type))
            .ToHashSet();

        Assert.NotEmpty(capabilities);
        Assert.True(walker.Length > 1);

        var mentioned = walker.SelectMany(TypesMentionedIn).Where(capabilities.Contains).Select(type => type.Name).Distinct().ToArray();

        Assert.True(mentioned.Length == 0, $"The walker asks about: {string.Join(", ", mentioned)}");
    }

    /// <summary>Every type a class's code names: in its signatures, and in the instructions of its methods and of the closures inside them.</summary>
    private static IEnumerable<Type> TypesMentionedIn(Type type)
    {
        const BindingFlags everything =
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        var codes = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
            .Select(field => (OpCode)field.GetValue(null)!)
            .ToDictionary(code => code.Value);

        foreach (var nested in type.GetNestedTypes(BindingFlags.NonPublic | BindingFlags.Public).SelectMany(TypesMentionedIn))
        {
            yield return nested;
        }

        foreach (var field in type.GetFields(everything))
        {
            yield return field.FieldType;
        }

        foreach (var method in type.GetMethods(everything).Cast<MethodBase>().Concat(type.GetConstructors(everything)))
        {
            foreach (var parameter in method.GetParameters())
            {
                yield return parameter.ParameterType;
            }

            if (method.GetMethodBody()?.GetILAsByteArray() is not { } il)
            {
                continue;
            }

            for (var at = 0; at < il.Length;)
            {
                var code = il[at] == 0xFE ? codes[(short)(0xFE00 | il[at + 1])] : codes[il[at]];
                at += code.Size;

                switch (code.OperandType)
                {
                    case OperandType.InlineType or OperandType.InlineTok or OperandType.InlineMethod or OperandType.InlineField:
                        var token = BitConverter.ToInt32(il, at);
                        var member = method.Module.ResolveMember(token, type.GetGenericArguments(), method.IsGenericMethod ? method.GetGenericArguments() : null);

                        if (member is Type named)
                        {
                            yield return named;
                        }
                        else if (member?.DeclaringType is { } owner)
                        {
                            yield return owner;
                        }

                        at += 4;
                        break;

                    case OperandType.InlineSwitch:
                        at += 4 + (4 * BitConverter.ToInt32(il, at));
                        break;

                    case OperandType.InlineI8 or OperandType.InlineR:
                        at += 8;
                        break;

                    case OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar:
                        at += 1;
                        break;

                    case OperandType.InlineVar:
                        at += 2;
                        break;

                    case OperandType.InlineNone:
                        break;

                    default:
                        at += 4;
                        break;
                }
            }
        }
    }

    /// <summary>A step that says something and does nothing a run could act on.</summary>
    private sealed record ClaimStep : IPipelineStep<ClaimStep>
    {
        public static string Name => "claim";

        public static string Purpose => "Claims to be a step.";

        public static StepParameters<ClaimStep> Parameters { get; } = new();

        public string Verb => Name;

        public static ClaimStep ReadFrom(JsonElement element) => new();
    }
}

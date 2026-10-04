// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A verb that does one thing to one column has the line that does it to many, or it is written down here with its
/// reason.
/// </summary>
/// <remarks>
/// The repetition this removes was four lines differing in a name, and it came back every time a verb of the shape was
/// added without anybody remembering the door. So the shape is the test: every verb whose first word is a column
/// either takes a line, or stands in the list below and says why — and a verb added to any package DeepSharp ships
/// turns this red until somebody decides which it is.
/// </remarks>
public class VerbShapeTests
{
    /// <summary>The verbs that do one thing to one column and deliberately have no line, each with its reason.</summary>
    private static readonly Dictionary<string, string> WithoutALine = new(StringComparer.Ordinal)
    {
        ["Encode"] =
            "EncodeCategories() already writes every category at once, taken where it stands, so a list of columns is "
            + "the exception rather than the rule.",
        ["Target"] = "It names the one column a model is asked to predict; a pipeline has one output.",
        ["Ahead"] = "It makes the one answer, from the one column it is read from.",
        ["SplitByTime"] = "It divides the rows once, by the one column that says when a row happened.",
        ["SplitStratified"] = "It divides the rows once, keeping the mixture of the one column it is told to keep.",
    };

    public static TheoryData<string, string> EveryVerbOfTheShape()
    {
        var verbs = new TheoryData<string, string>();

        foreach (var (verb, declaring) in OfTheShape())
        {
            verbs.Add(verb, declaring);
        }

        return verbs;
    }

    [Theory]
    [MemberData(nameof(EveryVerbOfTheShape))]
    public void EveryVerbThatDoesOneThingToOneColumn_HasTheLineOrSaysWhyNot(string verb, string declaring)
    {
        if (WithoutALine.TryGetValue(verb, out var reason))
        {
            Assert.NotEmpty(reason);

            return;
        }

        var lines = Builders()
            .SelectMany(builder => builder.GetMethods())
            .Where(method => method.Name == verb)
            .SelectMany(method => method.GetParameters())
            .Where(parameter => parameter.ParameterType.IsGenericType
                && parameter.ParameterType.GetGenericTypeDefinition() == typeof(Action<>))
            .Select(parameter => parameter.ParameterType.GetGenericArguments()[0])
            .ToArray();

        Assert.True(
            lines.Length > 0,
            $"'{verb}' on {declaring} does one thing to one column and has no line that does it to many. Give it one, "
            + $"or write down in {nameof(WithoutALine)} why it has none.");

        Assert.All(lines, line => Assert.Contains("IDeclaresSteps", line.GetInterfaces().Select(face => face.Name)));
    }

    [Fact]
    public void EveryLine_IsFannedOutByTheOnePlace()
    {
        // One fan-out, so the refusal of a line that names no column, and the order the steps are added in, are the
        // same for every verb that has one.
        var fanOuts = Builders()
            .SelectMany(builder => builder.GetMethods(BindingFlags.NonPublic | BindingFlags.Instance))
            .Where(method => method.Name == "Many")
            .ToArray();

        Assert.Equal(Builders().Count(), fanOuts.Length);
        Assert.All(fanOuts, method => Assert.True(method.IsGenericMethod));
    }

    [Fact]
    public void TheShapeFindsEveryVerbThereIs()
    {
        // The test is only worth anything if it finds them, so the list is pinned: a verb added or taken away says so
        // here first.
        Assert.Equal(
            [
                "Ahead", "ClipOutliers", "Cyclical", "Encode", "FillMissing", "FillNaN", "Normalise", "Reshape",
                "SettleGaps", "SplitByTime", "SplitStratified", "Target", "TimeParts", "TimePartsAsNumbers",
            ],
            OfTheShape().Select(each => each.Verb).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void NoVerbOfTheShape_IsLeftUnjudged()
    {
        // The list of reasons may not name a verb that is not of the shape any more, or the reason outlives the verb.
        var shape = OfTheShape().Select(each => each.Verb).ToHashSet(StringComparer.Ordinal);

        Assert.All(WithoutALine.Keys, verb => Assert.Contains(verb, shape));
    }

    private static IEnumerable<Type> Builders() => [typeof(PipelineBuilder), typeof(FittingBuilder)];

    // A verb of the shape: its first word is one column, so writing it for several columns writes it several times.
    private static IEnumerable<(string Verb, string Declaring)> OfTheShape() =>
        Builders()
            .SelectMany(builder => builder.GetMethods().Select(method => (Method: method, Declaring: builder.Name)))
            .Concat(Shipped.StepAssemblies
                .SelectMany(assembly => assembly.GetExportedTypes())
                .Where(type => type.IsAbstract && type.IsSealed)
                .SelectMany(type => type.GetMethods(BindingFlags.Public | BindingFlags.Static))
                .Where(method => method.GetParameters() is [{ } receiver, ..]
                    && Builders().Contains(receiver.ParameterType))
                .Select(method => (Method: method, Declaring: method.DeclaringType!.Name)))
            .Where(each => Column(each.Method) is not null)
            .Select(each => (each.Method.Name, each.Declaring))
            .Distinct();

    // The first word after the receiver, when it is one named column.
    private static ParameterInfo? Column(MethodInfo method) =>
        method.GetParameters()
            .SkipWhile(parameter => Builders().Contains(parameter.ParameterType))
            .FirstOrDefault() is { Name: "column", ParameterType.Name: nameof(String) } first
            ? first
            : null;
}

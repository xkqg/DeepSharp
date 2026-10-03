// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One line says what goes in the gaps of many columns, and the two verbs that choose from that vocabulary share it.
/// </summary>
/// <remarks>
/// A value that was never there and a value arithmetic could not make are different things, so they stay different
/// verbs — but both choose from one list of what to put there, and the list is written once. What reaches the
/// declaration is one step a column, as it always was.
/// </remarks>
public class FillLineTests
{
    private static FittingBuilder Split() =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema
                .Integer("survived")
                .Number("fare", "trades", "ratio")
                .Optional("age", ColumnKind.Number))
            .SplitStratified("survived", train: 0.70, validation: 0.15);

    [Fact]
    public void OneGapLine_FillsEachColumnTheWayItsKindNames()
    {
        var filled = Split()
            .FillMissing(fill => fill
                .Median("age")
                .Mean("fare", "trades"))
            .Build()
            .Declaration.Steps.OfType<FillMissingStep>().ToArray();

        Assert.Equal(["age", "fare", "trades"], filled.Select(step => step.Column));
        Assert.Equal(
            [With.Median.Name, With.Mean.Name, With.Mean.Name],
            filled.Select(step => step.Strategy.Name));
    }

    [Fact]
    public void TheOneLineAndTheLinePerColumn_DeclareTheSameSteps()
    {
        var one = Split()
            .FillMissing(fill => fill.Median("age").Constant(0, "fare"))
            .Build()
            .Declaration;

        var each = Split()
            .FillMissing("age", With.Median)
            .FillMissing("fare", With.Constant(0))
            .Build()
            .Declaration;

        Assert.Equal(each.Steps, one.Steps);
        Assert.Equal(each.ToJson(), one.ToJson());
    }

    [Fact]
    public void OneNotANumberLine_SaysWhatHappensInEachColumn()
    {
        var watched = Split()
            .FillNaN(fill => fill
                .Refuse("ratio")
                .Zero("trades"))
            .Build()
            .Declaration.Steps.OfType<FillNaNStep>().ToArray();

        Assert.Equal(["ratio", "trades"], watched.Select(step => step.Column));
        Assert.Equal([With.Refuse.Name, With.Zero.Name], watched.Select(step => step.Strategy.Name));
    }

    [Fact]
    public void TheTwoVerbs_ShareTheVocabularyTheyBothHave_AndNoMore()
    {
        // The shared list is what both verbs accept. The row above a value that is not a number says nothing about what
        // it should have been, so `fill.nan` refuses `previous` where it is read — and the line that writes it must not
        // offer a kind its verb would throw on.
        Assert.Equal(
            [.. Kinds(typeof(NotANumberLine)).Append("Previous").Order(StringComparer.Ordinal)],
            Kinds(typeof(GapLine)));
        Assert.DoesNotContain("Previous", Kinds(typeof(NotANumberLine)));
        Assert.NotEmpty(Kinds(typeof(NotANumberLine)));
    }

    [Fact]
    public void TheGapLine_CarriesTheValueBeforeTheGap()
    {
        // It reads the rows in their order, so the order is declared above it.
        var filled = Pdd.Create()
            .ReadCsv("apple.csv")
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close").Optional("trades", ColumnKind.Number))
            .OrderBy("Date")
            .SplitByTime("Date", train: 0.70, validation: 0.15)
            .FillMissing(fill => fill.Previous("trades"))
            .Build()
            .Declaration.Steps.OfType<FillMissingStep>().Single();

        Assert.Equal("trades", filled.Column);
        Assert.Equal(With.Previous.Name, filled.Strategy.Name);
    }

    [Theory]
    [InlineData("Mean")]
    [InlineData("Median")]
    [InlineData("Zero")]
    [InlineData("Previous")]
    [InlineData("Refuse")]
    public void EveryKindAGapLineOffers_IsOneItsVerbTakes(string kind)
    {
        // Every method is invoked, so a kind added to the line without its verb accepting it fails here rather than at
        // the first run of somebody's pipeline.
        var line = new GapLine();

        typeof(GapLine).GetMethod(kind, [typeof(string[])])!.Invoke(line, [new[] { "age" }]);

        Assert.Equal("fill.missing", Declared(line).Single().Verb);
    }

    [Theory]
    [InlineData("Mean")]
    [InlineData("Median")]
    [InlineData("Zero")]
    [InlineData("Refuse")]
    public void EveryKindANotANumberLineOffers_IsOneItsVerbTakes(string kind)
    {
        var line = new NotANumberLine();

        typeof(NotANumberLine).GetMethod(kind, [typeof(string[])])!.Invoke(line, [new[] { "fare" }]);

        Assert.Equal("fill.nan", Declared(line).Single().Verb);
    }

    [Fact]
    public void AConstantIsAKindOfBothLines_AndEachVerbTakesIt()
    {
        Assert.Equal("fill.missing", Declared(new GapLine().Constant(0, "age")).Single().Verb);
        Assert.Equal("fill.nan", Declared(new NotANumberLine().Constant(0, "fare")).Single().Verb);
    }

    private static IReadOnlyList<IPipelineStep> Declared(object line) =>
        ((IReadOnlyList<IPipelineStep>?)line.GetType()
            .GetInterfaceMap(line.GetType().GetInterfaces().Single(face => face.Name == "IDeclaresSteps"))
            .TargetMethods.Single(method => method.Name.EndsWith("get_Steps", StringComparison.Ordinal))
            .Invoke(line, null))!;

    private static string[] Kinds(Type line) =>
        [.. line.GetMethods()
            .Where(method => method.DeclaringType == line || method.DeclaringType == line.BaseType)
            .Where(method => method.GetParameters() is [{ ParameterType.IsArray: true }])
            .Select(method => method.Name)
            .Order(StringComparer.Ordinal)];

    [Fact]
    public void AShareAboveWhichFillingIsRefused_IsNotOnTheLine()
    {
        // That limit is a fact about one column and has no default on purpose, so it stays on the verb that takes one
        // column rather than becoming a number the group would have to share.
        Assert.DoesNotContain(
            typeof(GapLine).GetMethods().Concat(typeof(FillLine<GapLine>).GetMethods()),
            method => method.GetParameters().Any(parameter => parameter.Name == "refuseAbove"));
    }

    [Fact]
    public void AnEmptyFillLine_IsRefusedWhereItIsWritten()
    {
        Assert.Throws<ArgumentException>(() => Split().FillMissing(fill => { }));
        Assert.Throws<ArgumentException>(() => Split().FillNaN(fill => { }));
    }
}

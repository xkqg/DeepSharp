// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One line says what happens to many columns, and the kinds are the methods.
/// </summary>
/// <remarks>
/// A pipeline that scaled four columns was four lines that differed in a name, and the only way to say a kind was to
/// repeat the whole verb. The line now names the kind once and lists the columns it holds for, as the schema's own
/// builder does, and what reaches the declaration is what it always was: one step a column. So the file, the notebook's
/// blocks and every fit keyed by the steps above it are untouched — the change is at the door a person writes.
/// </remarks>
public class ManyColumnsTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema
                .Integer("survived", "sibsp", "parch")
                .Number("fare", "volume")
                .Optional("age", ColumnKind.Number));

    private static FittingBuilder Split() => Passengers().SplitStratified("survived", train: 0.70, validation: 0.15);

    [Fact]
    public void OneScaleLine_AddsOneStepPerColumnWithTheKindTheMethodNames()
    {
        var declared = Split()
            .Normalise(scale => scale
                .MidRange("age", "fare")
                .Robust("volume"))
            .Build()
            .Declaration;

        var scaled = declared.Steps.OfType<NormaliseStep>().ToArray();

        Assert.Equal(["age", "fare", "volume"], scaled.Select(step => step.Column));
        Assert.Equal([Scale.MidRange, Scale.MidRange, Scale.Robust], scaled.Select(step => step.Scale));
    }

    [Fact]
    public void AScaleLineThatNamesNoKind_WritesTheOneWrittenDownDefault()
    {
        var scaled = Split()
            .Normalise(scale => scale.Columns("age", "fare"))
            .Build()
            .Declaration.Steps.OfType<NormaliseStep>().ToArray();

        Assert.Equal([NormaliseStep.DefaultScale, NormaliseStep.DefaultScale], scaled.Select(step => step.Scale));
    }

    [Fact]
    public void TheOneLineAndTheLinePerColumn_DeclareTheSameSteps()
    {
        // The door is the only thing that changes: what the declaration holds is step for step what it held.
        var one = Split()
            .Normalise(scale => scale.MidRange("age", "fare").Robust("volume"))
            .Build()
            .Declaration;

        var each = Split()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("volume", Scale.Robust)
            .Build()
            .Declaration;

        Assert.Equal(each.Steps, one.Steps);
        Assert.Equal(each.ToJson(), one.ToJson());
    }

    [Fact]
    public void AKindThatCanHoldAValueInItsRange_SaysWhatHappensOutsideIt()
    {
        var scaled = Split()
            .Normalise(scale => scale
                .MinMax(OutOfRange.Refuse, "age")
                .Quantile(OutOfRange.Clip, "fare"))
            .Build()
            .Declaration.Steps.OfType<NormaliseStep>().ToArray();

        Assert.Equal([OutOfRange.Refuse, OutOfRange.Clip], scaled.Select(step => step.OutOfRange));
    }

    [Fact]
    public void AKindThatLandsItsRowsInNoRange_HasNoLineToSayWhatHappensOutsideIt()
    {
        // The rule lives in which methods there are: a scale that lands its rows nowhere has nothing to hold a value
        // in, and a rank has no place beyond the training rows to let one through, so neither can be written wrongly.
        var scale = typeof(ScaleBuilder);

        Assert.DoesNotContain(
            scale.GetMethods(),
            method => method.Name is "Standard" or "Robust" or "Power"
                && method.GetParameters().Any(parameter => parameter.ParameterType == typeof(OutOfRange)));

        Assert.All(
            scale.GetMethods().Where(method => method.Name == "Quantile"),
            method => Assert.Contains(method.GetParameters(), parameter => parameter.ParameterType == typeof(OutOfRange)));
    }

    [Fact]
    public void AnEmptyScaleLine_IsRefusedWhereItIsWritten()
    {
        // A line that scales nothing is a line somebody meant to finish.
        Assert.Throws<ArgumentException>(() => Split().Normalise(scale => { }));
    }
}

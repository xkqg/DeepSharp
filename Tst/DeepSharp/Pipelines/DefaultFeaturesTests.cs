// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Where the features land is said once, for the whole pipeline, and a column that names its own kind keeps it.
/// </summary>
/// <remarks>
/// A network takes every feature between minus one and one, and that is what a scaling which names no kind means. It is
/// not the only range anybody wants, so the pipeline says which — once, above the split, where it holds for every
/// column alike — and the scalings that name no kind are written as the kind that lands them there. Nothing new reaches
/// the file: the step still writes the word it always wrote, so a declaration that says the range and one that names
/// the kind are the same text.
/// </remarks>
public class DefaultFeaturesTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema
                .Integer("survived", "sibsp", "parch")
                .Number("fare")
                .Optional("age", ColumnKind.Number));

    [Fact]
    public void ADeclaredUnitRange_WritesACyclicalThatNamesNoFormInThatRange()
    {
        // A place on a circle is a feature too, so where it is written is where the pipeline says its features land.
        var declared = Pdd.Create()
            .ReadCsv("apple.csv")
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .DefaultFeatures(Form.Unit)
            .Cyclical("Date", Period.MonthOfYear)
            .Cyclical(cycle => cycle.DayOfWeek("Date"))
            .Declaration;

        Assert.Equal([Form.Unit, Form.Unit], declared.Steps.OfType<CyclicalStep>().Select(step => step.Form));
    }

    [Fact]
    public void ACyclicalThatNamesItsOwnForm_KeepsIt()
    {
        var declared = Pdd.Create()
            .ReadCsv("apple.csv")
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .DefaultFeatures(Form.Unit)
            .Cyclical("Date", Period.MonthOfYear, Form.SplitSign)
            .Declaration;

        Assert.Equal([Form.SplitSign], declared.Steps.OfType<CyclicalStep>().Select(step => step.Form));
    }

    [Fact]
    public void ARangeDeclaredAfterAStepItWouldGovern_IsRefused()
    {
        // It is said once and it governs what comes after, so a step already standing that it would have governed is a
        // question with two answers — refused where the second is written.
        var chain = Pdd.Create()
            .ReadCsv("apple.csv")
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .Cyclical("Date", Period.MonthOfYear);

        var faulty = Assert.Throws<InvalidOperationException>(() => { chain.DefaultFeatures(Form.Unit); });

        Assert.Contains("cyclical", faulty.Message, StringComparison.Ordinal);
    }

    private static IEnumerable<Scale> ScalesOf(Pipeline pipeline) =>
        pipeline.Declaration.Steps.OfType<NormaliseStep>().Select(step => step.Scale);

    [Fact]
    public void ADeclaredUnitRange_MakesAScalingThatNamesNoKindWriteMinMax()
    {
        var declared = Passengers()
            .DefaultFeatures(Form.Unit)
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Normalise("age", "fare")
            .Normalise(scale => scale.Columns("sibsp"))
            .Build();

        Assert.Equal([Scale.MinMax, Scale.MinMax, Scale.MinMax], ScalesOf(declared));
    }

    [Fact]
    public void NoDeclaredRange_LandsTheFeaturesBetweenMinusOneAndOne()
    {
        var declared = Passengers()
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Normalise("age", "fare")
            .Normalise(scale => scale.Columns("sibsp"))
            .Build();

        Assert.Equal([Scale.MidRange, Scale.MidRange, Scale.MidRange], ScalesOf(declared));
        Assert.All(ScalesOf(declared), scale => Assert.Equal(Form.Signed, scale.Lands()));
    }

    [Fact]
    public void AColumnThatNamesItsOwnKind_KeepsItWhateverRangeIsDeclared()
    {
        var declared = Passengers()
            .DefaultFeatures(Form.Unit)
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Normalise("age", Scale.Robust)
            .Normalise(scale => scale.MidRange("fare"))
            .Build();

        Assert.Equal([Scale.Robust, Scale.MidRange], ScalesOf(declared));
    }

    [Fact]
    public void ARangeThatIsTwoColumns_IsRefusedWhereItIsWritten()
    {
        // SplitSign writes a value as two columns rather than landing one in a range, so it is no answer to where the
        // features land.
        var faulty = Assert.Throws<ArgumentOutOfRangeException>(() => { Passengers().DefaultFeatures(Form.SplitSign); });

        Assert.Contains("SplitSign", faulty.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADeclaredRange_WritesNoNewKeyIntoTheFile()
    {
        // The range is a door, not a key: the file says the kind each column was scaled by, as it always did.
        var byRange = Passengers()
            .DefaultFeatures(Form.Unit)
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Normalise("age", "fare")
            .Build();

        var byName = Passengers()
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Normalise(scale => scale.MinMax("age", "fare"))
            .Build();

        Assert.Equal(byName.Declaration.ToJson(), byRange.Declaration.ToJson());
    }

    [Fact]
    public void TheRangeIsSaidOnceAndOnlyAboveTheSplit()
    {
        // It holds for every column of the pipeline, so it is declared where the columns are, and a second saying of it
        // would leave two answers to one question.
        var chain = Passengers().DefaultFeatures(Form.Unit);

        Assert.Throws<InvalidOperationException>(() => { chain.DefaultFeatures(Form.Signed); });
    }
}

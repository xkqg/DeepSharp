// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A network takes every feature on one scale, between minus one and one, and says so where the numbers are handed to
/// it. Each column carries where its values land — a scale's range, a form, an encoding's noughts and ones — as the
/// columns are followed down the steps, so the handover refuses a feature declared to land anywhere else without
/// reading a row, and a pipeline loaded from its file is held to it as the one that was fitted.
/// </summary>
public class OneScaleTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    private static PreparedData Passengers(Action<FittingBuilder> prepared)
    {
        var fitting = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived").Category("sex").Optional("age", ColumnKind.Number).Number("fare").Boolean("alone"))
            .SplitStratified("survived", 0.70, 0.15);

        prepared(fitting);

        return fitting.Target("survived").Build().Run();
    }

    private static void OnOneScale(FittingBuilder fitting) =>
        fitting.FillMissing("age", With.Median).Normalise("age", Scale.MidRange).Normalise("fare", Scale.MinMax).EncodeCategories();

    [Fact]
    public void FeaturesScaledIntoARange_EncodedOneHot_TrueOrFalse_OrMarkingAGap_AreOnOneScale()
    {
        var prepared = Passengers(OnOneScale);

        var batch = prepared.Batch(Part.Train, Needs.OneScale);

        Assert.Equal(prepared.Batch(Part.Train).FeatureNames, batch.FeatureNames);
        Assert.Contains("sex_male", batch.FeatureNames);
        Assert.Contains("age_was_missing", batch.FeatureNames);
        Assert.All(batch.Features, row => Assert.All(row, value => Assert.InRange(value, -1, 1)));
    }

    [Fact]
    public void AFeatureScaledIntoNoRange_OrEncodedAsPlaces_OrNeverScaled_IsRefused_EveryOneAtOnce()
    {
        var prepared = Passengers(fitting => fitting
            .FillMissing("age", With.Median)
            .Normalise("age", Scale.Standard)
            .EncodeCategories(As.Ordinal));

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Train, Needs.OneScale));

        Assert.Contains("'age'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'sex'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'fare'", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("'alone'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("between minus one and one", refused.Message, StringComparison.Ordinal);

        // A learner indifferent to scale takes them as they are.
        Assert.Equal(prepared.CountIn(Part.Train), prepared.Batch(Part.Train, Needs.Numbers).RowCount);
    }

    [Fact]
    public void APipelineLoadedFromItsFile_IsHeldToItAsTheOneThatWasFitted()
    {
        var onOneScale = PreparedData.FromJson(Passengers(OnOneScale).ToJson(), StepCatalog.BuiltIn());
        var standard = PreparedData.FromJson(
            Passengers(fitting => fitting.FillMissing("age", With.Median).Normalise("age", Scale.Standard).Normalise("fare", Scale.MinMax).EncodeCategories()).ToJson(),
            StepCatalog.BuiltIn());
        var row = new InMemoryRowSource(["survived", "sex", "age", "fare", "alone"], [["1", "female", "30", "7.25", "True"]]);

        Assert.Equal(1, onOneScale.Served(row, Needs.OneScale).RowCount);
        Assert.Contains("'age'", Assert.Throws<InvalidOperationException>(() => standard.Served(row, Needs.OneScale)).Message, StringComparison.Ordinal);
        Assert.Equal(1, standard.Served(row, Needs.Numbers).RowCount);
    }

    [Fact]
    public void AFillKeepsWhereAColumnLands_UnlessTheValueItWritesLiesOutside()
    {
        var inside = Passengers(fitting => fitting
            .Normalise("age", Scale.MidRange).FillMissing("age", With.Constant(-1)).Normalise("fare", Scale.MaxAbs).EncodeCategories());

        Assert.Equal(inside.CountIn(Part.Train), inside.Batch(Part.Train, Needs.OneScale).RowCount);

        var outside = Passengers(fitting => fitting
            .Normalise("age", Scale.MinMax).FillMissing("age", With.Constant(5)).Normalise("fare", Scale.MaxAbs).EncodeCategories());

        Assert.Contains("'age'", Assert.Throws<InvalidOperationException>(() => outside.Batch(Part.Train, Needs.OneScale)).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Form.Signed, Form.Signed)]
    [InlineData(Form.Unit, Form.Unit)]
    [InlineData(Form.SplitSign, Form.Unit)]
    public void AMomentOnACircle_LandsWhereItsFormSays(Form form, Form lands)
    {
        var declaration = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .Cyclical("Date", Period.DayOfWeek, form)
            .Declaration;

        var columns = declaration.ColumnsBefore(declaration.Steps.Count);
        var made = columns.Columns.Where(column => column.Name.StartsWith("Date_", StringComparison.Ordinal)).ToArray();

        Assert.Equal(form == Form.SplitSign ? 4 : 2, made.Length);
        Assert.All(made, column => Assert.Equal(lands, columns.LandsOf(column.Name)));
    }

    [Fact]
    public void WhereAColumnLands_IsFollowedDownTheSteps()
    {
        var declaration = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Number("fare", "age").Boolean("alone"))
            .SplitStratified("survived", 0.70, 0.15)
            .Normalise("fare", Scale.MidRange)
            .Normalise("age", Scale.MidRange)
            .ClipOutliers("age")
            .NormaliseRow(Norm.L1, "sibsp", "parch")
            .Encode("sex")
            .Drop("sex_other")
            .Declaration;

        var columns = declaration.ColumnsBefore(declaration.Steps.Count);

        Assert.Equal(Form.Signed, columns.LandsOf("fare"));
        Assert.Equal(Form.Signed, columns.LandsOf("sibsp"));
        Assert.Equal(Form.Unit, columns.LandsOf("alone"));
        Assert.Equal(Form.Unit, columns.LandsOf("sex_male"));
        Assert.Equal(Form.Unit, columns.LandsOf("sex_was_missing"));

        // A member of a family taken away by name is not there to land anywhere.
        Assert.Null(columns.LandsOf("sex_other"));

        // A step that writes a column anew without saying where it lands leaves it landing nowhere said.
        Assert.Null(columns.LandsOf("age"));
        Assert.Null(columns.LandsOf("survived"));
        Assert.Null(columns.LandsOf("nowhere"));
    }
}

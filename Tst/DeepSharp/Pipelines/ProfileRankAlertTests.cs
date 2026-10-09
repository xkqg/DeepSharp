// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A profile can be told how alike in order two columns may be before it says so: a threshold, declared and never defaulted,
/// on Spearman's coefficient between two columns over the training rows that have a number in both. A column that follows
/// another's order without repeating its values is not found by the alert for columns that go with each other value for
/// value, and a model is handed the same thing twice. Every reference value is what scipy answered, in
/// Fixtures/correlation-reference.txt.
/// </summary>
public class ProfileRankAlertTests
{
    private static readonly double[] Rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    // scipy 1.18.1: rows against rows + 2.5 * noise.
    private static readonly double[] Follows = [1.75, -0.25, 4.25, 7.5, 2.25, 6.5, 6.0, 10.5, 7.25, 12.0, 7.75, 13.5];

    // scipy 1.18.1: the same with its first two values swapped.
    private static readonly double[] Near = [-0.25, 1.75, 4.25, 7.5, 2.25, 6.5, 6.0, 10.5, 7.25, 12.0, 7.75, 13.5];

    // scipy 1.18.1: two columns with nothing to do with one another.
    private static readonly double[] One = [5, 12, 3, 8, 1, 10, 7, 2, 11, 4, 9, 6];
    private static readonly double[] Other = [2, 9, 6, 12, 4, 1, 10, 5, 3, 11, 7, 8];

    [Fact]
    public void WithNoThreshold_NoPairIsFlagged_WhateverTheirOrder()
    {
        var profile = Profiled(null, ["a", "b"], Rows, [.. Rows.Select(Math.Exp)]);

        Assert.Empty(RankAlerts(profile));
    }

    [Fact]
    public void AColumnThatKeepsAnothersOrder_IsFlagged_NamingBothAndTheCoefficient()
    {
        var profile = Profiled(0.9, ["a", "b"], Rows, [.. Rows.Select(Math.Exp)]);

        var alert = Assert.Single(RankAlerts(profile));

        Assert.Equal(["b", "a"], alert.Columns);
        Assert.Equal(AlertAnswer.LeaveOut("b"), alert.Answer);
        Assert.Contains("Spearman", alert.Says, StringComparison.Ordinal);
        Assert.Contains("1.000", alert.Says, StringComparison.Ordinal);
        Assert.Contains("12 rows", alert.Says, StringComparison.Ordinal);
        Assert.Contains("0.9", alert.Says, StringComparison.Ordinal);
    }

    [Fact]
    public void ColumnsWithNothingToDoWithOneAnother_AreNotFlagged()
    {
        // scipy: 0.04895104895104895.
        var profile = Profiled(0.5, ["a", "b"], One, Other);

        Assert.Empty(RankAlerts(profile));
    }

    [Theory]
    [InlineData(0.86, 1)]
    [InlineData(0.87, 0)]
    public void TheThreshold_IsHonouredOnBothSides(double threshold, int flagged)
    {
        // scipy: Spearman's coefficient of rows and rows + 2.5 * noise is 0.8601398601398602.
        var profile = Profiled(threshold, ["a", "b"], Rows, Follows);

        Assert.Equal(flagged, RankAlerts(profile).Length);
    }

    [Fact]
    public void ACoefficientBelowNought_IsFlaggedByItsSize()
    {
        var profile = Profiled(0.95, ["a", "b"], Rows, [.. Rows.Select(value => -Math.Exp(value))]);

        var alert = Assert.Single(RankAlerts(profile));

        Assert.Contains("-1.000", alert.Says, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0.90, 1)]
    [InlineData(0.91, 0)]
    public void TiesAreRankedAsTheirAverage_SoThePairSitsWhereScipyPutsIt(double threshold, int flagged)
    {
        // scipy: 0.9007775105401477 with average ranks. The same eight rows ranked by their lowest place would be 0.9142 and
        // flag at 0.91; ranked by their highest, 0.8427.
        var profile = Profiled(threshold, ["a", "b"], [1, 2, 2, 3, 5, 5, 5, 9], [2, 1, 2, 4, 4, 6, 7, 8]);

        Assert.Equal(flagged, RankAlerts(profile).Length);
    }

    [Fact]
    public void AColumnThatNeverChanges_IsNotComparedWithAnything()
    {
        var profile = Profiled(0.1, ["a", "b", "k"], Rows, [.. Rows.Select(Math.Exp)], [.. Rows.Select(_ => 3.0)]);

        var alert = Assert.Single(RankAlerts(profile));

        Assert.DoesNotContain("k", alert.Columns);
        Assert.Contains(profile.Alerts, found => found.Answer == AlertAnswer.LeaveOut("k"));
    }

    [Fact]
    public void AColumnThatOnlyNamesItsRow_IsNotComparedEither()
    {
        // The counter is a whole number running from one: the profile already leaves it out, and it would follow any column
        // that rises.
        var profile = ProfiledAs(0.5, ["id", "b"], [ColumnKind.Integer, ColumnKind.Number], Rows, [.. Rows.Select(Math.Exp)]);

        Assert.Contains(profile.Alerts, found => found.Answer == AlertAnswer.LeaveOut("id"));
        Assert.Empty(RankAlerts(profile));
    }

    [Fact]
    public void ColumnsThatGoWithEachOtherValueForValue_AreNotSaidAgainAsRanks()
    {
        double[] repeating = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
        var profile = Profiled(0.5, ["a", "again"], repeating, repeating);

        var exact = Assert.Single(profile.Alerts, found => found.Says.Contains("goes with one value", StringComparison.Ordinal));

        Assert.Equal(["again", "a"], exact.Columns);
        Assert.Empty(RankAlerts(profile));
    }

    [Fact]
    public void TheRowsOfAPair_AreThoseWhereBothHaveANumber_AndHowManyIsSaid()
    {
        double?[] gappy = [.. Rows.Select(value => (double?)value)];
        gappy[2] = null;
        gappy[7] = null;
        var profile = ProfiledWithGaps(0.9, ["a", "b"], gappy, [.. Rows.Select(value => (double?)Math.Exp(value))]);

        var alert = Assert.Single(RankAlerts(profile));

        Assert.Contains("10 rows", alert.Says, StringComparison.Ordinal);
    }

    [Fact]
    public void AGapInTheLaterColumn_LosesThatRowForTheirPairToo()
    {
        double?[] gappy = [.. Rows.Select(value => (double?)Math.Exp(value))];
        gappy[3] = null;
        gappy[9] = null;
        var profile = ProfiledWithGaps(0.9, ["a", "b"], [.. Rows.Select(value => (double?)value)], gappy);

        var alert = Assert.Single(RankAlerts(profile));

        Assert.Contains("10 rows", alert.Says, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnWithGaps_DoesNotCostEveryOtherPairItsRows()
    {
        double?[] gappy = [.. Rows.Select(value => (double?)value)];
        gappy[0] = null;
        gappy[1] = null;
        gappy[2] = null;
        var profile = ProfiledWithGaps(0.9, ["g", "a", "b"], gappy, [.. Rows.Select(value => (double?)value)], [.. Rows.Select(value => (double?)Math.Exp(value))]);

        var pair = Assert.Single(RankAlerts(profile), found => found.Columns.Contains("a") && found.Columns.Contains("b") && !found.Columns.Contains("g"));

        Assert.Contains("12 rows", pair.Says, StringComparison.Ordinal);
    }

    [Fact]
    public void EachColumnIsFlaggedOnce_AgainstTheEarlierOneItFollowsMost()
    {
        // scipy: the third column is 0.993 from the second and 0.867 from the first, and the second is 0.860 from the first:
        // all three pairs are over the threshold, and the third is said once, against the second.
        var profile = Profiled(0.5, ["a", "b", "c"], Rows, Follows, Near);

        var alerts = RankAlerts(profile);

        Assert.Equal(2, alerts.Length);
        Assert.Equal(["b", "a"], alerts[0].Columns);
        Assert.Equal(["c", "b"], alerts[1].Columns);
        Assert.Equal(AlertAnswer.LeaveOut("c"), alerts[1].Answer);
        Assert.Contains("0.993", alerts[1].Says, StringComparison.Ordinal);
    }

    [Fact]
    public void WhenTwoEarlierColumnsAreAsNear_TheFirstOneIsTheOneSaid()
    {
        var profile = Profiled(0.5, ["a", "b", "c"], Rows, [.. Rows.Select(Math.Exp)], Follows);

        var alerts = RankAlerts(profile);

        Assert.Equal(["c", "a"], alerts[^1].Columns);
    }

    [Fact]
    public void WordsAreNotCompared_AndOnlyTheColumnsNamedAre()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b", "c", "w"], Rows.Select(value => (string[])[Text(value), Text(Math.Exp(value)), Text(Math.Exp(value)), value < 6 ? "x" : "y"]).ToArray()), "rows")
            .Declare(schema => schema.Number("a", "b", "c").Category("w"))
            .Profile(0.5, "a", "b", "w")
            .Build()
            .Run();
        var profile = (DataProfile)prepared.Evidence.Values.Single();

        var alert = Assert.Single(RankAlerts(profile));

        Assert.Equal(["b", "a"], alert.Columns);
    }

    [Fact]
    public void TheAnswer_IsNotComparedWithTheColumnsThatFollowIt()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "y", "c"], Rows.Select(value => (string[])[Text(value), Text(Math.Exp(value)), Text(Math.Exp(value) + 1)]).ToArray()), "rows")
            .Declare(schema => schema.Number("a", "y", "c"))
            .Profile(0.5)
            .SplitAtRandom(0.7, 0.15)
            .Target("y")
            .Build()
            .Run();
        var profile = (DataProfile)prepared.Evidence.Values.Single();

        var alert = Assert.Single(RankAlerts(profile));

        Assert.Equal(["c", "a"], alert.Columns);
    }

    [Fact]
    public void TheThreshold_IsDeclaredInTheFile_AndOnlyWhenThereIsOne()
    {
        var without = new ProfileStep(["a"]);
        var with = new ProfileStep(["a"], 0.9);

        Assert.Null(without.RankAbove);
        Assert.Equal(0.9, with.RankAbove);
        Assert.DoesNotContain("rankAbove", Written(without), StringComparison.Ordinal);
        Assert.Contains("\"rankAbove\":0.9", Written(with).Replace(" ", string.Empty, StringComparison.Ordinal), StringComparison.Ordinal);
        Assert.NotEqual(without, with);
        Assert.NotEqual(without.GetHashCode(), with.GetHashCode());
        Assert.Equal(new ProfileStep(["a"], 0.9), with);
        Assert.Equal(new ProfileStep(["a"], 0.9).GetHashCode(), with.GetHashCode());
        Assert.Equal(new ProfileStep(["a"], null), without);
    }

    [Fact]
    public void AProfileDeclaredBeforeTheThresholdWasThere_ReadsAndIsWrittenAsItWas()
    {
        const string file = """
            {"version": 7, "declaration": [
              {"step": "read.csv", "path": "rows.csv"},
              {"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}, {"name": "b", "kind": "number", "optional": false}]},
              {"step": "evidence.profile", "columns": ["a", "b"]}
            ]}
            """;

        var declaration = PipelineDeclaration.FromJson(file, StepCatalog.BuiltIn());
        var profile = Assert.IsType<ProfileStep>(declaration.Steps[^1]);

        Assert.Null(profile.RankAbove);
        Assert.DoesNotContain("rankAbove", declaration.ToJson(), StringComparison.Ordinal);
        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AThresholdInAFile_IsAShareAboveNought_OrTheFileIsRefused()
    {
        string Declared(string rank) => $$"""
            {"version": 8, "declaration": [
              {"step": "read.csv", "path": "rows.csv"},
              {"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}]},
              {"step": "evidence.profile", "rankAbove": {{rank}}}
            ]}
            """;

        Assert.Equal(0.75, Assert.IsType<ProfileStep>(PipelineDeclaration.FromJson(Declared("0.75"), StepCatalog.BuiltIn()).Steps[^1]).RankAbove);
        Assert.Equal(1.0, Assert.IsType<ProfileStep>(PipelineDeclaration.FromJson(Declared("1"), StepCatalog.BuiltIn()).Steps[^1]).RankAbove);

        foreach (var refused in new[] { "0", "-0.5", "1.5", "\"high\"" })
        {
            Assert.ThrowsAny<Exception>(() => PipelineDeclaration.FromJson(Declared(refused), StepCatalog.BuiltIn()));
        }
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(double.NaN)]
    public void AThreshold_ThatIsNoShareAboveNought_IsRefusedWhereItIsDeclared(double threshold)
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => new ProfileStep(["a"], threshold));

        Assert.Contains("rankAbove", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheBuilder_DeclaresTheThreshold_WherverAProfileCanStand()
    {
        var before = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b"], [["1", "2"]]), "rows")
            .Declare(schema => schema.Number("a", "b"))
            .Profile(0.8, "a", "b")
            .Declaration;
        var after = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b"], [["1", "2"]]), "rows")
            .Declare(schema => schema.Number("a", "b"))
            .SplitAtRandom(0.7, 0.15)
            .Profile(0.8)
            .Declaration;

        Assert.Equal(new ProfileStep(["a", "b"], 0.8), before.Steps[^1]);
        Assert.Equal(new ProfileStep([], 0.8), after.Steps[^1]);
    }

    [Fact]
    public void TheThresholdIsInTheReference_AndIsNotRequired()
    {
        var verb = StepCatalog.BuiltIn().Describe("evidence.profile");
        var rank = Assert.Single(verb.Parameters, each => each.Key == "rankAbove");

        Assert.Empty(rank.RequiredKeys);
        Assert.Contains("rankAbove", StepCatalog.BuiltIn().VerbReference(), StringComparison.Ordinal);
    }

    private static ProfileAlert[] RankAlerts(DataProfile profile) =>
        [.. profile.Alerts.Where(alert => alert.Says.Contains("Spearman", StringComparison.Ordinal))];

    private static string Written(ProfileStep step)
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            ((IPipelineStep)step).WriteTo(writer);
        }

        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static string Text(double value) => value.ToString("R", CultureInfo.InvariantCulture);

    private static DataProfile Profiled(double? rankAbove, string[] names, params double[][] columns) =>
        ProfiledAs(rankAbove, names, [.. columns.Select(_ => ColumnKind.Number)], columns);

    private static DataProfile ProfiledAs(double? rankAbove, string[] names, ColumnKind[] kinds, params double[][] columns) =>
        Declared(rankAbove, names, kinds, [.. columns.Select(column => column.Select(value => (double?)value).ToArray())]);

    private static DataProfile ProfiledWithGaps(double? rankAbove, string[] names, params double?[][] columns) =>
        Declared(rankAbove, names, [.. columns.Select(_ => ColumnKind.Number)], columns);

    private static DataProfile Declared(double? rankAbove, string[] names, ColumnKind[] kinds, double?[][] columns)
    {
        var count = columns[0].Length;
        string[][] rows =
        [
            .. Enumerable.Range(0, count).Select(row => columns.Select(column => column[row] is { } value ? Text(value) : string.Empty).ToArray()),
        ];
        var builder = Pdd.Create()
            .Read(new InMemoryRowSource(names, rows), "rows")
            .Declare(schema => Enumerable.Range(0, names.Length).Aggregate(schema, (each, at) => each.Optional(names[at], kinds[at])));
        var declared = rankAbove is { } threshold ? builder.Profile(threshold) : builder.Profile();

        return (DataProfile)declared.Build().Run().Evidence.Values.Single();
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A column the schema says is the id: the flock number a team keeps beside every row. It is carried with each row, as
/// read, to whatever the rows are handed to, and it is never a feature — where every column that is not an answer used to
/// become one: an id of whole numbers a feature, an id of categories one column per flock.
/// </summary>
public sealed class IdColumnTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-id-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // Forty flocks: the id is a whole number, the age is a tenth of it, so a row's id and its features can be told apart
    // and held to each other wherever the rows went.
    private static InMemoryRowSource Flocks(int count = 40, Func<int, string?>? id = null) => new(
        ["Flock", "Farm", "Age", "Weight", "When"],
        [.. Enumerable.Range(1, count).Select(row => (IReadOnlyList<string?>)
        [
            id is null ? (1000 + row).ToString(CultureInfo.InvariantCulture) : id(row),
            row % 3 == 0 ? "north" : "south",
            ((1000 + row) / 10.0).ToString(CultureInfo.InvariantCulture),
            (row * 2.5).ToString(CultureInfo.InvariantCulture),
            new DateTime(2026, 1, 1).AddDays(row).ToString("o", CultureInfo.InvariantCulture),
        ])]);

    private static Action<SchemaBuilder> WithId(ColumnKind kind = ColumnKind.Integer) =>
        schema => schema.Id("Flock", kind).Category("Farm").Number("Age", "Weight").Timestamp("When");

    // Every row of a batch carries the id it was read with: the age it was read with is a tenth of its flock number.
    private static void AssertAligned(IReadOnlyList<string> ids, IReadOnlyList<string> features, IReadOnlyList<double[]> rows)
    {
        var age = features.ToList().IndexOf("Age");

        Assert.Equal(rows.Count, ids.Count);
        Assert.All(Enumerable.Range(0, rows.Count), row => Assert.Equal(double.Parse(ids[row], CultureInfo.InvariantCulture) / 10, rows[row][age], 9));
    }

    [Fact]
    public void TheIdPart_IsWrittenOnlyWhereItIsSaid_AndReadBackAsIt()
    {
        var declaration = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).Declaration;
        var json = declaration.ToJson();
        var columns = JsonDocument.Parse(json).RootElement.GetProperty("declaration")[1].GetProperty("columns");

        Assert.True(columns[0].GetProperty("id").GetBoolean());
        Assert.All(Enumerable.Range(1, 4), at => Assert.False(columns[at].TryGetProperty("id", out _)));
        Assert.Equal(declaration, PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
        Assert.True(((DeclareStep)declaration.Steps[1]).Columns[0].Id);
        Assert.DoesNotContain("\"id\"", Pdd.Create().Read(Flocks(), "flocks").Declare(schema => schema.Integer("Flock")).Declaration.ToJson(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(ColumnKind.Number)]
    [InlineData(ColumnKind.Boolean)]
    [InlineData(ColumnKind.Timestamp)]
    public void AnIdOfAKindThatIsMeasured_IsRefused(ColumnKind kind)
    {
        var refused = Assert.Throws<ArgumentException>(() => new DeclareStep([new ColumnDeclaration("Flock", kind, false) { Id = true }]));

        Assert.Contains("'Flock'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("id", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(ColumnKind.Integer)]
    [InlineData(ColumnKind.Category)]
    [InlineData(ColumnKind.Text)]
    public void AnIdOfWholeNumbersCategoriesOrWords_IsTaken(ColumnKind kind)
    {
        Assert.Equal("Flock", new DeclareStep([new ColumnDeclaration("Flock", kind, false) { Id = true }]).IdColumn);
    }

    [Fact]
    public void TwoIds_AreRefused_NamingBoth()
    {
        var refused = Assert.Throws<ArgumentException>(() => new DeclareStep(
        [
            new ColumnDeclaration("Flock", ColumnKind.Integer, false) { Id = true },
            new ColumnDeclaration("Farm", ColumnKind.Text, false) { Id = true },
        ]));

        Assert.Contains("'Flock'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'Farm'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIdNamedAsTheAnswer_IsRefusedByName()
    {
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).SplitAtRandom(0.6, 0.2).Target("Flock"));

        var fault = Assert.Single(refused.Faults);

        Assert.Equal("target", fault.Verb);
        Assert.Contains("'Flock'", fault.Message, StringComparison.Ordinal);
        Assert.Contains("id", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIdOfWholeNumbers_IsNeverAFeature_AndIsCarriedAsTheTextOfEachRow()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).Drop("When", "Farm").SplitAtRandom(0.6, 0.2).Target("Weight").Build().Run();

        var train = prepared.Batch(Part.Train);

        Assert.Equal(["Age"], train.FeatureNames);
        Assert.NotNull(train.Ids);
        AssertAligned(train.Ids, train.FeatureNames, train.Features);
        Assert.All(train.Ids, id => Assert.StartsWith("10", id, StringComparison.Ordinal));
    }

    [Fact]
    public void AnIdOfCategories_IsNotWrittenDownAsNumbers_ByTheStepThatEncodesEveryCategory()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId(ColumnKind.Category)).Drop("When")
            .SplitAtRandom(0.6, 0.2).EncodeCategories().Target("Weight").Build().Run();

        var train = prepared.Batch(Part.Train);

        Assert.Contains("Farm_north", train.FeatureNames);
        Assert.DoesNotContain(train.FeatureNames, name => name.StartsWith("Flock", StringComparison.Ordinal));
        AssertAligned(train.Ids!, train.FeatureNames, train.Features);
    }

    [Fact]
    public void AnEncoderOfEveryCategory_WhereTheOnlyCategoryIsTheId_HasNothingToEncode()
    {
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create().Read(Flocks(), "flocks")
            .Declare(schema => schema.Id("Flock", ColumnKind.Category).Number("Age"))
            .SplitAtRandom(0.6, 0.2)
            .EncodeCategories());

        Assert.Contains("no categories", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIdOfWords_NeedsNoEncoding_SinceItNeverReachesTheFeatures()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId(ColumnKind.Text)).Drop("When", "Farm")
            .SplitAtRandom(0.6, 0.2).Target("Weight").Build().Run();

        var test = prepared.Batch(Part.Test);

        Assert.Equal(["Age"], test.FeatureNames);
        AssertAligned(test.Ids!, test.FeatureNames, test.Features);
    }

    [Fact]
    public void AProfileOfEveryColumn_LeavesTheIdOut_AndOneThatNamesIt_ProfilesIt()
    {
        var every = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).SplitAtRandom(0.6, 0.2).Profile().Build().Run();
        var named = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).SplitAtRandom(0.6, 0.2).Profile("Flock").Build().Run();

        Assert.DoesNotContain("Flock", ((DataProfile)every.Evidence[3]).Columns.Select(column => column.Name));
        Assert.Equal(["Flock"], ((DataProfile)named.Evidence[3]).Columns.Select(column => column.Name));
    }

    [Fact]
    public void AProfileThatFlagsColumnsKeepingOneAnothersOrder_NeverFlagsTheId()
    {
        // The age is a tenth of the flock number, so by their order they are one column twice: said of a number that
        // names the row, it would be said of every running number, and a model is never handed the id to begin with.
        var plain = Pdd.Create().Read(Flocks(), "flocks")
            .Declare(schema => schema.Integer("Flock").Category("Farm").Number("Age", "Weight").Timestamp("When"))
            .SplitAtRandom(0.6, 0.2).Profile(0.9).Build().Run();
        var withId = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).SplitAtRandom(0.6, 0.2).Profile(0.9).Build().Run();

        Assert.Contains(((DataProfile)plain.Evidence[3]).Alerts, alert => alert.Says.Contains("Spearman", StringComparison.Ordinal));
        Assert.DoesNotContain(((DataProfile)withId.Evidence[3]).Alerts, alert => alert.Columns.Contains("Flock"));
    }

    [Fact]
    public void TheWarmUp_IsCountedOverEveryColumnButTheId()
    {
        // An id the first rows lack says nothing of how much history the features have yet: the warm-up is the features'.
        var prepared = Pdd.Create().Read(Flocks(id: row => row <= 3 ? null : (1000 + row).ToString(CultureInfo.InvariantCulture)), "flocks")
            .Declare(WithId())
            .OrderBy("When")
            .DropWarmUp()
            .Build()
            .Run();

        Assert.Equal(40, prepared.Table.RowCount);
    }

    [Fact]
    public void AStepThatNamesTheIdOutright_KeepsItsOwnRule()
    {
        // No special case: a scaling of a column of words is refused for the words, as it is refused for any.
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create().Read(Flocks(), "flocks")
            .Declare(WithId(ColumnKind.Text)).SplitAtRandom(0.6, 0.2).Normalise("Flock"));

        Assert.Contains("'Flock', which holds text", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheIds_StayWithTheirRows_ThroughARandomSplit_InEveryPart()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).Drop("When", "Farm").SplitAtRandom(0.5, 0.25).Target("Weight").Build().Run();

        foreach (var part in new[] { Part.Train, Part.Validation, Part.Test })
        {
            var batch = prepared.Batch(part);

            AssertAligned(batch.Ids!, batch.FeatureNames, batch.Features);
        }

        Assert.Equal(40, new[] { Part.Train, Part.Validation, Part.Test }.SelectMany(part => prepared.Batch(part).Ids!).Distinct().Count());
    }

    [Fact]
    public void TheIds_StayWithTheirRows_ThroughASplitInTime()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).Drop("Farm").SplitByTime("When", 0.5, 0.25).Drop("When").Target("Weight").Build().Run();

        var train = prepared.Batch(Part.Train);
        var test = prepared.Batch(Part.Test);

        AssertAligned(train.Ids!, train.FeatureNames, train.Features);
        Assert.Equal([.. Enumerable.Range(1001, 20).Select(id => id.ToString(CultureInfo.InvariantCulture))], train.Ids);
        Assert.Equal([.. Enumerable.Range(1031, 10).Select(id => id.ToString(CultureInfo.InvariantCulture))], test.Ids);
    }

    [Fact]
    public void TheIds_StayWithTheirRows_WhenRowsWithAGapAreDropped()
    {
        var rows = new InMemoryRowSource(
            ["Flock", "Age", "Weight"],
            [.. Enumerable.Range(1, 30).Select(row => (IReadOnlyList<string?>)
            [
                (1000 + row).ToString(CultureInfo.InvariantCulture),
                row % 4 == 0 ? null : ((1000 + row) / 10.0).ToString(CultureInfo.InvariantCulture),
                (row * 2.5).ToString(CultureInfo.InvariantCulture),
            ])]);

        var prepared = Pdd.Create().Read(rows, "flocks").Declare(schema => schema.Id("Flock", ColumnKind.Integer).Number("Age", "Weight"))
            .DropGaps("Age").SplitAtRandom(0.6, 0.2).Target("Weight").Build().Run();

        string[] ids = [.. new[] { Part.Train, Part.Validation, Part.Test }.SelectMany(part => prepared.Batch(part).Ids!)];

        Assert.Equal(23, ids.Length);
        Assert.DoesNotContain(ids, id => int.Parse(id, CultureInfo.InvariantCulture) % 4 == 0);
        Assert.All(new[] { Part.Train, Part.Validation, Part.Test }, part =>
        {
            var batch = prepared.Batch(part);
            AssertAligned(batch.Ids!, batch.FeatureNames, batch.Features);
        });
    }

    [Fact]
    public void TheIds_StayWithTheirRows_ThroughAJoinThatLeavesUnmatchedRowsOut()
    {
        File.WriteAllText(Path.Join(_folder, "planned.csv"), "Flock,Age\n1001,100.1\n1002,100.2\n1003,100.3\n1004,100.4\n1005,100.5\n1006,100.6\n1007,100.7\n1008,100.8\n");
        File.WriteAllText(Path.Join(_folder, "arrived.csv"), "Flock,Weight\n1008,8\n1006,6\n1001,1\n1003,3\n1002,2\n1007,7\n");

        var prepared = new Pipeline(
                Pdd.Create().ReadJoin("planned.csv", "arrived.csv", ["Flock"], Unmatched.Drop)
                    .Declare(schema => schema.Id("Flock", ColumnKind.Integer).Number("Age", "Weight"))
                    .SplitAtRandom(0.5, 0.25).Target("Weight").Declaration,
                rows: null,
                SourceFolder.Of(_folder))
            .Run();

        string[] ids = [.. new[] { Part.Train, Part.Validation, Part.Test }.SelectMany(part => prepared.Batch(part).Ids!)];

        Assert.Equal(["1001", "1002", "1003", "1006", "1007", "1008"], ids.Order(StringComparer.Ordinal));
        Assert.All(new[] { Part.Train, Part.Validation, Part.Test }, part =>
        {
            var batch = prepared.Batch(part);
            AssertAligned(batch.Ids!, batch.FeatureNames, batch.Features);
            Assert.All(Enumerable.Range(0, batch.RowCount), row => Assert.Equal(double.Parse(batch.Ids![row], CultureInfo.InvariantCulture) - 1000, batch.Labels![row]));
        });
    }

    [Fact]
    public void AServedBatch_CarriesTheIdsOfTheRowsHandedIn_InTheOrderItHandsThemOver()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks").Declare(WithId()).OrderBy("When").Drop("When", "Farm").SplitAtRandom(0.6, 0.2).Target("Weight").Build().Run();
        var reversed = Flocks(5);
        var handedIn = new InMemoryRowSource(reversed.ColumnNames, reversed.Rows.Reverse());

        var served = prepared.Served(handedIn);

        // The rows are put back in the declared order, so the served row each id belongs to is found by where it was handed in.
        Assert.Equal(["1001", "1002", "1003", "1004", "1005"], served.Ids);
        Assert.Equal([4, 3, 2, 1, 0], served.HandedInAt);
        Assert.All(Enumerable.Range(0, 5), row => Assert.Equal(handedIn.Rows.ElementAt(served.HandedInAt[row])[0], served.Ids![row]));
        AssertAligned(served.Ids!, served.FeatureNames, served.Features);
    }

    [Fact]
    public void APipelineThatSaysNoId_HandsNoneOver()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks").Declare(schema => schema.Number("Age", "Weight")).SplitAtRandom(0.6, 0.2).Target("Weight").Build().Run();

        Assert.Null(prepared.Batch(Part.Train).Ids);
        Assert.Null(prepared.Served(Flocks(3)).Ids);
    }

    [Fact]
    public void AnIdTheSchemaAllowsToBeAbsent_IsCarriedWhereTheRowsHaveIt_AndNotWhereTheyDoNot()
    {
        var prepared = Pdd.Create().Read(Flocks(), "flocks")
            .Declare(schema => schema.Column(new ColumnDeclaration("Flock", ColumnKind.Integer, Optional: true) { Id = true }).Number("Age", "Weight"))
            .SplitAtRandom(0.6, 0.2).Target("Weight").Build().Run();

        var withoutIds = new InMemoryRowSource(["Age"], [["100.1"], ["100.2"]]);

        Assert.NotNull(prepared.Batch(Part.Train).Ids);
        Assert.Null(prepared.Served(withoutIds).Ids);
        Assert.Equal(2, prepared.Served(withoutIds).RowCount);
    }

    [Fact]
    public void AnIdThatIsAGap_IsRefusedWhereTheRowsAreHandedOver_NamingTheRow()
    {
        var prepared = Pdd.Create().Read(Flocks(id: row => row == 7 ? null : (1000 + row).ToString(CultureInfo.InvariantCulture)), "flocks")
            .Declare(WithId()).Drop("When", "Farm").Build().Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Undivided));

        Assert.Contains("Row 7", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'Flock'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnIsMadeTheId_OrNoLonger_ByTheSchemasOwnOperation()
    {
        var schema = new DeclareStep([new ColumnDeclaration("Flock", ColumnKind.Integer, false), new ColumnDeclaration("Farm", ColumnKind.Text, false)]);

        var made = schema.WithColumnId("Flock", id: true);

        Assert.True(made.Columns[0].Id);
        Assert.Equal("Flock", made.IdColumn);
        Assert.Same(made, made.WithColumnId("Flock", id: true));
        Assert.Null(made.WithColumnId("Flock", id: false).IdColumn);
        Assert.Throws<ArgumentException>(() => made.WithColumnId("Farm", id: true));
        Assert.Throws<ArgumentException>(() => schema.WithColumnId("Barn", id: true));
        Assert.Null(schema.IdColumn);
    }

    [Fact]
    public void AnExcludedId_IsNoId()
    {
        var schema = new DeclareStep([new ColumnDeclaration("Flock", ColumnKind.Integer, false) { Id = true, Excluded = true }, new ColumnDeclaration("Age", ColumnKind.Number, false)]);

        Assert.Null(schema.IdColumn);
        Assert.Null(new Pipeline(new PipelineDeclaration([new ReadRowsStep("flocks"), schema]), Flocks()).Run().Batch(Part.Undivided).Ids);
    }
}

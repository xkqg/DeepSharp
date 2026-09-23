// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Microsoft.Data.Analysis;
using Microsoft.Data.Sqlite;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The second reader, and the one that proves the seam is real: a data frame is nothing like a file, and
/// the pipeline cannot tell the difference. Everything a frame can be filled from — a query, a file, rows
/// somebody already had — comes in through this one door, so the long tail of formats stays somebody
/// else's solved problem.
/// </summary>
public class DataFrameSourceTests
{
    private static DataFrame Frame()
    {
        var age = new PrimitiveDataFrameColumn<double>("age", [22.0, null, 35.0]);
        var sex = new StringDataFrameColumn("sex", ["male", "female", "female"]);
        var fare = new PrimitiveDataFrameColumn<double>("fare", [7.25, 71.28, 8.05]);

        return new DataFrame(age, sex, fare);
    }

    [Fact]
    public void AFrameIsReadLikeAnyOtherRows()
    {
        var source = new DataFrameRowSource(Frame());

        Assert.Equal(["age", "sex", "fare"], source.ColumnNames);
        Assert.Equal(3, source.Rows.Count());
        Assert.Equal("male", source.Rows.First()[1]);
    }

    [Fact]
    public void AnAbsentValueStaysAbsent()
    {
        // The obvious shortcut is to hand back a not-a-number, which is what a plotting reader does because
        // a not-a-number means "do not draw". Here it would mean "arithmetic went wrong", and the pipeline
        // would fill a column with a number nobody measured.
        var source = new DataFrameRowSource(Frame());

        Assert.Null(source.Rows.ElementAt(1)[0]);
    }

    [Fact]
    public void AWholePipelineRunsOffAFrame()
    {
        var prepared = Pdd.Create()
            .ReadDataFrame(Frame())
            .Declare(schema => schema.Optional("age", ColumnKind.Number).Text("sex").Number("fare"))
            .SplitAtRandom(0.60, 0.20, seed: 3)
            .FillMissing("age", With.Median)
            .Encode("sex")
            .Normalise("fare")
            .Build()
            .Run();

        Assert.Equal(3, prepared.Table.RowCount);
        Assert.Equal(1, ((Column<double>)prepared.Table["age_was_missing"])[1]);
        Assert.True(prepared.Table.Has("sex_male") || prepared.Table.Has("sex_female"));
    }

    [Fact]
    public void TheFileSaysTheRowsWereHandedIn_AndSaysSoAgainWhenItIsRun()
    {
        var declaration = Pdd.Create()
            .ReadDataFrame(Frame())
            .Declare(schema => schema.Number("fare"))
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal("read.rows", returned.Steps[0].Verb);

        // Loaded from a file, the rows are not in it — which is the truth, said out loud rather than by a
        // null reference three steps later.
        var refused = Assert.Throws<InvalidOperationException>(() => new Pipeline(returned).Run());

        Assert.Contains("handed in", refused.Message);
        Assert.Contains("data frame", refused.Message);
    }

    [Fact]
    public void AndTheSameDeclarationRunsOnRowsHandedInLater()
    {
        var declaration = Pdd.Create()
            .ReadDataFrame(Frame())
            .Declare(schema => schema.Number("fare"))
            .Declaration;

        var prepared = new Pipeline(PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()))
            .Run(new DataFrameRowSource(Frame()));

        Assert.Equal(3, prepared.Table.RowCount);
    }

    [Fact]
    public void AFrameReadFromAFile_IsTheSameDataAsTheReaderOfOurOwn()
    {
        var path = Repository.Data("titanic.csv");

        var ours = Pdd.Create()
            .ReadCsv(path)
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .Build()
            .Prepare();

        var theirs = Pdd.Create()
            .ReadCsvFrame(path)
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .Build()
            .Prepare();

        Assert.Equal(ours.RowCount, theirs.RowCount);
        Assert.Equal(((Column<double>)ours["fare"])[0], ((Column<double>)theirs["fare"])[0]);
        Assert.Equal(((Column<long>)ours["survived"])[890], ((Column<long>)theirs["survived"])[890]);
    }

    [Fact]
    public void ReadingNothingAtAll_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => new DataFrameRowSource(null!));
        Assert.Throws<ArgumentNullException>(() => Pdd.Create().ReadDataFrame(null!));
        Assert.Throws<ArgumentNullException>(() => DataFrameSourceExtensions.ReadDataFrame(null!, Frame()));
        Assert.Throws<ArgumentNullException>(() => Pdd.Create().Read(null!, "nothing"));
        Assert.Throws<ArgumentException>(() => Pdd.Create().Read(new DataFrameRowSource(Frame()), " "));
    }

    [Fact]
    public async Task AQueryIsWaitedForOnceAndThenReadLikeAnythingElse()
    {
        // A real database rather than a stand-in for one: the question is whether a provider's reader
        // arrives as rows, and only a provider can answer that.
        await using var connection = new SqliteConnection("Data Source=:memory:");
        await connection.OpenAsync(TestContext.Current.CancellationToken);

        await using (var create = connection.CreateCommand())
        {
            create.CommandText =
                "create table passenger (fare real, sex text);"
                + "insert into passenger values (7.25, 'male'), (71.28, 'female'), (8.05, 'female');";

            await create.ExecuteNonQueryAsync(TestContext.Current.CancellationToken);
        }

        await using var query = connection.CreateCommand();
        query.CommandText = "select fare, sex from passenger";

        await using var reader = await query.ExecuteReaderAsync(TestContext.Current.CancellationToken);

        var pipeline = await Pdd.Create().ReadDbAsync(reader);

        var prepared = pipeline
            .Declare(schema => schema.Number("fare").Text("sex"))
            .SplitAtRandom(0.60, 0.20, seed: 1)
            .Encode("sex")
            .Build()
            .Run();

        Assert.Equal(3, prepared.Table.RowCount);
        Assert.Equal("read.rows", prepared.Declaration.Steps[0].Verb);
        Assert.Equal(7.25, ((Column<double>)prepared.Table["fare"])[0]);
    }

    [Fact]
    public async Task ReadingFromNoQueryAtAll_IsRefused()
    {
        await using var connection = new SqliteConnection("Data Source=:memory:");
        await connection.OpenAsync(TestContext.Current.CancellationToken);

        await using var query = connection.CreateCommand();
        query.CommandText = "select 1";

        await using var reader = await query.ExecuteReaderAsync(TestContext.Current.CancellationToken);

        await Assert.ThrowsAsync<ArgumentNullException>(() => Pdd.Create().ReadDbAsync(null!));
        await Assert.ThrowsAsync<ArgumentNullException>(
            () => DataFrameSourceExtensions.ReadDbAsync(null!, reader));
    }

    [Fact]
    public void AMomentInTimeCrossesTheSeamAsAMomentInTime()
    {
        // Everything leaves a frame as text, and a date written the short way is read back as a different
        // day on a machine set to another country. The round-trip form is the one that cannot be.
        var frame = new DataFrame(
            new PrimitiveDataFrameColumn<DateTime>("when", [new DateTime(2016, 12, 8, 14, 30, 0, DateTimeKind.Utc)]),
            new PrimitiveDataFrameColumn<double>("close", [111.5]));

        var source = new DataFrameRowSource(frame);
        var row = source.Rows.Single();

        Assert.StartsWith("2016-12-08T14:30:00", row[0], StringComparison.Ordinal);
        Assert.Equal("111.5", row[1]);
    }
}

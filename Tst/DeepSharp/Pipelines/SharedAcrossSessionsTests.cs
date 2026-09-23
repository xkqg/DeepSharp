// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What two sessions may share, and what they may not. The question is never which collection type is used
/// but which whole thing has to stay intact: one catalog read by everything in a process, one saved
/// pipeline serving many callers at once, and a table that belongs to exactly one run.
/// </summary>
public class SharedAcrossSessionsTests
{
    [Fact]
    public void OneCatalogIsReadFromEverywhereAtOnce()
    {
        var catalog = StepCatalog.BuiltIn();

        Parallel.For(0, 200, _ =>
        {
            Assert.True(catalog.Knows("read.csv"));
            Assert.False(catalog.Knows("read.avro"));
        });
    }

    [Fact]
    public void TwoPackagesRegisteringOneVerbAtOnce_LeaveExactlyOneWinner()
    {
        // The rule is not "a safe dictionary" but "one verb, one reader": whichever thread loses has to be
        // told so, rather than both believing they took the slot.
        var catalog = StepCatalog.BuiltIn();
        var refused = 0;

        Parallel.For(0, 50, _ =>
        {
            try
            {
                catalog.Register("read.avro", ReadCsvStep.ReadFrom);
            }
            catch (InvalidOperationException)
            {
                Interlocked.Increment(ref refused);
            }
        });

        Assert.Equal(49, refused);
        Assert.True(catalog.Knows("read.avro"));
    }

    [Fact]
    public void ACatalogTaughtAVerbWhileBeingRead_StaysReadable()
    {
        var catalog = StepCatalog.BuiltIn();

        Parallel.Invoke(
            () => Parallel.For(0, 100, at => catalog.Register($"read.{at}", ReadCsvStep.ReadFrom)),
            () => Parallel.For(0, 500, _ => Assert.True(catalog.Knows("declare"))));

        Assert.True(catalog.Knows("read.99"));
    }

    [Fact]
    public void OneSavedPipelineServesManyCallersAtOnce()
    {
        // This is the shape a service has: one pipeline loaded from its file, many requests. Nothing is
        // fitted again, so nothing is written to -- each call builds its own table out of its own rows.
        var trained = Pdd.Create()
            .ReadCsv(Path.Join(RepoRoot(), "Samples", "data", "titanic.csv"))
            .Declare(schema => schema.Integer("pclass").Text("sex").Optional("age", ColumnKind.Number))
            .SplitStratified("pclass", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Encode("sex")
            .Normalise("age")
            .Build()
            .Run();

        var served = PreparedData.FromJson(trained.ToJson());
        var answers = new double[100];

        Parallel.For(0, answers.Length, at =>
        {
            var row = served.Replay(new InMemoryRowSource(
                ["pclass", "sex", "age"], [["3", at % 2 == 0 ? "female" : "male", null]]));

            answers[at] = ((Column<double>)row["age"])[0]!.Value;
        });

        // Every caller gets the same number for the same row, because the fit is over and nothing learns.
        Assert.Single(answers.Distinct());
        Assert.Equal(0, trained.Table.RowCount > 0 ? 0 : 1);
    }

    [Fact]
    public void ATableBelongsToOneRun_AndAReplayNeverTouchesIt()
    {
        var trained = Pdd.Create()
            .ReadCsv(Path.Join(RepoRoot(), "Samples", "data", "titanic.csv"))
            .Declare(schema => schema.Integer("pclass"))
            .SplitAtRandom(0.70, 0.15)
            .Normalise("pclass")
            .Build()
            .Run();

        var before = ((Column<double>)trained.Table["pclass"])[0];

        trained.Replay(new InMemoryRowSource(["pclass"], [["1"]]));

        Assert.Equal(before, ((Column<double>)trained.Table["pclass"])[0]);
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "DeepSharp.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return directory!.FullName;
    }
}

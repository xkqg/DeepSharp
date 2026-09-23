// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a run has to show as proof is declared with it, before the numbers exist: a profile of the columns
/// and the rows a correlation is drawn from, at the place in the pipeline they are written. They are measured
/// on the training rows the split below will train on, never on every row, and they are output rather than
/// the pipeline — kept with the run, never written into its file, and not produced again by a replay.
/// </summary>
public class EvidenceTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema
                .Integer("survived", "pclass", "sibsp")
                .Category("sex")
                .Optional("age", ColumnKind.Number)
                .Number("fare")
                .Text("who"));

    [Fact]
    public void AnEvidenceStep_SurvivesTheFile()
    {
        var declaration = Passengers()
            .Profile()
            .Correlation(["age", "fare"], Shown.Numbers)
            .SplitStratified("survived", 0.70, 0.15)
            .Profile("age", "fare")
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal(["age", "fare"], ((ProfileStep)returned.Steps[5]).Columns);
        Assert.Empty(((ProfileStep)returned.Steps[2]).Columns);
        Assert.Equal(Shown.Numbers, ((CorrelationStep)returned.Steps[3]).Shown);
    }

    [Fact]
    public void TheProfileReadsTheTrainingValuesAFitWould()
    {
        var prepared = Passengers()
            .Profile("age")
            .SplitStratified("survived", 0.70, 0.15)
            .Normalise("age", Scale.Robust)
            .Build()
            .Run();

        var profile = (DataProfile)prepared.Evidence[2];
        var age = Assert.Single(profile.Columns);

        Assert.Equal(Standing.Train, profile.Over);
        Assert.Equal(prepared.Fitted[4].Number("centre"), age.Median);
        Assert.Equal(prepared.CountIn(Part.Train), profile.Rows);
        Assert.Equal(133, age.Gaps);
    }

    [Fact]
    public void EvidenceAboveTheSplit_MeasuresTheTrainingRowsOfTheSplitBelowIt()
    {
        var prepared = Passengers().Profile("fare").SplitStratified("survived", 0.70, 0.15).Build().Run();

        var fare = Assert.Single(((DataProfile)prepared.Evidence[2]).Columns);

        Assert.Equal(prepared.CountIn(Part.Train), fare.Rows);
        Assert.NotEqual(891, fare.Rows);
    }

    [Fact]
    public void EvidenceIsNotWrittenIntoThePipeline_NorProducedAgainByAReplay()
    {
        var prepared = Passengers().Profile().SplitStratified("survived", 0.70, 0.15).Build().Run();

        Assert.Single(prepared.Evidence);

        using (var file = JsonDocument.Parse(prepared.ToJson()))
        {
            Assert.Equal(
                ["split.stratified"],
                file.RootElement.GetProperty("fitted").EnumerateArray().Select(entry => entry.GetProperty("step").GetString()));
        }

        Assert.Empty(PreparedData.FromJson(prepared.ToJson(), StepCatalog.BuiltIn()).Evidence);

        var replayed = prepared.Replay(new InMemoryRowSource(
            ["survived", "pclass", "sibsp", "sex", "age", "fare", "who"], [["1", "3", "0", "female", "30", "7.25", "woman"]]));

        Assert.Equal(1, replayed.RowCount);
    }

    [Fact]
    public void EveryAlertInAProfile_NamesTheStepThatAnswersIt()
    {
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("a,b,c,d,w\n1,5,NaN,x,p\n,5,2,y,q\n3,5,4,x,r\n4,5,6,y,s\n"), "four rows")
            .Declare(schema => schema.Optional("a", ColumnKind.Number).Number("b", "c").Category("d").Text("w"))
            .Profile()
            .Build()
            .Run();

        var profile = (DataProfile)prepared.Evidence[2];

        Assert.Equal(Standing.Undivided, profile.Over);
        Assert.Contains(profile.Alerts, alert => alert.Column == "a" && alert.Verb == "fill.missing");
        Assert.Contains(profile.Alerts, alert => alert.Column == "b" && alert.Verb == "drop.columns");
        Assert.Contains(profile.Alerts, alert => alert.Column == "c" && alert.Verb == "fill.nan");
        Assert.Contains(profile.Alerts, alert => alert.Column == "d" && alert.Verb == "encode.categories");
        Assert.Contains(profile.Alerts, alert => alert.Column == "w" && alert.Verb == "encode");
        Assert.All(profile.Alerts, alert => Assert.False(string.IsNullOrWhiteSpace(alert.Says)));
    }

    [Fact]
    public void AProfileCountsTheRowsThatAreThereTwice()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived"), Remainder.Keep)
            .Profile()
            .SplitStratified("survived", 0.70, 0.15)
            .Build()
            .Run();

        var profile = (DataProfile)prepared.Evidence[2];

        Assert.Equal(53, profile.Duplicates.Groups);
        Assert.Equal(0, profile.Duplicates.GroupsAcrossParts);
    }

    [Fact]
    public void TheRowsACorrelationIsDrawnFrom_SayHowManyWereKept()
    {
        // A correlation needs a value in every column of a row, so the rows with a gap in any of them are left
        // out — and how many were left out is part of what it shows, beside the rule that left them out.
        var prepared = Passengers()
            .Correlation(["age", "fare", "sibsp"])
            .SplitStratified("survived", 0.70, 0.15)
            .Build()
            .Run();

        var input = (CorrelationInput)prepared.Evidence[2];

        Assert.Equal(["age", "fare", "sibsp"], input.Columns);
        Assert.Equal(prepared.CountIn(Part.Train), input.Total);
        Assert.Equal(prepared.CountIn(Part.Train) - 133, input.Kept);
        Assert.Equal(input.Kept, input.Rows.Count);
        Assert.All(input.Rows, row => Assert.Equal(3, row.Length));
        Assert.Equal("complete rows", input.Policy);
        Assert.Equal(Shown.Drawn, input.Shown);
        Assert.Equal(Standing.Train, input.Over);
    }

    [Fact]
    public void EveryRowAboveADropAndASplit_StandsWhereItWillLand()
    {
        // Rows the drop below takes stand as dropped, and a share held back to predict on as that; the profile
        // measures the training rows alone, and counts copies across every part the rows land in.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number), Remainder.Keep)
            .Profile("age")
            .DropGaps("age")
            .Predict(10)
            .SplitStratified("survived", 70, 10)
            .Build()
            .Run();

        var profile = (DataProfile)prepared.Evidence[2];

        Assert.Equal(prepared.CountIn(Part.Train), profile.Rows);
        Assert.Equal(0, Assert.Single(profile.Columns).Gaps);
        Assert.True(prepared.CountIn(Part.Predict) > 0);
    }

    [Fact]
    public void ACorrelationLeavesOutARowWithAValueThatIsNotAFiniteNumber()
    {
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("a,b\n1,2\nNaN,3\n4,5\n"), "three rows")
            .Declare(schema => schema.Number("a", "b"))
            .Correlation(["a", "b"], Shown.Numbers)
            .Build()
            .Run();

        var input = (CorrelationInput)prepared.Evidence[2];

        Assert.Equal(2, input.Kept);
        Assert.Equal(3, input.Total);
        Assert.Equal(Shown.Numbers, input.Shown);
        Assert.Throws<ArgumentNullException>(() => input.Accept<string>(null!));
    }

    [Fact]
    public void TwoEvidenceStepsThatSayTheSameThing_AreTheSameStep()
    {
        Assert.Equal(new ProfileStep(["a"]), new ProfileStep(["a"]));
        Assert.Equal(new ProfileStep(["a"]).GetHashCode(), new ProfileStep(["a"]).GetHashCode());
        Assert.NotEqual(new ProfileStep(["a"]), new ProfileStep());
        Assert.False(new ProfileStep().Equals(null));
        Assert.Empty(new ProfileStep(null).Columns);

        Assert.Equal(new CorrelationStep(["a", "b"]), new CorrelationStep(["a", "b"]));
        Assert.Equal(new CorrelationStep(["a", "b"]).GetHashCode(), new CorrelationStep(["a", "b"]).GetHashCode());
        Assert.NotEqual(new CorrelationStep(["a", "b"]), new CorrelationStep(["a", "b"], Shown.Numbers));
        Assert.False(new CorrelationStep(["a", "b"]).Equals(null));
        Assert.Throws<ArgumentNullException>(() => new ProfileStep().Produce(null!));
        Assert.Throws<ArgumentNullException>(() => new CorrelationStep(["a", "b"]).Produce(null!));
        Assert.Throws<ArgumentNullException>(() => prepared().Evidence[2].Accept<string>(null!));

        static PreparedData prepared() =>
            Pdd.Create().Read(CsvRowSource.FromText("a,b\n1,2\n"), "one row").Declare(schema => schema.Number("a", "b")).Profile().Build().Run();
    }

    [Fact]
    public void ACorrelationOfOneColumn_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new CorrelationStep(["age"]));
        Assert.Throws<ArgumentNullException>(() => new CorrelationStep(null!));
    }

    [Fact]
    public void EvidenceIsVisitedAsWhatItIs()
    {
        var prepared = Passengers().Profile().Correlation(["age", "fare"]).Build().Run();

        Assert.Equal("profile", prepared.Evidence[2].Accept(new EvidenceName()));
        Assert.Equal("correlation", prepared.Evidence[3].Accept(new EvidenceName()));
    }

    /// <summary>Says which evidence it was handed.</summary>
    private sealed class EvidenceName : IEvidenceVisitor<string>
    {
        public string Visit(DataProfile profile) => "profile";

        public string Visit(CorrelationInput correlation) => "correlation";
    }
}

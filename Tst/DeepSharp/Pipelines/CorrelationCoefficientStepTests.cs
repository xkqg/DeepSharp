// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Which coefficient a correlation shows is declared with it: Pearson's, which a file that leaves the word out has always
/// meant and which is never written, or Spearman's on the ranks. The word is optional, so a file written before it was
/// there reads, and writes, as it was.
/// </summary>
public class CorrelationCoefficientStepTests
{
    private static readonly string[] Columns = ["a", "b"];

    [Fact]
    public void Pearson_IsWhatLeavingTheWordOutMeans_AndIsNeverWritten()
    {
        var step = new CorrelationStep(Columns);

        Assert.Equal(Coefficient.Pearson, step.Coefficient);
        Assert.DoesNotContain("coefficient", TextOf(step), StringComparison.Ordinal);
        Assert.Equal(Coefficient.Pearson, new CorrelationStep(Columns, Shown.Numbers, Coefficient.Pearson).Coefficient);
        Assert.DoesNotContain("coefficient", TextOf(new CorrelationStep(Columns, Shown.Numbers, Coefficient.Pearson)), StringComparison.Ordinal);
    }

    [Fact]
    public void Spearman_IsWrittenAsItsWord_AndReadBackAsItself()
    {
        var step = new CorrelationStep(Columns, Shown.Drawn, Coefficient.Spearman);

        using var written = JsonDocument.Parse(TextOf(step));

        Assert.Equal("spearman", written.RootElement.GetProperty("coefficient").GetString());
        Assert.Equal(step, CorrelationStep.ReadFrom(written.RootElement));
        Assert.Equal(Coefficient.Spearman, CorrelationStep.ReadFrom(written.RootElement).Coefficient);
    }

    [Theory]
    [InlineData("""{"step": "evidence.correlation", "columns": ["a", "b"], "shown": "drawn"}""", Coefficient.Pearson)]
    [InlineData("""{"step": "evidence.correlation", "columns": ["a", "b"], "shown": "drawn", "coefficient": "pearson"}""", Coefficient.Pearson)]
    [InlineData("""{"step": "evidence.correlation", "columns": ["a", "b"], "shown": "numbers", "coefficient": "Spearman"}""", Coefficient.Spearman)]
    public void AFile_ReadsTheWordItHolds_AndPearsonWhenItHoldsNone(string json, Coefficient expected)
    {
        using var document = JsonDocument.Parse(json);

        Assert.Equal(expected, CorrelationStep.ReadFrom(document.RootElement).Coefficient);
    }

    [Fact]
    public void AWordThatIsNoCoefficient_IsRefusedNamingTheOnesThereAre()
    {
        using var document = JsonDocument.Parse("""{"step": "evidence.correlation", "columns": ["a", "b"], "shown": "drawn", "coefficient": "kendall"}""");

        var refused = Assert.Throws<FormatException>(() => CorrelationStep.ReadFrom(document.RootElement));

        Assert.Contains("kendall", refused.Message, StringComparison.Ordinal);
        Assert.Contains("pearson, spearman", refused.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => new CorrelationStep(Columns, Shown.Drawn, (Coefficient)9));
    }

    [Fact]
    public void ADeclarationWithNoWord_IsWrittenAsItWasBeforeTheWordWasThere()
    {
        var declaration = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b"], [["1", "2"]]), "rows")
            .Declare(schema => schema.Number("a", "b"))
            .Correlation(Columns, Shown.Numbers)
            .Declaration;
        var written = declaration.ToJson();

        Assert.DoesNotContain("coefficient", written, StringComparison.Ordinal);
        Assert.Equal(written, PipelineDeclaration.FromJson(written, StepCatalog.BuiltIn()).ToJson());
    }

    [Fact]
    public void ADeclarationOfSpearman_SurvivesTheFile()
    {
        var declaration = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b"], [["1", "2"]]), "rows")
            .Declare(schema => schema.Number("a", "b"))
            .Correlation(Columns, Shown.Numbers, Coefficient.Spearman)
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal(Coefficient.Spearman, ((CorrelationStep)returned.Steps[^1]).Coefficient);
        Assert.Contains("\"coefficient\": \"spearman\"", declaration.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void TwoCorrelationsThatDifferInTheirCoefficient_AreNotTheSameStep()
    {
        var pearson = new CorrelationStep(Columns);
        var spearman = new CorrelationStep(Columns, Shown.Drawn, Coefficient.Spearman);

        Assert.NotEqual(pearson, spearman);
        Assert.NotEqual(pearson.GetHashCode(), spearman.GetHashCode());
        Assert.Equal(new CorrelationStep(Columns, Shown.Drawn, Coefficient.Spearman), spearman);
        Assert.Equal(new CorrelationStep(Columns, Shown.Drawn, Coefficient.Spearman).GetHashCode(), spearman.GetHashCode());
    }

    [Fact]
    public void TheEvidence_CarriesTheCoefficientItWasDeclaredWith()
    {
        foreach (var coefficient in Enum.GetValues<Coefficient>())
        {
            var input = Pdd.Create()
                .Read(new InMemoryRowSource(["a", "b"], [["1", "2"], ["2", "1"], ["3", "5"]]), "rows")
                .Declare(schema => schema.Number("a", "b"))
                .Correlation(Columns, Shown.Numbers, coefficient)
                .Build()
                .Run()
                .Evidence.Values.OfType<CorrelationInput>().Single();

            Assert.Equal(coefficient, input.Coefficient);
        }
    }

    [Fact]
    public void TheWordIsOptionalInTheSchemaAndTheReference_AndSaysWhatLeavingItOutMeans()
    {
        var verb = StepCatalog.BuiltIn().Describe("evidence.correlation");
        var parameter = Assert.Single(verb.Parameters, each => each.Key == "coefficient");

        Assert.Empty(parameter.RequiredKeys);
        Assert.Contains("columns", verb.Parameters.SelectMany(each => each.RequiredKeys));
        Assert.Contains("shown", verb.Parameters.SelectMany(each => each.RequiredKeys));

        var schema = JsonNode.Parse(StepCatalog.BuiltIn().JsonSchema())!;
        var step = schema["$defs"]!["evidence.correlation"]!;

        Assert.NotNull(step["properties"]!["coefficient"]);
        Assert.Equal(["pearson", "spearman"], step["properties"]!["coefficient"]!["anyOf"]![0]!["enum"]!.AsArray().Select(item => item!.GetValue<string>()));
        Assert.DoesNotContain("coefficient", step["required"]!.AsArray().Select(item => item!.GetValue<string>()));
        Assert.Contains("one of `pearson` or `spearman`; left out, pearson", StepCatalog.BuiltIn().VerbReference(), StringComparison.Ordinal);
    }

    private static string TextOf(CorrelationStep step)
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            ((IPipelineStep)step).WriteTo(writer);
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The keys evidence gained that a file may leave out: which coefficient a correlation shows, and how alike in order two
/// columns of a profile may be. The form shows what leaving each out means, and clearing or picking it takes it out of the
/// text again, so a text that never had it stays as it was.
/// </summary>
public sealed class EvidenceFormTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-evidence-form-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "rows.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static StepForm Form(Notebook notebook) => notebook.Host.GetPropertyProviders().OfType<StepForm>().Single();

    private static Task<PropertySection> SectionAsync(Notebook notebook, CellModel cell) =>
        Form(notebook).GetPropertiesSectionAsync(cell, notebook.RenderContext(cell));

    private static Task ChangeAsync(Notebook notebook, CellModel cell, string field, object? value) =>
        Form(notebook).OnPropertyChangedAsync(cell, field, value, notebook.RenderContext(cell));

    private static PropertyField Field(PropertySection section, string name) => section.Fields.Single(field => field.Name == name);

    private static IPipelineStep Step(CellModel cell) => NotebookVerbs.Catalog().ReadStep(cell.Source);

    [Fact]
    public async Task ACoefficientAFileLeavesOut_ShowsPearson_AndPearsonIsPickedByLeavingItOut()
    {
        await using var notebook = await NotebookAsync("""{"step": "evidence.correlation", "columns": ["a", "b"], "shown": "drawn"}""");
        var correlation = notebook.Scaffold.Cells[0];

        var field = Field(await SectionAsync(notebook, correlation), "coefficient");

        Assert.Equal(PropertyFieldType.Select, field.FieldType);
        Assert.Equal("pearson", field.CurrentValue);
        Assert.Equal(["pearson", "spearman"], field.Options!.Select(option => option.Value));

        await ChangeAsync(notebook, correlation, "coefficient", "spearman");
        Assert.Equal(Coefficient.Spearman, ((CorrelationStep)Step(correlation)).Coefficient);
        Assert.Contains("\"coefficient\": \"spearman\"", correlation.Source, StringComparison.Ordinal);
        Assert.Equal("spearman", Field(await SectionAsync(notebook, correlation), "coefficient").CurrentValue);

        await ChangeAsync(notebook, correlation, "coefficient", "pearson");
        Assert.Equal(Coefficient.Pearson, ((CorrelationStep)Step(correlation)).Coefficient);
        Assert.DoesNotContain("coefficient", correlation.Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task APearsonCorrelation_WrittenWithItsWord_ShowsItAndIsRewrittenWithoutIt()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "evidence.correlation", "columns": ["a", "b"], "shown": "drawn", "coefficient": "pearson"}""");
        var correlation = notebook.Scaffold.Cells[0];

        Assert.Equal("pearson", Field(await SectionAsync(notebook, correlation), "coefficient").CurrentValue);

        await ChangeAsync(notebook, correlation, "shown", "numbers");
        Assert.Equal(Shown.Numbers, ((CorrelationStep)Step(correlation)).Shown);
        Assert.DoesNotContain("coefficient", correlation.Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AThresholdAProfileMayLeaveOut_ShowsNothing_AndIsSetAndClearedLikeAShare()
    {
        await using var notebook = await NotebookAsync("""{"step": "evidence.profile"}""");
        var profile = notebook.Scaffold.Cells[0];

        Assert.Equal(string.Empty, Field(await SectionAsync(notebook, profile), "rankAbove").CurrentValue);

        await ChangeAsync(notebook, profile, "rankAbove", "0.85");
        Assert.Equal(0.85, ((ProfileStep)Step(profile)).RankAbove);
        Assert.Contains("\"rankAbove\": 0.85", profile.Source, StringComparison.Ordinal);
        Assert.Equal("0.85", Field(await SectionAsync(notebook, profile), "rankAbove").CurrentValue);

        await ChangeAsync(notebook, profile, "rankAbove", string.Empty);
        Assert.Null(((ProfileStep)Step(profile)).RankAbove);
        Assert.DoesNotContain("rankAbove", profile.Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AThresholdNoShare_IsRefused_AndTheTextStaysAsItWas()
    {
        await using var notebook = await NotebookAsync("""{"step": "evidence.profile"}""");
        var profile = notebook.Scaffold.Cells[0];
        var before = profile.Source;

        await ChangeAsync(notebook, profile, "rankAbove", "0");

        Assert.Equal(before, profile.Source);
        Assert.Contains("rankAbove", (await SectionAsync(notebook, profile)).Description, StringComparison.Ordinal);
    }
}

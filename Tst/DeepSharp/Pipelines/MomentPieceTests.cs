// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The pieces of a moment in time: each written as the word 0.8.0 wrote for it, each number a number, and the two pieces that
/// are words — a weekday and a season — never asked for as numbers, since a name parsed as a number is a run that fails in
/// the middle of a fit instead of a declaration refused where it is written.
/// </summary>
public class MomentPieceTests
{
    // A Tuesday in spring, and the last Sunday of a year: the words and the numbers each piece is.
    private static readonly Dictionary<TimePart, string[]> Pieces = new()
    {
        [TimePart.Minute] = ["37", "0"],
        [TimePart.Hour] = ["14", "0"],
        [TimePart.DayOfWeek] = ["tuesday", "sunday"],
        [TimePart.DayOfMonth] = ["5", "31"],
        [TimePart.Month] = ["3", "12"],
        [TimePart.Quarter] = ["1", "4"],
        [TimePart.Season] = ["spring", "winter"],
        [TimePart.Year] = ["2024", "2023"],
    };

    public static TheoryData<TimePart> Parts => [.. Enum.GetValues<TimePart>()];

    public static TheoryData<TimePart> PartsThatAreWords => [TimePart.DayOfWeek, TimePart.Season];

    public static TheoryData<TimePart> PartsThatAreNumbers => [.. Enum.GetValues<TimePart>().Except([TimePart.DayOfWeek, TimePart.Season])];

    [Fact]
    public void EveryPieceThereIs_HasItsWordsHere_SoAPieceAddedLater_CannotBeLeftWithoutOne() =>
        Assert.Equal(Enum.GetValues<TimePart>().Order(), Pieces.Keys.Order());

    [Theory]
    [MemberData(nameof(Parts))]
    public void AsAGroup_EveryPieceIsTheWord080WroteForIt(TimePart part)
    {
        var table = Moments();

        new TimePartsStep("when", [part]).AddTo(table);

        var written = table[$"when_{part.ToString().ToLowerInvariant()}"];

        Assert.Equal(ColumnKind.Category, written.Kind);
        Assert.Equal(Pieces[part][0], written.TextAt(0));
        Assert.Equal(Pieces[part][1], written.TextAt(1));
    }

    [Theory]
    [MemberData(nameof(PartsThatAreNumbers))]
    public void AsNumbers_EveryPieceThatIsAQuantityIsTheNumberItIs(TimePart part)
    {
        var table = Moments();

        new TimePartsStep("when", [part], asCategories: false).AddTo(table);

        var written = (Column<double>)table[$"when_{part.ToString().ToLowerInvariant()}"];

        Assert.Equal(ColumnKind.Number, written.Kind);
        Assert.Equal(double.Parse(Pieces[part][0], System.Globalization.CultureInfo.InvariantCulture), written[0]);
        Assert.Equal(double.Parse(Pieces[part][1], System.Globalization.CultureInfo.InvariantCulture), written[1]);
    }

    [Theory]
    [MemberData(nameof(PartsThatAreWords))]
    public void AWeekdayOrASeason_IsRefusedAsANumber_WhereItIsWritten_NamingWhatToUseInstead(TimePart part)
    {
        var word = part.ToString().ToLowerInvariant();

        var refused = Assert.Throws<ArgumentException>(() => new TimePartsStep("when", [TimePart.Month, part], asCategories: false));

        Assert.Contains(word, refused.Message, StringComparison.Ordinal);
        Assert.Contains("Cyclical", refused.Message, StringComparison.Ordinal);
        Assert.Contains("group", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(PartsThatAreWords))]
    public void TheChain_RefusesThemToo_BeforeAnythingRuns(TimePart part)
    {
        var chain = Pdd.Create().ReadCsv("never-opened.csv").Declare(schema => schema.Timestamp("when"));

        Assert.Throws<ArgumentException>(() => chain.TimePartsAsNumbers("when", part));
        Assert.Throws<ArgumentException>(() => chain.TimePartsAsNumbers(parts => parts.Taking(part).Of("when")));
    }

    [Theory]
    [MemberData(nameof(PartsThatAreWords))]
    public void AFileThatAsksForThemAsNumbers_IsRefusedAtTheStep_InTheFilesWords(TimePart part)
    {
        var word = part.ToString().ToLowerInvariant();
        var refused = Assert.Throws<PipelineFileException>(() => StepCatalog.BuiltIn().ReadStep(
            $$"""{"step":"feature.timeParts","column":"when","asCategories":false,"parts":["{{word}}"]}"""));

        var fault = Assert.Single(refused.Faults);

        Assert.Contains("feature.timeParts", fault.Message, StringComparison.Ordinal);
        Assert.Contains(word, fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(PartsThatAreWords))]
    public void AsAGroup_TheyAreAsAcceptedAsEver(TimePart part)
    {
        var step = new TimePartsStep("when", [part]);

        Assert.True(step.AsCategories);
        Assert.Equal([$"when_{part.ToString().ToLowerInvariant()}"], step.Categories);
    }

    private static Table Moments() => SchemaBinding.Bind(
        new DeclareStep([new ColumnDeclaration("when", ColumnKind.Timestamp, false)]),
        CsvRowSource.FromText("when\n2024-03-05T14:37:00\n2023-12-31T00:00:00\n"));
}

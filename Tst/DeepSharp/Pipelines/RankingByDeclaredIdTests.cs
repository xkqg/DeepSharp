// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A split at random, a stratified split and a shuffle all rank a row by a digest of the whole record, so a
/// column the source gains or loses moves rows between parts even though nothing about the rows themselves
/// changed. When the schema declares an id, the three now rank by that id instead, and every row of one id
/// lands together: a source column gained or lost no longer moves anyone, and a flock's own two deliveries
/// never end up in two different parts.
/// </summary>
public class RankingByDeclaredIdTests
{
    // Ten flocks, two deliveries each, with different weights so the two rows of a flock are never exact
    // copies of one another and landing together is the id rule's doing, not the existing exact-duplicate
    // rule's. "sensor" is the column a newer export has and an older one does not: present only when asked for.
    private static InMemoryRowSource Flocks(bool withSensor) =>
        new(
            withSensor ? ["flock", "weight", "sensor"] : ["flock", "weight"],
            [.. Enumerable.Range(1, 10).SelectMany(flock => Enumerable.Range(0, 2).Select(delivery =>
                (IReadOnlyList<string?>)(withSensor
                    ? [flock.ToString(), (flock * 100 + delivery).ToString(), (flock + delivery).ToString()]
                    : [flock.ToString(), (flock * 100 + delivery).ToString()])))]);

    private static PreparedData Split(bool withSensor, bool declareId, int seed = 7) =>
        Pdd.Create()
            .Read(Flocks(withSensor), "flocks")
            .Declare(schema =>
            {
                if (declareId)
                {
                    schema.Id("flock", ColumnKind.Text);
                }
                else
                {
                    schema.Text("flock");
                }

                schema.Number("weight");

                if (withSensor)
                {
                    schema.Number("sensor");
                }
            })
            .SplitAtRandom(0.6, 0.2, seed)
            .Build()
            .Run();

    [Fact]
    public void WithAnIdDeclared_AGainedSourceColumn_MovesNoRowToAnotherPart()
    {
        var withSensor = Split(withSensor: true, declareId: true);
        var withoutSensor = Split(withSensor: false, declareId: true);

        Assert.Equal(withoutSensor.Parts, withSensor.Parts);
    }

    [Fact]
    public void WithNoIdDeclared_AGainedSourceColumn_CanStillMoveARowToAnotherPart()
    {
        // The existing, unchanged behaviour: ranking by the whole record means a column the source gains is part
        // of what every row is ranked by, so gaining one can and does redeal the rows. This pins today's rule so
        // the id branch above is seen to be the one thing that changed for a declared id.
        var moved = Enumerable.Range(1, 20)
            .Select(seed => !Split(withSensor: true, declareId: false, seed).Parts
                .SequenceEqual(Split(withSensor: false, declareId: false, seed).Parts));

        Assert.Contains(true, moved);
    }

    [Fact]
    public void WithAnIdDeclared_EveryRowOfOneId_LandsInOnePart()
    {
        var prepared = Split(withSensor: true, declareId: true);

        var byFlock = Enumerable.Range(0, prepared.Table.RowCount).GroupBy(row => prepared.Table["flock"].TextAt(row));

        Assert.All(byFlock, group => Assert.Single(group.Select(row => prepared.Parts[row]).Distinct()));
    }

    [Theory]
    [InlineData(typeof(SplitAtRandomStep))]
    [InlineData(typeof(SplitStratifiedStep))]
    [InlineData(typeof(ShuffleStep))]
    public void AStepThatRanksById_WithNoIdDeclaredAboveIt_IsRefused(Type stepType)
    {
        IPipelineStep RankingStep() => stepType == typeof(SplitAtRandomStep)
            ? new SplitAtRandomStep(SplitShares.Of(70, 15), seed: 1) { ById = true }
            : stepType == typeof(SplitStratifiedStep)
                ? new SplitStratifiedStep("weight", SplitShares.Of(70, 15), seed: 1) { ById = true }
                : new ShuffleStep(seed: 1) { ById = true };

        var refused = Assert.Throws<DeclarationException>(() => new PipelineDeclaration(
        [
            new DeclareStep([new ColumnDeclaration("weight", ColumnKind.Number, Optional: false)]),
            RankingStep(),
        ]));

        Assert.Contains("ranks by the declared id", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithAnIdDeclared_ArrivingInAnotherOrder_DealsTheSameWay()
    {
        PreparedData Run(bool reversed)
        {
            var rows = Flocks(withSensor: false);

            return Pdd.Create()
                .Read(new InMemoryRowSource(rows.ColumnNames, reversed ? rows.Rows.Reverse() : rows.Rows), "flocks")
                .Declare(schema => schema.Id("flock", ColumnKind.Text).Number("weight"))
                .SplitAtRandom(0.6, 0.2, seed: 7)
                .Build()
                .Run();
        }

        // Every row of one flock lands in the same part (the rule under test), so the single distinct part per
        // flock is well-defined regardless of which physical row a run happens to read first.
        Dictionary<string, Part> PartsByFlock(PreparedData prepared) => Enumerable.Range(0, prepared.Table.RowCount)
            .GroupBy(row => prepared.Table["flock"].TextAt(row)!)
            .ToDictionary(group => group.Key, group => group.Select(row => prepared.Parts[row]).Distinct().Single());

        Assert.Equal(PartsByFlock(Run(reversed: false)), PartsByFlock(Run(reversed: true)));
    }
}

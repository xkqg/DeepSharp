// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A number in a message reads the same on every machine. Left to the machine's own culture, 7.25 is
/// written 7,25 in half of Europe and an infinity as a symbol, so the same refusal says two different things
/// depending on who reads it — and a test written on one machine fails on the next.
/// </summary>
public class CultureTests
{
    private static T Dutch<T>(Func<T> act)
    {
        var was = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("nl-NL");

            return act();
        }
        finally
        {
            CultureInfo.CurrentCulture = was;
        }
    }

    [Fact]
    public void AShapeWithNoAnswer_SaysTheValueTheInvariantWay()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a\n-0.5\n"));

        var refused = Dutch(() => Assert.Throws<InvalidOperationException>(() => new MathsStep("a", Maths.Log).AddTo(table)));

        Assert.Contains("is -0.5,", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInfinityAtTheHandover_IsCalledWhatItIsEverywhere()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["a"], [["Infinity"]]), "one row")
            .Declare(schema => schema.Number("a"))
            .Build()
            .Run();

        var refused = Dutch(() => Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Undivided)));

        Assert.Contains("(Infinity)", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInfinityFoundByTheStepThatWatchesForIt_IsCalledWhatItIsToo()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [double.NegativeInfinity])]);
        var step = new FillNaNStep("a");

        var refused = Dutch(() => Assert.Throws<InvalidOperationException>(
            () => step.ApplyTo(table, step.Fit(table, [Part.Train]))));

        Assert.Contains("(-Infinity)", refused.Message, StringComparison.Ordinal);
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A learner says which of four needs it has of the features it is handed: numbers on the scales their steps declare, every
/// feature between minus one and one, numbers of any size, or each category as its place. Each need is one value that keeps
/// its number, and what it says is written once: whether it does without scaling, whether it takes categories, whether it
/// needs one scale. A number no need is named by is refused wherever a need is stated.
/// </summary>
public class NeedsTests
{
    [Fact]
    public void TheNeeds_KeepTheirNumbers_AndTheTwoNewOnesComeAfter()
    {
        Assert.Equal(0, (int)Needs.Numbers);
        Assert.Equal(1, (int)Needs.OneScale);
        Assert.Equal(2, (int)Needs.NoScale);
        Assert.Equal(3, (int)Needs.Categories);
        Assert.Equal([Needs.Numbers, Needs.OneScale, Needs.NoScale, Needs.Categories], Enum.GetValues<Needs>());
        Assert.Equal(Needs.Numbers, default);
    }

    [Theory]
    [InlineData(Needs.Numbers, false, false, false)]
    [InlineData(Needs.OneScale, false, false, true)]
    [InlineData(Needs.NoScale, true, false, false)]
    [InlineData(Needs.Categories, true, true, false)]
    public void EachNeed_SaysWhetherItDoesWithoutScaling_TakesCategories_OrNeedsOneScale(Needs needs, bool withoutScaling, bool categories, bool oneScale)
    {
        Assert.Equal(withoutScaling, needs.DoesWithoutScaling());
        Assert.Equal(categories, needs.TakesCategories());
        Assert.Equal(oneScale, needs.NeedsOneScale());
    }

    [Fact]
    public void ANumberNoNeedIsNamedBy_IsRefusedByEveryStatement_AndWhereverANeedIsStated()
    {
        var none = (Needs)42;
        var prepared = Passengers();
        var row = new InMemoryRowSource(["survived", "sex", "fare"], [["1", "female", "7.25"]]);

        Assert.Equal("needs", Assert.Throws<ArgumentOutOfRangeException>(() => none.DoesWithoutScaling()).ParamName);
        Assert.Throws<ArgumentOutOfRangeException>(() => none.TakesCategories());
        Assert.Throws<ArgumentOutOfRangeException>(() => none.NeedsOneScale());
        Assert.Throws<ArgumentOutOfRangeException>(() => ((Needs)(-1)).NeedsOneScale());
        Assert.Contains("42", Assert.Throws<ArgumentOutOfRangeException>(() => prepared.Batch(Part.Train, none)).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => prepared.Served(row, none));
    }

    [Fact]
    public void ABatchHandedOverWithoutANeed_IsHandedOverForNumbers()
    {
        var prepared = Passengers();

        var plain = prepared.Batch(Part.Train);
        var numbers = prepared.Batch(Part.Train, Needs.Numbers);

        Assert.Equal(numbers.FeatureNames, plain.FeatureNames);
        Assert.Equal(numbers.Features, plain.Features);
        Assert.Equal(numbers.Keys, plain.Keys);
    }

    private static PreparedData Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Category("sex").Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Normalise("fare", Scale.MidRange)
            .Target("survived")
            .Build()
            .Run();
}

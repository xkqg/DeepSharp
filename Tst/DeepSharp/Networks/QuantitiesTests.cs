// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A setting that is "a finite number above nothing", "a finite number of at least nothing" or "a power of two" is held to it
/// by one rule each, said in one sentence each: what the setting is, and then what it must be. Every guard of a layer, a loss,
/// a schedule, an optimizer and a run's options asks that rule, so the same mistake is refused in the same words wherever it is made.
/// </summary>
public class QuantitiesTests
{
    public static TheoryData<double> NotAboveNothing =>
        [0, -0.0, -1, double.Epsilon * -1, double.NaN, double.PositiveInfinity, double.NegativeInfinity];

    public static TheoryData<double> BelowNothingOrNotANumber => [-1, double.Epsilon * -1, double.NaN, double.PositiveInfinity, double.NegativeInfinity];

    public static TheoryData<double> NotAPowerOfTwo => [0, -2, 3, 0.3, 1.5, double.NaN, double.PositiveInfinity, double.NegativeInfinity];

    [Theory]
    [InlineData(double.Epsilon)]
    [InlineData(0.001)]
    [InlineData(1)]
    [InlineData(double.MaxValue)]
    public void AFiniteNumberAboveNothing_IsGivenBack(double value)
    {
        Assert.Equal(value, value.AboveNothing("rate", "A rate"));
        Assert.True(value.IsAboveNothing());
    }

    [Theory]
    [MemberData(nameof(NotAboveNothing))]
    public void ANumberNotAboveNothing_IsRefused_NamingTheParameterAndWhatTheSettingIs(double value)
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => value.AboveNothing("rate", "A rate"));

        Assert.Equal("rate", refused.ParamName);
        Assert.Equal(value, refused.ActualValue);
        Assert.StartsWith("A rate is a finite number above nothing. (Parameter 'rate')", refused.Message, StringComparison.Ordinal);
        Assert.False(value.IsAboveNothing());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(double.Epsilon)]
    [InlineData(1)]
    [InlineData(double.MaxValue)]
    public void AFiniteNumberOfAtLeastNothing_IsGivenBack(double value)
    {
        Assert.Equal(value, value.AtLeastNothing("decay", "A decay"));
        Assert.True(value.IsAtLeastNothing());
    }

    [Theory]
    [MemberData(nameof(BelowNothingOrNotANumber))]
    public void ANumberBelowNothing_OrNotANumber_IsRefused_NamingTheParameterAndWhatTheSettingIs(double value)
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => value.AtLeastNothing("decay", "A decay"));

        Assert.Equal("decay", refused.ParamName);
        Assert.Equal(value, refused.ActualValue);
        Assert.StartsWith("A decay is a finite number of at least nothing. (Parameter 'decay')", refused.Message, StringComparison.Ordinal);
        Assert.False(value.IsAtLeastNothing());
    }

    [Fact]
    public void ANumberThatIsNotThere_PassesBothGuards_ForNothingWasSaid()
    {
        double? none = null;

        Assert.Null(none.AboveNothing("rate", "A rate"));
        Assert.Null(none.AtLeastNothing("decay", "A decay"));
    }

    [Fact]
    public void ANumberThatIsThere_IsHeldToTheSameRule_WhateverItIsHandedAs()
    {
        double? given = 0.5;
        double? nought = 0;

        Assert.Equal(0.5, given.AboveNothing("rate", "A rate"));
        Assert.Equal(0.5, given.AtLeastNothing("decay", "A decay"));
        Assert.Equal(0, nought.AtLeastNothing("decay", "A decay"));
        Assert.Throws<ArgumentOutOfRangeException>(() => nought.AboveNothing("rate", "A rate"));
        Assert.Throws<ArgumentOutOfRangeException>(() => ((double?)-1).AtLeastNothing("decay", "A decay"));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(0.5)]
    [InlineData(65536)]
    [InlineData(1.0 / 1024)]
    public void APowerOfTwo_IsGivenBack(double value)
    {
        Assert.Equal(value, value.PowerOfTwo("scale", "A scale"));
        Assert.True(value.IsPowerOfTwo());
    }

    [Theory]
    [MemberData(nameof(NotAPowerOfTwo))]
    public void ANumberThatIsNotAPowerOfTwo_IsRefused_NamingTheParameterAndWhatTheSettingIs(double value)
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => value.PowerOfTwo("scale", "A scale"));

        Assert.Equal("scale", refused.ParamName);
        Assert.StartsWith("A scale is a power of two. (Parameter 'scale')", refused.Message, StringComparison.Ordinal);
        Assert.False(value.IsPowerOfTwo());
    }

    public static TheoryData<string, Action> AboveNothingGuards => new()
    {
        { "Huber", () => _ = new Huber(0) },
        { "Orthogonal", () => _ = new Orthogonal(0) },
        { "GradientClip", () => _ = new GradientClip(0) },
        { "Optimizer rate", () => _ = new Sgd(0) },
        { "Optimizer epsilon", () => _ = new Adam { Epsilon = 0 } },
        { "FitOptions.ClipValue", () => _ = new FitOptions(1) { ClipValue = 0 } },
        { "SlotRules.MaxNorm", () => _ = new SlotRules { MaxNorm = 0 } },
        { "SlotRules.RateShare", () => _ = new SlotRules { RateShare = -1 } },
        { "LowRank", () => _ = new LowRank(2, 0) },
        { "Poisson", () => _ = new Poisson(true) { Epsilon = 0 } },
        { "Rate schedule", () => new ConstantRate().RateAt(0, 0) },
    };

    [Theory]
    [MemberData(nameof(AboveNothingGuards))]
    public void EveryGuardOfANumberAboveNothing_SaysItInTheSameWords(string guard, Action make)
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(make);

        Assert.True(
            refused.Message.Contains(" is a finite number above nothing. (Parameter '", StringComparison.Ordinal),
            $"{guard} said: {refused.Message}");
    }

    public static TheoryData<string, Action> AtLeastNothingGuards => new()
    {
        { "LeakyRelu", () => _ = new LeakyRelu(-1) },
        { "Elu", () => _ = new Elu(-1) },
        { "Focal", () => _ = new Focal(-1) },
        { "GaussianNoise", () => _ = new GaussianNoise(-1, 1) },
        { "CosineDecay", () => _ = new CosineDecay(3, -1) },
        { "SlotRules.L1", () => _ = new SlotRules { L1 = -1 } },
        { "SlotRules.L2", () => _ = new SlotRules { L2 = -1 } },
        { "EarlyStopping.MinDelta", () => _ = new EarlyStopping { MinDelta = -1 } },
        { "Sgd.Momentum", () => _ = new Sgd { Momentum = -1 } },
        { "Weight decay", () => _ = new Sgd { WeightDecay = -1 } },
        { "TrainingPlateau.Minimum", () => _ = new TrainingPlateau { Minimum = -1 } },
    };

    [Theory]
    [MemberData(nameof(AtLeastNothingGuards))]
    public void EveryGuardOfANumberOfAtLeastNothing_SaysItInTheSameWords(string guard, Action make)
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(make);

        Assert.True(
            refused.Message.Contains(" is a finite number of at least nothing. (Parameter '", StringComparison.Ordinal),
            $"{guard} said: {refused.Message}");
    }
}

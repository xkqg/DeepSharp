// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The rule this library exists to keep — nothing learns from the data before it has been split — is not
/// enforced by a warning or a paragraph but by which methods exist at that point in the chain. That is a
/// property of the API surface, so it is tested as one: a step that learns must be absent before the split
/// and present after it. If a later increment adds a learning verb to the wrong builder, this fails.
/// </summary>
public class ChainBarrierTests
{
    private static IEnumerable<string> PublicMethodsOf<T>() =>
        typeof(T).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                 .Select(method => method.Name);

    [Fact]
    public void BeforeTheSplit_NothingThatLearnsIsEvenOffered()
    {
        var offered = PublicMethodsOf<PipelineBuilder>().ToList();

        Assert.DoesNotContain("FillMissing", offered);
        Assert.DoesNotContain("Normalise", offered);
        Assert.DoesNotContain("Encode", offered);
    }

    [Fact]
    public void AfterTheSplit_TheyAre()
    {
        Assert.Contains("FillMissing", PublicMethodsOf<FittingBuilder>());
    }

    [Fact]
    public void TheSplitIsWhatMovesYouAcross()
    {
        // The two builders are different types on purpose: reaching the second one is only possible by
        // saying how the data is split, so "fit on everything" is not a mistake you can make and be warned
        // about later — it is a method that does not exist yet.
        Assert.Contains("SplitByTime", PublicMethodsOf<PipelineBuilder>());
        Assert.NotEqual(typeof(PipelineBuilder), typeof(FittingBuilder));

        var split = typeof(PipelineBuilder).GetMethod("SplitByTime");

        Assert.NotNull(split);
        Assert.Equal(typeof(FittingBuilder), split.ReturnType);
    }

    [Fact]
    public void AndYouCannotGoBack()
    {
        // A way back to the pre-split builder would let a caller declare a feature after the split, which
        // is the same leak from the other side: the feature would be computed over data the model is meant
        // never to have seen.
        var returns = typeof(FittingBuilder)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(method => method.ReturnType);

        Assert.DoesNotContain(typeof(PipelineBuilder), returns);
    }
}

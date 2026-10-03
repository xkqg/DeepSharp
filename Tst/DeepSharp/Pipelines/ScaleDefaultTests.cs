// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A scale that is not named lands the training rows between minus one and one, and that is written down once.
/// </summary>
/// <remarks>
/// It used to be written twice and read a third way: the chain and the step's own constructor each said
/// <c>Scale.Standard</c>, while the value a new block starts with said midrange — so the same omission meant two
/// different things depending on which door it came through, and a reader of either could not tell. A network takes
/// every feature between minus one and one, which is what nearly every pipeline here asks for, so that is the one
/// default; the other scales are named where they are wanted.
/// </remarks>
public class ScaleDefaultTests
{
    /// <summary>What the <c>normalise</c> step's own parameter says a scale is when a file leaves it out.</summary>
    private static Scale WrittenDown() =>
        Assert.IsType<OneOfParameter<Scale>>(
            Shipped.Catalog().Describe(NormaliseStep.Name).Parameters.Single(parameter => parameter.Key == "scale")).Example;

    [Fact]
    public void EveryScaleTypedOptionalParameter_IsTheOneWrittenDownDefault()
    {
        // Every door that lets a caller leave the scale out: the chain's verb and the step's own constructor, and
        // anything a package adds later. Each has to mean what the step says it means, or one of them is a trap.
        var written = WrittenDown();

        var doors = Shipped.StepAssemblies
            .SelectMany(assembly => assembly.GetExportedTypes())
            .SelectMany(type => type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)
                .Cast<MethodBase>()
                .Concat(type.GetConstructors()))
            .SelectMany(door => door.GetParameters().Select(parameter => (Door: door, Parameter: parameter)))
            .Where(each => each.Parameter.ParameterType == typeof(Scale) && each.Parameter.IsOptional)
            .ToArray();

        Assert.NotEmpty(doors);

        foreach (var (door, parameter) in doors)
        {
            Assert.Equal(
                written,
                (Scale?)parameter.DefaultValue
                    ?? throw new InvalidOperationException(
                        $"{door.DeclaringType!.Name}.{door.Name}'s '{parameter.Name}' has no default value to read."));
        }
    }

    [Fact]
    public void TheOneWrittenDownDefault_LandsTheTrainingRowsBetweenMinusOneAndOne()
    {
        // The reason it is this scale and no other: it is the one a network can be handed without a second thought.
        Assert.Equal(Form.Signed, WrittenDown().Lands());
    }

    [Fact]
    public void AScalingThatNamesNoScale_IsWrittenAsTheOneWrittenDownDefault()
    {
        // The file says what was meant, whichever door wrote it, so a step built by leaving the scale out writes the
        // same word as one that names it.
        Assert.Equal(WrittenDown(), new NormaliseStep("age").Scale);
    }
}

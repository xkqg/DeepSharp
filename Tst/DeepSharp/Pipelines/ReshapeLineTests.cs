// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One line says how many columns are pulled into another shape, and the shapes are the methods.
/// </summary>
/// <remarks>
/// A price, a volume and a count that all want a logarithm were three lines that differed in a column name, and the only
/// way to say a shape was to repeat the whole verb. The line now names the shape once and lists the columns it holds for,
/// and what reaches the declaration is what it always was: one step a column. What to call the result is a fact about one
/// column — two columns cannot both be written into one name — so it stays on the verb that takes one.
/// </remarks>
public class ReshapeLineTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema
                .Integer("survived")
                .Number("fare", "volume", "age"));

    private static MathsStep[] Reshaped(PipelineBuilder chain) =>
        [.. chain.Declaration.Steps.OfType<MathsStep>()];

    private static MethodInfo[] Offered() =>
        typeof(MathsLine).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

    [Fact]
    public void OneReshapeLine_AddsOneStepPerColumnWithTheShapeTheMethodNames()
    {
        var reshaped = Reshaped(Passengers()
            .Reshape(shape => shape
                .Log1P("fare", "volume")
                .Sqrt("age")));

        Assert.Equal(["fare", "volume", "age"], reshaped.Select(step => step.Column));
        Assert.Equal([Maths.Log1P, Maths.Log1P, Maths.Sqrt], reshaped.Select(step => step.Maths));
    }

    [Fact]
    public void EveryColumnOfTheLine_IsReshapedInPlace()
    {
        // The result keeps the column's own name, as it does for the verb that names no other.
        var reshaped = Reshaped(Passengers().Reshape(shape => shape.Abs("fare", "age")));

        Assert.Equal(["fare", "age"], reshaped.Select(step => step.Into));
    }

    [Fact]
    public void TheOneLineAndTheLinePerColumn_DeclareTheSameSteps()
    {
        // The door is the only thing that changes: what the declaration holds is step for step what it held.
        var one = Passengers()
            .Reshape(shape => shape.Log1P("fare", "volume").Sqrt("age"))
            .Declaration;

        var each = Passengers()
            .Reshape("fare", Maths.Log1P)
            .Reshape("volume", Maths.Log1P)
            .Reshape("age", Maths.Sqrt)
            .Declaration;

        Assert.Equal(each.Steps, one.Steps);
        Assert.Equal(each.ToJson(), one.ToJson());
    }

    [Fact]
    public void EveryShape_HasAMethodOfItsOwnName_ThatWritesThatShape()
    {
        // Open-closed: a shape added to the list that the line has no method for fails here, instead of being reachable
        // one column at a time only.
        foreach (var shape in Enum.GetValues<Maths>())
        {
            var method = Assert.Single(Offered(), candidate => candidate.Name == shape.ToString());

            var written = Reshaped(Passengers().Reshape(line => method.Invoke(line, [new[] { "fare" }])));

            Assert.Equal(shape, Assert.Single(written).Maths);
        }
    }

    [Fact]
    public void AResultNamedForOneColumn_IsNotOnTheLine()
    {
        // What to call the result belongs to one column, so it stays on the verb that takes one column rather than
        // becoming a name the whole group would have to share: no method takes it, as a name or as any single word.
        Assert.NotEmpty(Offered());

        Assert.DoesNotContain(
            Offered(),
            method => method.GetParameters().Any(
                parameter => parameter.Name == "into" || parameter.ParameterType == typeof(string)));
    }

    [Fact]
    public void AnEmptyReshapeLine_IsRefusedWhereItIsWritten()
    {
        // A line that reshapes nothing is a line somebody meant to finish.
        Assert.Throws<ArgumentException>(() => Passengers().Reshape(shape => { }));
    }

    [Fact]
    public void ALineWrittenAfterTheSplit_IsRefusedAsTheVerbForOneColumnIs()
    {
        // Going through the line must not be a way round the barrier the chain keeps.
        var chain = Passengers();
        chain.SplitAtRandom(train: 0.70, validation: 0.15);

        Assert.Throws<InvalidOperationException>(() => chain.Reshape("fare", Maths.Log1P));
        Assert.Throws<InvalidOperationException>(() => chain.Reshape(shape => shape.Log1P("fare")));
    }
}

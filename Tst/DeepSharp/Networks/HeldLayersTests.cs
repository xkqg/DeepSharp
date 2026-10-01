// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using static DeepSharp.Tests.Backends.Contract.LayerContract;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// Every layer a layer holds, and every one those hold, each with its path from the layer asked — walked in the order the
/// slots are, so the layer holding a slot is found by the slot's path: what a reader of another framework's numbers needs
/// to know how each is laid out, in a network written as code as much as in a stack.
/// </summary>
public class HeldLayersTests
{
    [Fact]
    public void ANetworkWrittenAsCode_ListsTheLayersItHolds_ByTheNamesItGaveThem_InTheOrderItAddedThem()
    {
        var network = new TwoLayers();

        Assert.Equal(["hidden", "tally", "out"], network.HeldLayers().Select(held => held.Path));
        Assert.All(network.HeldLayers(), held => Assert.Equal(held.Path, held.Layer.Path));
    }

    [Fact]
    public void AStack_ListsItsLayersByTheirPlace_EachFollowedAtOnceByTheLayersItHolds()
    {
        var inner = new LayerStack(new Relu(), new TwoLayers());
        var stack = new LayerStack(new Dense(Tensor.Zeros(new Shape(2, 2)), Tensor.Zeros(new Shape(2))), inner, new Tanh());

        var held = stack.HeldLayers().ToArray();

        Assert.Equal(["0", "1", "1.0", "1.1", "1.1.hidden", "1.1.tally", "1.1.out", "2"], held.Select(each => each.Path));
        Assert.Same(inner, held[1].Layer);
        Assert.Same(inner.Layers[1], held[3].Layer);
        Assert.Equal(stack.Layers, stack.HeldLayers().Where(each => !each.Path.Contains('.', StringComparison.Ordinal)).Select(each => each.Layer));
    }

    [Fact]
    public void EverySlot_IsHeldByALayerTheWalkNames_InTheOrderTheSlotsAreWalked()
    {
        var stack = new LayerStack(new BatchNorm(2), new LayerStack(new TwoLayers()), new Dense(Tensor.Zeros(new Shape(3, 1)), Tensor.Zeros(new Shape(1))));

        var holders = stack.Slots().Select(named => named.Path[..named.Path.LastIndexOf('.')]).Distinct();

        Assert.Equal(holders, stack.HeldLayers().Select(held => held.Path).Where(path => stack.Slots().Any(named => named.Path.StartsWith($"{path}.", StringComparison.Ordinal) && !named.Path[(path.Length + 1)..].Contains('.', StringComparison.Ordinal))));
    }

    [Fact]
    public void ALayerThatHoldsNoOther_ListsNone()
    {
        Assert.Empty(new Dense(Tensor.Zeros(new Shape(2, 2)), Tensor.Zeros(new Shape(2))).HeldLayers());
        Assert.Empty(new Relu().HeldLayers());
    }

    [Fact]
    public void AStacksOwnListOfItsLayers_StaysWhatItWas_BesideTheWalk()
    {
        var relu = new Relu();
        var stack = new LayerStack(relu);

        Assert.Equal([relu], stack.Layers);
        Assert.Equal([new NamedLayer("0", relu)], stack.HeldLayers());
    }

    [Fact]
    public void ANetworkOfYourOwn_MayNameAMemberLayers_AsLayerStackDoes_WithoutAWarning()
    {
        // The walk has a name no layer of anybody's is likely to hold already: a member of a layer of your own called Layers,
        // as LayerStack has one, hides nothing, so code that builds with every warning an error still builds.
        const string source = """
            using System.Collections.Generic;
            using DeepSharp.Networks;
            using DeepSharp.Tensors;
            public sealed class Mine : Network
            {
                public IReadOnlyList<Layer> Layers => [];
                protected override Tensor Compute(Tensor input, Pass pass) => input;
            }
            """;
        var compilation = CSharpCompilation.Create(
            "Mine",
            [CSharpSyntaxTree.ParseText(source, cancellationToken: TestContext.Current.CancellationToken)],
            ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator).Select(path => MetadataReference.CreateFromFile(path)),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        Assert.Empty(compilation.GetDiagnostics(TestContext.Current.CancellationToken).Where(each => each.Severity >= DiagnosticSeverity.Warning).Select(each => each.Id));
    }
}

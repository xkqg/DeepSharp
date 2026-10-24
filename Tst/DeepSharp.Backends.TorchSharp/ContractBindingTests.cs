// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tests.Backends.Contract;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What every engine is held to is held on libtorch too: a contract added to the shared suite and bound here on neither the
/// processor nor a card is a promise nobody checks on the engine most likely to break it.
/// </summary>
public class ContractBindingTests
{
    [Fact]
    public void EveryContractTheSuiteHolds_IsBoundOnTheProcessorOnLibtorch()
        => AssertBound("DeepSharp.Tests.Backends.Torch", "the processor");

    [Fact]
    public void EveryContractTheSuiteHolds_IsBoundOnTheGraphicsCardOnLibtorch() => AssertBound("DeepSharp.Tests.Backends.TorchGpu", "a graphics card");

    // A contract bound in a namespace of its own on one device of libtorch: the shared suite's abstract contracts, each with a test that hands it an engine.
    private static void AssertBound(string where, string device)
    {
        var contracts = typeof(TensorBackendContract).Assembly.GetTypes()
            .Where(type => type is { IsAbstract: true, IsNested: false } && type.Name.EndsWith("Contract", StringComparison.Ordinal))
            .ToArray();
        var bound = typeof(ContractBindingTests).Assembly.GetTypes()
            .Where(type => type is { IsAbstract: false } && type.Namespace == where && type.BaseType is { } parent && contracts.Contains(parent))
            .Select(type => type.BaseType!)
            .ToHashSet();

        var unbound = contracts.Where(contract => !bound.Contains(contract)).Select(contract => contract.Name).Order(StringComparer.Ordinal).ToArray();

        Assert.True(unbound.Length == 0, $"Held on no test of {device} on libtorch: {string.Join(", ", unbound)}.");
    }
}

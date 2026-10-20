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
    {
        var contracts = typeof(TensorBackendContract).Assembly.GetTypes()
            .Where(type => type is { IsAbstract: true, IsNested: false } && type.Name.EndsWith("Contract", StringComparison.Ordinal))
            .ToArray();
        var bound = typeof(ContractBindingTests).Assembly.GetTypes()
            .Where(type => type is { IsAbstract: false } && type.Namespace == "DeepSharp.Tests.Backends.Torch" && type.BaseType is { } parent && contracts.Contains(parent))
            .Select(type => type.BaseType!)
            .ToHashSet();

        var unbound = contracts.Where(contract => !bound.Contains(contract)).Select(contract => contract.Name).Order(StringComparer.Ordinal).ToArray();

        Assert.True(unbound.Length == 0, $"Held on no test of libtorch's: {string.Join(", ", unbound)}.");
    }
}

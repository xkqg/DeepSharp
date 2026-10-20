// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tests.Backends.Contract;

namespace DeepSharp.Tests.Backends;

/// <summary>
/// What every engine is held to is held on every engine: a contract added to the shared suite and bound on none, or on the
/// light engine alone, is a promise nobody checks on the engine that would break it. The engines the core's suite binds are the
/// light one and the tests' own native one; libtorch's suite binds its own.
/// </summary>
public class ContractBindingTests
{
    [Theory]
    [InlineData("DeepSharp.Tests.Backends.Cpu")]
    [InlineData("DeepSharp.Tests.Backends.NativeMemory")]
    public void EveryContractTheSuiteHolds_IsBoundExactlyOnceOnEveryEngineTheCoreBinds(string engine)
    {
        var contracts = typeof(TensorBackendContract).Assembly.GetTypes()
            .Where(type => type is { IsAbstract: true, IsNested: false } && type.Name.EndsWith("Contract", StringComparison.Ordinal))
            .ToArray();
        var bound = typeof(ContractBindingTests).Assembly.GetTypes()
            .Where(type => type is { IsAbstract: false } && type.Namespace == engine && type.BaseType is { } parent && contracts.Contains(parent))
            .GroupBy(type => type.BaseType!)
            .ToDictionary(group => group.Key, group => group.Count());

        var unbound = contracts.Where(contract => !bound.ContainsKey(contract)).Select(contract => contract.Name).Order(StringComparer.Ordinal).ToArray();
        var twice = bound.Where(pair => pair.Value > 1).Select(pair => pair.Key.Name).Order(StringComparer.Ordinal).ToArray();

        Assert.True(unbound.Length == 0, $"Held on no test of {engine}: {string.Join(", ", unbound)}.");
        Assert.True(twice.Length == 0, $"Held twice on {engine}: {string.Join(", ", twice)}.");
    }
}

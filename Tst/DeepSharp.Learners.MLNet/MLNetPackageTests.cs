// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Learners.MLNet;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// The half that carries ML.NET is a package of its own, so a notebook or a server that only reads a model never
/// brings it: what it may reference is a closed list, held to the same test the charts already are.
/// </summary>
public class MLNetPackageTests
{
    private static readonly Assembly Package = typeof(TrainedMLModel).Assembly;

    [Fact]
    public void MLNetReferencesTheRuntimeAndAClosedListOfPackages()
    {
        // DeepSharp (the tensor core, for TrainingData) and DeepSharp.Pipelines meet here as they already meet in the
        // charts: never with each other, each held to this list.
        string[] allowed =
        [
            "DeepSharp", "DeepSharp.Pipelines", "DeepSharp.Learners.ML",
            "Microsoft.ML", "Microsoft.ML.Core", "Microsoft.ML.Data", "Microsoft.ML.DataView", "Microsoft.ML.FastTree",
        ];

        Assert.Equal("DeepSharp.Learners.MLNet", Package.GetName().Name);
        Assert.Empty(Outside(Package, allowed));
    }

    [Fact]
    public void MLNetDoesNotReachIntoTheNetworksLearnerOrAnEngine()
    {
        // The learner seam holds one implementation a package, never two: the bare door reaches the tensor core for
        // TrainingData alone, never for the network learner beside it or an engine behind it.
        string[] forbidden = ["DeepSharp.Learners.Networks", "DeepSharp.Backends.TorchSharp"];

        Assert.Empty(forbidden.Intersect(Package.GetReferencedAssemblies().Select(assembly => assembly.Name)));
    }

    // The references of an assembly that are neither the runtime itself nor on the list.
    private static string[] Outside(Assembly assembly, IEnumerable<string> allowed)
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        return [.. assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(allowed)];
    }
}

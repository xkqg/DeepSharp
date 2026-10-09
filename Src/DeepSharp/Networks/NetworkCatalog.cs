// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// The kinds a network file may name — layers, networks written as code, losses, optimizers, learning-rate schedules — and
/// how each is rebuilt from what it wrote.
/// </summary>
/// <remarks>
/// Handed to whatever reads a network file, never found by itself: a kind this catalog does not know is refused, naming
/// the package it came from when the file says, or the nearest kind this catalog knows when it does not. A network written
/// as code is read back once it is registered, like any kind another package brings.
/// </remarks>
public sealed class NetworkCatalog
{
    private readonly Dictionary<string, Registered> _kinds = new(StringComparer.Ordinal);

    private NetworkCatalog()
    {
    }

    /// <summary>The names every kind here is registered under.</summary>
    public IEnumerable<string> Names => _kinds.Keys;

    /// <summary>A catalog of every kind this package ships.</summary>
    /// <returns>A catalog of its own, to register more kinds with.</returns>
    public static NetworkCatalog BuiltIn() => new NetworkCatalog()
        .Register<LayerStack>()
        .Register<Dense>()
        .Register<Relu>()
        .Register<Tanh>()
        .Register<Sigmoid>()
        .Register<Dropout>()
        .Register<BatchNorm>()
        .Register<LayerNorm>()
        .Register<Conv2D>()
        .Register<Conv1D>()
        .Register<Conv3D>()
        .Register<MaxPool1D>()
        .Register<MaxPool2D>()
        .Register<MaxPool3D>()
        .Register<AvgPool1D>()
        .Register<AvgPool2D>()
        .Register<AvgPool3D>()
        .Register<GlobalMaxPool1D>()
        .Register<GlobalMaxPool2D>()
        .Register<GlobalMaxPool3D>()
        .Register<GlobalAvgPool1D>()
        .Register<GlobalAvgPool2D>()
        .Register<GlobalAvgPool3D>()
        .Register<SpatialDropout1D>()
        .Register<SpatialDropout2D>()
        .Register<SpatialDropout3D>()
        .Register<Flatten>()
        .Register<Reshape>()
        .Register<MeanSquaredError>()
        .Register<CrossEntropy>()
        .Register<BinaryCrossEntropy>()
        .Register<EarthMoversDistance>()
        .Register<Sgd>()
        .Register<Adam>()
        .Register<AdamW>()
        .Register<RmsProp>()
        .Register<Nadam>()
        .Register<ConstantRate>()
        .Register<StepDecay>()
        .Register<ExponentialDecay>()
        .Register<CosineDecay>()
        .Register<LinearWarmup>();

    /// <summary>Registers a kind: a layer, a network written as code, a loss, an optimizer or a learning-rate schedule.</summary>
    /// <typeparam name="TKind">The kind.</typeparam>
    /// <returns>This catalog, so the next registration can be written after it.</returns>
    /// <exception cref="ArgumentException">A kind of that name is registered already, or the kind is none of the five.</exception>
    public NetworkCatalog Register<TKind>()
        where TKind : ISaved<TKind>
    {
        var role = typeof(Layer).IsAssignableFrom(typeof(TKind)) ? Role.Layer
            : typeof(Loss).IsAssignableFrom(typeof(TKind)) ? Role.Loss
            : typeof(Optimizer).IsAssignableFrom(typeof(TKind)) ? Role.Optimizer
            : typeof(LearningRateSchedule).IsAssignableFrom(typeof(TKind)) ? Role.Schedule
            : throw new ArgumentException($"'{TKind.Name}' is no layer, loss, optimizer or learning-rate schedule, and a network file names nothing else.", nameof(TKind));

        if (!_kinds.TryAdd(TKind.Name, new Registered(role, (settings, rebuilding) => TKind.Rebuild(settings, rebuilding))))
        {
            throw new ArgumentException($"A kind named '{TKind.Name}' is registered with this catalog already.", nameof(TKind));
        }

        return this;
    }

    /// <summary>How the kind a file names is rebuilt, where one of the given role stands.</summary>
    /// <param name="kind">The name the file gives the kind.</param>
    /// <param name="package">The package the file says the kind comes from, when it says.</param>
    /// <param name="role">What stands where the kind is written: a layer, a loss, an optimizer or a schedule.</param>
    /// <returns>The kind's own way back from what it wrote, handed its settings and the rebuilding under way.</returns>
    /// <exception cref="FormatException">The name is not one of a kind this catalog knows, or it is a kind of another role.</exception>
    internal Func<JsonElement, Rebuilding, object> Rebuilder(string kind, string? package, Role role)
    {
        if (!_kinds.TryGetValue(kind, out var registered))
        {
            throw new FormatException(package is not null
                ? $"'{kind.Quoted()}' comes from {package.Quoted()}, which this catalog does not know: register it with Register<T>() on the catalog that reads this file."
                : $"'{kind.Quoted()}' is not a kind this catalog knows. The nearest one it knows is '{Nearest(kind)}'.");
        }

        if (registered.Role != role)
        {
            throw new FormatException($"'{kind}' is {registered.Role.Named()}, and {role.Named()} stands here.");
        }

        return registered.Rebuild;
    }

    private string Nearest(string kind) => _kinds.Keys.MinBy(known => Distance(kind, known))!;

    // How many letters have to be added, taken away or changed to make one name the other.
    private static int Distance(string one, string other)
    {
        var above = Enumerable.Range(0, other.Length + 1).ToArray();

        for (var row = 1; row <= one.Length; row++)
        {
            var current = new int[other.Length + 1];
            current[0] = row;

            for (var column = 1; column <= other.Length; column++)
            {
                var changed = one[row - 1] == other[column - 1] ? 0 : 1;
                current[column] = Math.Min(Math.Min(above[column] + 1, current[column - 1] + 1), above[column - 1] + changed);
            }

            above = current;
        }

        return above[other.Length];
    }

    /// <summary>What a kind is to a network: where it stands in a file.</summary>
    internal enum Role
    {
        Layer,
        Loss,
        Optimizer,
        Schedule,
    }

    private readonly record struct Registered(Role Role, Func<JsonElement, Rebuilding, object> Rebuild);
}

/// <summary>What each role a kind plays in a network file is called.</summary>
internal static class RoleExtensions
{
    extension(NetworkCatalog.Role role)
    {
        /// <summary>The role as a refusal says it, with its article: a layer, a loss, an optimizer, a learning-rate schedule.</summary>
        /// <returns>Its name.</returns>
        public string Named() => role switch
        {
            NetworkCatalog.Role.Layer => "a layer",
            NetworkCatalog.Role.Loss => "a loss",
            NetworkCatalog.Role.Optimizer => "an optimizer",
            _ => "a learning-rate schedule",
        };
    }
}

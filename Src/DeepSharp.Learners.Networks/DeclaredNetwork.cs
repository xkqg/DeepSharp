// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The engines an application hands the names its declarations use.
/// </summary>
/// <remarks>
/// A declaration names the engine its run happens on; which engine that name stands for is the application's to say, so
/// a model declared once runs on the light engine on one machine and on libtorch on another without a word of it
/// changing. The light engine is there under its own name from the start, because it needs nothing installed; every
/// other name is given an engine here, once, by whoever starts the program.
/// </remarks>
public sealed class Engines
{
    private readonly Dictionary<string, ITensorBackend> _engines = new(StringComparer.Ordinal);

    /// <summary>Gives a name an engine.</summary>
    /// <param name="name">The name a declaration uses.</param>
    /// <param name="engine">The engine it stands for.</param>
    /// <returns>These engines, so names are given one after another.</returns>
    /// <exception cref="ArgumentException">The name is empty, or already stands for an engine.</exception>
    /// <exception cref="ArgumentNullException">There is no engine.</exception>
    public Engines Use(string name, ITensorBackend engine)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(engine);

        if (!_engines.TryAdd(name, engine))
        {
            throw new ArgumentException($"'{name}' already stands for an engine: one name, one engine.", nameof(name));
        }

        return this;
    }

    /// <summary>The engine a name stands for.</summary>
    /// <param name="name">The name.</param>
    /// <returns>The engine.</returns>
    /// <exception cref="InvalidOperationException">Nothing was given that name.</exception>
    public ITensorBackend Named(string name)
    {
        if (_engines.TryGetValue(name, out var engine))
        {
            return engine;
        }

        if (name == LearnNetworkStep.LightEngine)
        {
            return new CpuBackend();
        }

        var known = _engines.Keys.Append(LearnNetworkStep.LightEngine).Order(StringComparer.Ordinal);

        throw new InvalidOperationException(
            $"This declaration runs on '{name}', and nothing here was given that name: {string.Join(", ", known.Select(each => $"'{each}'"))}. "
            + "Hand the engine over before the run: new Engines().Use(\"" + name + "\", TorchBackend.OnCpu()), say.");
    }
}

/// <summary>
/// Training the network a pipeline declares.
/// </summary>
public static class TrainExtensions
{
    extension(Pipeline pipeline)
    {
        /// <summary>Runs the pipeline for the network it declares, and trains that network on what it hands over.</summary>
        /// <param name="engines">The engines the names in the declaration stand for; the light engine alone, unless given.</param>
        /// <returns>The network behind its pipeline: what it predicts, its file, what the run did and how the report measured it.</returns>
        /// <exception cref="ArgumentNullException">There is no pipeline.</exception>
        /// <exception cref="InvalidOperationException">
        /// The pipeline names no learner, or names an engine nothing was given; or anything the run itself refuses.
        /// </exception>
        public TrainedNetwork Train(Engines? engines = null)
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            var step = Declared(pipeline.Declaration);

            return pipeline.RunFor(step.Needs).Train(engines);
        }
    }

    extension(PreparedData prepared)
    {
        /// <summary>Trains the network a pipeline declares on the rows a run of it prepared.</summary>
        /// <param name="engines">The engines the names in the declaration stand for; the light engine alone, unless given.</param>
        /// <returns>The network behind its pipeline.</returns>
        /// <exception cref="ArgumentNullException">There is nothing prepared.</exception>
        /// <exception cref="InvalidOperationException">
        /// The pipeline names no learner, or names an engine nothing was given; or anything <see cref="CompiledNetworkExtensions.Fit"/> refuses.
        /// </exception>
        public TrainedNetwork Train(Engines? engines = null)
        {
            ArgumentNullException.ThrowIfNull(prepared);

            var step = Declared(prepared.Declaration);

            return step.Layers
                .Described()
                .Compile(step.Optimizer.Moved(), step.Loss.Judged())
                .Fit(prepared, new FitOptions(step.Seed)
                {
                    Epochs = step.Epochs,
                    BatchSize = step.Batch,
                    EarlyStopping = step.Stopping.Stops(),
                    Backend = (engines ?? new Engines()).Named(step.Engine),
                });
        }
    }

    private static LearnNetworkStep Declared(PipelineDeclaration declaration) =>
        declaration.Learner as LearnNetworkStep
        ?? throw new InvalidOperationException(
            declaration.Learner is null
                ? "This pipeline names no learner, so there is nothing here to train: declare one with .WithTorch(…) or .WithTensorflow(…)."
                : $"This pipeline names '{declaration.Learner.Verb}', which is not a network this package trains.");
}

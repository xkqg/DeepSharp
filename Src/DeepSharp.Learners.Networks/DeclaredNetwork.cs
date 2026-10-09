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

        /// <summary>
        /// Runs the pipeline for the network it declares, and trains that network on what it hands over — handing every epoch
        /// over as it ends, and stopping when asked.
        /// </summary>
        /// <param name="engines">The engines the names in the declaration stand for; the light engine alone, when nothing.</param>
        /// <param name="onEpoch">
        /// What is handed every epoch as it ends, once judged, before the next begins, on the thread the run trains on; nothing
        /// for none. One that throws abandons the run, its exception reaching the caller as it was thrown.
        /// </param>
        /// <param name="cancellation">What stops the run before its next batch or its next look at the validation rows; nothing stops it, unless said.</param>
        /// <returns>The network behind its pipeline, as <see cref="Train(Pipeline, Engines)"/> returns it.</returns>
        /// <exception cref="ArgumentNullException">There is no pipeline.</exception>
        /// <exception cref="InvalidOperationException">
        /// The pipeline names no learner, or names an engine nothing was given; or anything the run itself refuses.
        /// </exception>
        /// <exception cref="OperationCanceledException">
        /// The token was cancelled: nothing is returned, and nothing of the run is kept — every call trains a network of its own.
        /// </exception>
        /// <remarks>
        /// Handing the epochs over, and a token that never stops the run, change no number: the run trains the network
        /// <see cref="Train(Pipeline, Engines)"/> trains, to the last bit.
        /// </remarks>
        public TrainedNetwork Train(Engines? engines, Action<Epoch>? onEpoch, CancellationToken cancellation = default)
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            var step = Declared(pipeline.Declaration);

            return pipeline.RunFor(step.Needs).Train(engines, onEpoch, cancellation);
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
        public TrainedNetwork Train(Engines? engines = null) => prepared.Train(engines, onEpoch: null);

        /// <summary>
        /// Trains the network a pipeline declares on the rows a run of it prepared — handing every epoch over as it ends, and
        /// stopping when asked.
        /// </summary>
        /// <param name="engines">The engines the names in the declaration stand for; the light engine alone, when nothing.</param>
        /// <param name="onEpoch">
        /// What is handed every epoch as it ends, once judged, before the next begins, on the thread the run trains on; nothing
        /// for none. One that throws abandons the run, its exception reaching the caller as it was thrown.
        /// </param>
        /// <param name="cancellation">What stops the run before its next batch or its next look at the validation rows; nothing stops it, unless said.</param>
        /// <returns>The network behind its pipeline.</returns>
        /// <exception cref="ArgumentNullException">There is nothing prepared.</exception>
        /// <exception cref="InvalidOperationException">
        /// The pipeline names no learner, or names an engine nothing was given; or anything <see cref="CompiledNetworkExtensions.Fit"/> refuses.
        /// </exception>
        /// <exception cref="OperationCanceledException">
        /// The token was cancelled: nothing is returned, and nothing of the run is kept — every call trains a network of its own.
        /// </exception>
        public TrainedNetwork Train(Engines? engines, Action<Epoch>? onEpoch, CancellationToken cancellation = default)
        {
            ArgumentNullException.ThrowIfNull(prepared);

            return Trained(prepared, Declared(prepared.Declaration), new RunControls(engines, onEpoch, cancellation));
        }

        /// <summary>
        /// Trains the network a step declares on the rows a run of a pipeline prepared, whichever learner that pipeline names.
        /// </summary>
        /// <param name="step">The network to train: its layers, what moves them, what judges them, and how the run goes.</param>
        /// <param name="engines">The engines the name in the step stands for; the light engine alone, when nothing.</param>
        /// <param name="cancellation">What stops the run before its next batch or its next look at the validation rows; nothing stops it, unless said.</param>
        /// <returns>The network behind its pipeline.</returns>
        /// <exception cref="ArgumentNullException">There is nothing prepared, or no step.</exception>
        /// <exception cref="InvalidOperationException">The step names an engine nothing was given; or anything <see cref="CompiledNetworkExtensions.Fit"/> refuses.</exception>
        /// <exception cref="OperationCanceledException">
        /// The token was cancelled: nothing is returned, and nothing of the run is kept — every call trains a network of its own.
        /// </exception>
        /// <remarks>
        /// Every network step asks the same of the rows — every feature on one scale — so rows prepared once serve any number
        /// of networks: what the preparation learned is learned from the training rows alone and replayed, whichever network
        /// is trained behind it. A search over networks prepares the rows once for each split and trains its candidates here.
        /// </remarks>
        public TrainedNetwork Train(LearnNetworkStep step, Engines? engines = null, CancellationToken cancellation = default)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(step);

            return Trained(prepared, step, new RunControls(engines, null, cancellation));
        }
    }

    // The one place a declared network is lowered onto the loop and trained: every door above ends here.
    private static TrainedNetwork Trained(PreparedData prepared, LearnNetworkStep step, RunControls controls) =>
        step.Layers
            .Described()
            .Compile(step.Optimizer.Moved(), JudgedFor(step.Loss, prepared.Declaration.Output), step.Schedule.Scheduled())
            .Fit(prepared, new FitOptions(step.Seed)
            {
                Epochs = step.Epochs,
                BatchSize = step.Batch,
                EarlyStopping = step.Stopping.Stops(),
                GradientClip = step.Clip > 0 ? new GradientClip(step.Clip) : null,
                Backend = (controls.Engines ?? new Engines()).Named(step.Engine),
                Cancellation = controls.Cancellation,
                OnEpoch = controls.OnEpoch,
            });

    // What a call may add to a declared run without changing any number of it: the engines its name stands for, a hook that
    // hears each epoch, and a token that stops it.
    private readonly record struct RunControls(Engines? Engines, Action<Epoch>? OnEpoch, CancellationToken Cancellation);

    // The loss a declared part names, for the output it judges: a distance along an order leaves what is left of a
    // distribution's whole outside that order, when the distribution makes one.
    private static Loss JudgedFor(PartDeclaration loss, INamesTheAnswer? output) =>
        loss.Judged() is EarthMoversDistance && output is DistributionStep { Remainder: not null } ? new EarthMoversDistance(remainder: true) : loss.Judged();

    private static LearnNetworkStep Declared(PipelineDeclaration declaration) =>
        declaration.Learner as LearnNetworkStep
        ?? throw new InvalidOperationException(
            declaration.Learner is null
                ? "This pipeline names no learner, so there is nothing here to train: declare one with .WithTorch(…) or .WithTensorflow(…)."
                : $"This pipeline names '{declaration.Learner.Verb}', which is not a network this package trains.");
}

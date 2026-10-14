// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A network with what it is trained by — an optimizer, a loss, a learning-rate schedule — ready to be fitted to rows and
/// to predict: what Keras's <c>compile</c> makes.
/// </summary>
/// <remarks>
/// Only a compiled network is fitted, so nothing trains without a loss somebody named. <see cref="Fit"/> is the one loop:
/// the training rows shuffled afresh every epoch and taken a batch at a time, each batch through a recording pass of its
/// own, the optimizer's step, the validation rows looked at once an epoch and never trained on.
/// </remarks>
public sealed class CompiledNetwork
{
    private const string NotBuilt =
        "A network described in Keras's words is built at its first Fit, from the shape of the rows it learns from and the run's seed, and this one has not been fitted yet. "
        + "To hold it now, lower the description with Lower(shape, new RandomStream(seed)) — without a last activation its loss applies itself, which compiling leaves out — "
        + "and compile that: fitted under the same seed, it starts and trains as this one would.";

    private readonly Sequential? _description;
    private Network? _network;
    private Shape? _takes;

    internal CompiledNetwork(Network network, Optimizer optimizer, Loss loss, LearningRateSchedule? schedule)
        : this(optimizer, loss, schedule) => _network = network;

    internal CompiledNetwork(Sequential description, Optimizer optimizer, Loss loss, LearningRateSchedule? schedule)
        : this(optimizer, loss, schedule) => _description = description;

    private CompiledNetwork(Optimizer optimizer, Loss loss, LearningRateSchedule? schedule)
    {
        optimizer.Validate();
        Optimizer = optimizer;
        Loss = loss;
        Schedule = schedule ?? new ConstantRate();
    }

    /// <summary>The network. One described in Keras's words is built at its first fit, from the shape of the rows it learns from and the run's seed.</summary>
    /// <exception cref="InvalidOperationException">It is described in Keras's words, and has not been fitted yet.</exception>
    public Network Network => _network ?? throw new InvalidOperationException(NotBuilt);

    /// <summary>What moves its parameters.</summary>
    public Optimizer Optimizer { get; }

    /// <summary>What it is trained to bring down.</summary>
    public Loss Loss { get; }

    /// <summary>How the rate changes from epoch to epoch.</summary>
    public LearningRateSchedule Schedule { get; }

    /// <summary>Trains the network on the training rows, looking at the validation rows once an epoch.</summary>
    /// <param name="train">The rows it learns from.</param>
    /// <param name="validation">The rows it is judged by at the end of every epoch, and never trained on; nothing for none.</param>
    /// <param name="options">
    /// The seed, the epochs, the batches, early stopping, a clip of the gradients, the engine, checkpoints, a checkpoint to go
    /// on from — under the seed, the batch size, the early stopping and the clip it was taken under, on the engine it was taken on, and to as many epochs as asked.
    /// </param>
    /// <returns>What the run did, epoch by epoch.</returns>
    /// <exception cref="ArgumentException">
    /// There are no training rows; early stopping or keeping only the best checkpoints is asked for without validation rows,
    /// which is what judges an epoch; a row's answers are not ones the loss could have meant; the rows' examples are of
    /// another shape than a network described in Keras's words takes; or the checkpoint to go on from was taken under another
    /// seed, another batch size, other early stopping or another clip, or on another engine — each difference named — or of another
    /// network. Nothing is built, restored or trained before all of that is checked.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// A batch's loss, or an epoch's validation loss, is not a finite number: something upstream went wrong, and training on it
    /// would learn nothing, as judging an epoch by it would say nothing.
    /// </exception>
    /// <exception cref="OperationCanceledException">
    /// The options' <see cref="FitOptions.Cancellation"/> was cancelled, which is read before every batch and before every look
    /// at the validation rows. Nothing is returned, and no best epoch is put back: the network holds what the last batch it
    /// finished left it, and the optimizer remembers the steps it took up to there, so a fit of it again goes on from there
    /// rather than from its start. A checkpoint taken before is as it was taken; a run gone on from it is the run that never
    /// stopped.
    /// </exception>
    /// <remarks>
    /// Each epoch, once judged and its checkpoint kept, is handed to <see cref="FitOptions.OnEpoch"/> on this thread before the
    /// next begins; an exception it throws abandons the run as a cancel does, and reaches the caller as it was thrown.
    /// </remarks>
    public History Fit(TrainingData train, TrainingData? validation, FitOptions options)
    {
        ArgumentNullException.ThrowIfNull(train);
        ArgumentNullException.ThrowIfNull(options);

        if (train.Count == 0)
        {
            throw new ArgumentException("A network is trained on at least one row.", nameof(train));
        }

        var watched = validation is { Count: > 0 };

        if (!watched && (options.EarlyStopping is not null || options.Checkpoints is { BestOnly: true }))
        {
            throw new ArgumentException(
                "Early stopping and keeping only the best checkpoints judge each epoch by its validation rows, and there are none.", nameof(validation));
        }

        if (options.Average is not null && options.EarlyStopping is { RestoreBest: true })
        {
            throw new ArgumentException(
                "An average of the weights and restoring the best epoch's weights both say what the run ends holding, and are not asked for together.", nameof(options));
        }

        RequireAnswers(train);

        if (watched)
        {
            RequireAnswers(validation!);
        }

        var takes = Takes(train, validation);

        if (options.ResumeFrom?.State.Unlike(options) is { } unlike)
        {
            throw new ArgumentException(unlike, nameof(options));
        }

        var network = _network ?? _description!.Lower(takes!.Value, new RandomStream(options.Seed));

        RequireRules(network);

        if (options.ResumeFrom is { } from)
        {
            RequireSlots(network, from.State);
        }

        _network = network;
        _takes = takes;

        var loop = new Loop(this, options);
        var epochs = new List<Epoch>();
        Judgement? judgement = null;

        if (options.ResumeFrom is { } resume)
        {
            judgement = loop.Resume(resume.State, epochs, watched);
        }

        for (var number = epochs.Count; number < options.Epochs && judgement is not { Stops: true }; number++)
        {
            var rate = Schedule.RateAt(number, Optimizer.Rate);
            var loss = loop.Train(train, number, rate);
            var validationLoss = watched ? loop.Evaluate(validation!, number) : (double?)null;

            epochs.Add(new Epoch(number, loss, validationLoss, rate));

            var improved = false;

            if (validationLoss is { } judged)
            {
                judgement = judgement?.After(number, judged, options.EarlyStopping) ?? Judgement.First(number, judged);
                improved = judgement.BestEpoch == number;

                if (improved && options.EarlyStopping is { RestoreBest: true })
                {
                    judgement = judgement with { BestSlots = loop.Slots() };
                }
            }

            if (options.Checkpoints is { } checkpoints && (!checkpoints.BestOnly || improved))
            {
                checkpoints.Keep(loop.Checkpoint(epochs, judgement));
            }

            options.OnEpoch?.Invoke(epochs[^1]);
        }

        if (options.EarlyStopping is { RestoreBest: true } && judgement?.BestSlots is { } best)
        {
            loop.Restore(best);
        }

        loop.ApplyAverages();

        return new History(options.Seed, epochs, judgement?.BestEpoch, judgement is { Stops: true } ? Stopping.NoLongerImproving : Stopping.AllEpochsRan);
    }

    /// <summary>What the network predicts for the given rows: an evaluation pass, through the loss's output activation.</summary>
    /// <param name="features">The rows, as the network takes them.</param>
    /// <param name="backend">Where the arithmetic runs.</param>
    /// <returns>A row of predictions for each row: numbers, shares or probabilities, as the loss has them.</returns>
    /// <remarks>The network's own <see cref="Network.Predict"/>, through this loss.</remarks>
    public Tensor Predict(Tensor features, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(backend);

        return Network.Predict(features, Loss, backend);
    }

    // The shape of one example a network described in Keras's words takes — its input, stated; or taken from the rows of its
    // first fit and kept — with the rows held to it; nothing for a network written as code, whose layers say what they take.
    private Shape? Takes(TrainingData train, TrainingData? validation)
    {
        if (_description is null)
        {
            return null;
        }

        var takes = _takes ?? _description.InputShape ?? Example(train);

        RequireExamples(train, takes, nameof(train));

        if (validation is { Count: > 0 })
        {
            RequireExamples(validation, takes, nameof(validation));
        }

        return takes;
    }

    private static Shape Example(TrainingData rows) => new([.. rows.Features.Shape.Axes[1..]]);

    private static void RequireExamples(TrainingData rows, Shape takes, string name)
    {
        var each = Example(rows);

        if (each != takes)
        {
            throw new ArgumentException($"This network takes each example as {takes}, and was handed examples of {each}.", name);
        }
    }

    // A checkpoint goes on from where it left this network alone: the same slots by path, each of the same shape — checked
    // before a single slot is replaced, so a refused checkpoint leaves the network as it was.
    private static void RequireSlots(Network network, Resumable state)
    {
        var named = network.Slots().ToArray();

        if (named.Length != state.Slots.Count || named.Any(slot => !state.Slots.TryGetValue(slot.Path, out var kept) || kept.Shape != slot.Slot.Value.Shape))
        {
            throw new ArgumentException("The checkpoint was taken of another network: its slots are not this one's.", "options");
        }

        var held = named.Where(slot => slot.Slot is Parameter { Learns: false }).Select(slot => slot.Path).Order(StringComparer.Ordinal).ToArray();

        if (state.Held is { } was && !was.SequenceEqual(held))
        {
            throw new ArgumentException(
                $"The checkpoint was taken of a network whose {Held([.. was])}, and going on with a network whose {Held(held)} would learn otherwise.", "options");
        }
    }

    // A bound is held per unit, and a transposed convolution's numbers are not laid out by unit: the rules a network carries are
    // checked before anything is built, restored or trained.
    private static void RequireRules(Network network)
    {
        foreach (var held in network.HeldLayers().Select(each => each.Layer).Prepend(network))
        {
            if (held is TransposedConvolution { Weight.Rules.MaxNorm: not null })
            {
                throw new ArgumentException(
                    $"A bound on the numbers of a {held.GetType().Name} is not held: its kernel is laid out by the channels it takes, not by each unit it makes, so the norm of a unit is no column of it.",
                    nameof(network));
            }
        }
    }

    // Every row's answers are asked of the loss before anything is trained on them, and a refused row is named as the rows
    // name it.
    private void RequireAnswers(TrainingData rows)
    {
        var width = rows.Answers.Shape[1];
        var answers = rows.Answers.Values;

        for (var row = 0; row < rows.Count; row++)
        {
            var values = new double[width];

            for (var at = 0; at < width; at++)
            {
                values[at] = answers[(row * width) + at];
            }

            if (Loss.Refusal(values) is { } refusal)
            {
                throw new ArgumentException($"{rows.NameOf(row)} cannot be trained on by this loss: {refusal}", nameof(rows));
            }
        }
    }

    // Which parameters a refusal says are held still: the paths, or that none is.
    private static string Held(string[] paths) => paths.Length == 0 ? "parameters all learned" : $"parameters {string.Join(", ", paths.Select(path => $"'{path}'"))} were held still";

    // One run of the loop: the backend, the stream and the parameters it moves, fixed for the run.
    private sealed class Loop(CompiledNetwork compiled, FitOptions options)
    {
        private readonly ITensorBackend _backend = options.Backend;
        private readonly RandomStream _stream = new(options.Seed);
        private readonly Parameter[] _parameters = [.. compiled.Network.Parameters().Where(parameter => parameter.Learns).Distinct()];

        // The parameters the loss adds a penalty for: only those that learn.
        private readonly Parameter[] _penalised = [.. compiled.Network.Parameters().Where(parameter => parameter.Learns && parameter.Rules.Penalises).Distinct()];

        // The average of each parameter the run keeps, when it keeps one: the parameter itself at the first step, and after it
        // the last average moved a share of the way to the parameter.
        private readonly Dictionary<Parameter, Tensor> _averages = [];

        // One epoch over the training rows, shuffled afresh for it: its loss, every batch's weighted by the rows it held. A step is
        // taken from the mean of the gradients of a window of batches, cut short where the epoch ends.
        internal double Train(TrainingData train, int epoch, double rate)
        {
            var order = _stream.Draw("shuffle", epoch, 0).Permutation(train.Count);
            var total = 0d;
            Gradients? window = null;
            var counted = 0;

            for (int start = 0, step = 0; start < order.Length; start += options.BatchSize, step++)
            {
                options.Cancellation.ThrowIfCancellationRequested();

                var batch = train.Rows(order.AsSpan(start, Math.Min(options.BatchSize, order.Length - start)));
                var recording = new RecordingBackend(_backend);
                var outputs = compiled.Network.Forward(batch.Features, Pass.Training(recording, _stream, epoch, step));
                var loss = Penalised(compiled.Loss.Of(outputs, batch.Answers, recording), recording);
                var value = loss.Values[0];

                if (!float.IsFinite(value))
                {
                    throw new InvalidOperationException(string.Create(
                        CultureInfo.InvariantCulture,
                        $"The loss of batch {step + 1} of epoch {epoch + 1} is {value}, not a finite number: the rows, the rate or the network's start have driven it past what a float holds, and training on it would learn nothing."));
                }

                var gradients = recording.GradientsOf(loss, _parameters.Select(parameter => parameter.Value));

                window = window is null ? gradients : window.Added(gradients, _backend);
                counted++;
                total += (double)value * batch.Count;

                if (counted == options.StepEvery || start + options.BatchSize >= order.Length)
                {
                    Step(window.Averaged(counted, _backend), rate);
                    (window, counted) = (null, 0);
                }
            }

            return total / train.Count;
        }

        // One step from the gradients of a window: each value clipped first, then their whole norm, then the optimizer's step, then
        // the averages moved on.
        private void Step(Gradients gradients, double rate)
        {
            if (options.ClipValue is { } bound)
            {
                gradients = gradients.Clamped(_parameters, bound, _backend);
            }

            compiled.Optimizer.Step(_parameters, options.GradientClip?.Clipped(_parameters, gradients, _backend) ?? gradients, rate, _backend);

            if (options.Average is { } share)
            {
                foreach (var parameter in _parameters.Distinct())
                {
                    _averages[parameter] = _averages.TryGetValue(parameter, out var average) ? _backend.Lerped(average, parameter.Value, 1 - share) : parameter.Value;
                }
            }
        }

        // The network holds its averages in place of its weights once the run is done, the running statistics aside.
        internal void ApplyAverages()
        {
            foreach (var (parameter, average) in _averages)
            {
                parameter.Replace(average);
            }
        }

        // The loss on rows the network is judged by: evaluation passes a batch at a time, each weighted by the rows it held.
        internal double Evaluate(TrainingData rows, int epoch)
        {
            options.Cancellation.ThrowIfCancellationRequested();

            var total = 0d;
            var penalty = _penalised.Length == 0 ? 0f : Penalised(_backend.Fill(new Shape(), 0f), _backend).Values[0];

            for (var start = 0; start < rows.Count; start += options.BatchSize)
            {
                var batch = rows.Slice(start, Math.Min(options.BatchSize, rows.Count - start));
                var outputs = compiled.Network.Forward(batch.Features, Pass.Evaluation(_backend));
                var value = compiled.Loss.Of(outputs, batch.Answers, _backend).Values[0] + penalty;

                if (!float.IsFinite(value))
                {
                    throw new InvalidOperationException(string.Create(
                        CultureInfo.InvariantCulture,
                        $"The validation loss of epoch {epoch + 1} is {value}, not a finite number: the network's outputs on the validation rows have gone past what a float holds, and no epoch can be judged by it."));
                }

                total += (double)value * batch.Count;
            }

            return total / rows.Count;
        }

        // The loss with the penalty of every parameter that learns added to it: what the run reports, and what it is differentiated by.
        private Tensor Penalised(Tensor loss, ITensorBackend backend)
        {
            foreach (var parameter in _penalised)
            {
                loss = backend.Add(loss, parameter.PenaltyOver(backend)!);
            }

            return loss;
        }

        internal IReadOnlyDictionary<string, Tensor> Slots() => compiled.Network.Slots().ToDictionary(named => named.Path, named => named.Slot.Value);

        internal void Restore(IReadOnlyDictionary<string, Tensor> slots)
        {
            foreach (var named in compiled.Network.Slots())
            {
                named.Slot.Replace(slots[named.Path]);
            }
        }

        internal Checkpoint Checkpoint(List<Epoch> epochs, Judgement? judgement)
        {
            var memory = new Dictionary<string, SlotMemory>();

            foreach (var named in compiled.Network.Slots())
            {
                if (named.Slot is Parameter parameter && compiled.Optimizer.KeptOf(parameter) is { } remembered)
                {
                    memory[named.Path] = remembered;
                }
            }

            return new Checkpoint(new Resumable(options.Seed, [.. epochs], Slots(), memory)
            {
                Judgement = judgement,
                Pace = Pace.Of(options),
                Engine = Engine.Of(options.Backend),
                Averaged = options.Average is null ? null : Averaged(),
                Held = [.. compiled.Network.Slots().Where(named => named.Slot is Parameter { Learns: false }).Select(named => named.Path).Order(StringComparer.Ordinal)],
            });
        }

        // The averages the run keeps, by the path of the parameter each belongs to.
        private Dictionary<string, Tensor> Averaged() =>
            compiled.Network.Slots()
                .Where(named => named.Slot is Parameter parameter && _averages.ContainsKey(parameter))
                .ToDictionary(named => named.Path, named => _averages[(Parameter)named.Slot]);

        // Puts the network, the optimizer and the history back where the checkpoint left them — a checkpoint already found to be
        // of this run's seed, batch size, early stopping, clip and engine, as far as it records them, and of this network's slots; the
        // judgement goes on only where this run is judged too.
        internal Judgement? Resume(Resumable state, List<Epoch> epochs, bool watched)
        {
            Restore(state.Slots);

            foreach (var slot in compiled.Network.Slots())
            {
                if (slot.Slot is Parameter parameter && state.Memory.TryGetValue(slot.Path, out var remembered))
                {
                    compiled.Optimizer.PutBack(parameter, remembered);
                }
            }

            if (state.Averaged is { } averaged)
            {
                foreach (var slot in compiled.Network.Slots().Where(each => each.Slot is Parameter && averaged.ContainsKey(each.Path)))
                {
                    _averages[(Parameter)slot.Slot] = averaged[slot.Path];
                }
            }

            epochs.AddRange(state.History);

            return watched ? state.Judgement : null;
        }
    }
}

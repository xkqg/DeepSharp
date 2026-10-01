// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>The tests that count every tensor libtorch holds in the process, run alone so no other test's tensors are counted.</summary>
[CollectionDefinition(nameof(LibtorchCounted), DisableParallelization = true)]
public sealed class LibtorchCounted;

/// <summary>
/// What the engine makes lives as long as something holds it, and no longer. TorchSharp ends a tensor with whatever dispose
/// scope was open on the thread that made it; the engine takes everything it makes out of such a scope at once, so a scope
/// of the caller's ends without touching a network's slots, its optimizer's memory or a checkpoint the caller keeps — and
/// once nothing holds a tensor, the collector lets its memory go, told how many bytes each holds.
/// </summary>
[Collection(nameof(LibtorchCounted))]
public class LifetimeTests
{
    // A million floats: four megabytes of libtorch's memory, and a few dozen bytes of the collector's.
    private static readonly Shape Large = new(1024, 1024);

    [Fact]
    public void WhatItMakes_InsideADisposeScopeOfTheCallers_OutlivesTheScope_AsDoesWhatItTookIn()
    {
        var engine = TorchBackend.OnCpu();
        var rows = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);
        Tensor made;

        using (torch.NewDisposeScope())
        {
            made = engine.Add(rows, rows);
        }

        // The tensor made inside the scope reads, and the rows taken in inside it are still held for the next operation.
        Assert.Equal<float[]>([2f, 4f, 6f, 8f], made.Values.ToArray());
        Assert.Equal<float[]>([3f, 6f, 9f, 12f], engine.Add(made, rows).Values.ToArray());
    }

    [Fact]
    public void ARunInsideADisposeScopeOfTheCallers_LeavesEverySlotStanding_ToServeAndToBeWritten()
    {
        var engine = TorchBackend.OnCpu();
        var prepared = Passengers.Prepared();
        TrainedNetwork trained;

        using (torch.NewDisposeScope())
        {
            trained = new Sequential().Dense(8).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy())
                .Fit(prepared, new FitOptions(seed: 7) { Backend = engine, Epochs = 2 });
        }

        var slots = trained.Network.Slots().Select(named => named.Slot.Value).ToArray();
        var served = trained.Predict(Passengers.Served(), engine);
        var read = TrainedNetwork.FromJson(trained.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        Assert.All(slots, slot => Assert.IsType<TorchStorage>(slot.Storage));
        Assert.All(served.Answers, answer => Assert.InRange(answer[0], 0, 1));
        Assert.Equal(served.Answers, read.Predict(Passengers.Served(), engine).Answers);
    }

    [Fact]
    public void WhatItMade_IsLetGoOf_OnceNothingHoldsIt_EvenInsideADisposeScopeTheCallerKeepsOpen()
    {
        var engine = TorchBackend.OnCpu();

        Collected();
        var before = torch.Tensor.TotalCount;

        using (torch.NewDisposeScope())
        {
            MadeAndDropped(engine, 200, new Shape(256));
            Collected();

            Assert.True(torch.Tensor.TotalCount <= before, $"{torch.Tensor.TotalCount - before} of libtorch's tensors are still held.");
        }
    }

    [Fact]
    public void TheBytesItHolds_AreToldToTheCollector_SoTensorsNobodyHoldsDoNotPileUpUnseen()
    {
        // Five hundred tensors of four megabytes each, each dropped as soon as it is made, and no collection asked for: the
        // managed heap barely grows, so only the bytes the storages tell the collector about make it look.
        // What is asserted is the mechanism, not how fast the finalizer thread keeps up on a busy machine: the bytes told to
        // the collector make it collect though nothing managed asked for it, and once what those collections found is
        // finalized, few of the dropped tensors are left — without the bytes told, no collection runs and all 500 are.
        var engine = TorchBackend.OnCpu();

        Collected();
        var before = torch.Tensor.TotalCount;
        var collections = GC.CollectionCount(0);

        // Only libtorch's memory is made — engine.Fill allocates nothing managed of four megabytes — so no collection the
        // managed heap would have asked for on its own is counted.
        var most = MostAliveAtOnce(engine, 500, before);
        var collected = GC.CollectionCount(0) - collections;
        GC.WaitForPendingFinalizers();
        var left = torch.Tensor.TotalCount - before;

        Assert.True(collected > 0, $"Five hundred tensors of four megabytes were made and dropped, {most} of them alive at once, and the collector never ran.");
        Assert.True(left < 100, $"{left} of 500 dropped tensors were still alive once what the collector found was finalized.");
    }

    [Fact]
    public void ACheckpointTheCallerKeeps_OutlivesForcedCollections_AndARunGoesOnFromItBitForBit()
    {
        var engine = TorchBackend.OnCpu();
        var kept = new List<Checkpoint>();
        var rows = Sloped(96);

        var straight = Stack();
        straight.Compile(new Adam(0.01), new MeanSquaredError()).Fit(rows, validation: null, Options(engine, epochs: 6, keep: null, from: null));

        // The run that stops is dropped once it has handed its checkpoints out: whatever of it nobody holds is let go of,
        // and the checkpoints are held, every tensor in them with them.
        StoppedAfterThree(engine, rows, kept);
        Collected();

        var resumed = Stack();
        resumed.Compile(new Adam(0.01), new MeanSquaredError()).Fit(rows, validation: null, Options(engine, epochs: 6, keep: null, from: kept[^1]));

        Assert.Equal(
            straight.Slots().Select(named => named.Slot.Value.Values.ToArray()),
            resumed.Slots().Select(named => named.Slot.Value.Values.ToArray()));
    }

    // Three epochs of the run, in a frame of their own, so nothing but the checkpoints it handed out outlives it.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StoppedAfterThree(TorchBackend engine, TrainingData rows, List<Checkpoint> kept) =>
        Stack().Compile(new Adam(0.01), new MeanSquaredError()).Fit(rows, validation: null, Options(engine, epochs: 3, keep: new Checkpoints(kept.Add), from: null));

    // The most of the tensors made and dropped one after another that were alive at once, as libtorch counts them.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long MostAliveAtOnce(TorchBackend engine, int count, long before)
    {
        var most = 0L;

        for (var made = 0; made < count; made++)
        {
            engine.Fill(Large, 1f);
            most = Math.Max(most, torch.Tensor.TotalCount - before);
        }

        return most;
    }

    // Tensors made and dropped in a frame of their own, so nothing of the test's keeps them.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void MadeAndDropped(TorchBackend engine, int count, Shape shape)
    {
        for (var made = 0; made < count; made++)
        {
            engine.Add(engine.Fill(shape, 1f), Tensor.From(shape, new float[shape.Count]));
        }
    }

    // A forced collection: whatever nothing holds is found, its storage let go of, and what that left is collected too.
    private static void Collected()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    private static FitOptions Options(TorchBackend engine, int epochs, Checkpoints? keep, Checkpoint? from) => new(seed: 5)
    {
        Backend = engine,
        Epochs = epochs,
        BatchSize = 16,
        Checkpoints = keep,
        ResumeFrom = from,
    };

    private static LayerStack Stack()
    {
        var stream = new RandomStream(43);

        return new LayerStack(new Dense(3, 8, stream.Draw("initialise:0", 0, 0)), new BatchNorm(8), new Relu(), new Dropout(0.25), new Dense(8, 1, stream.Draw("initialise:4", 0, 0)));
    }

    // Rows of evenly spread inputs whose answer is twice the input's first column.
    private static TrainingData Sloped(int count)
    {
        var features = new float[count * 3];
        var answers = new float[count];

        for (var row = 0; row < count; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                features[(row * 3) + column] = MathF.Sin((row * 3) + column + 1);
            }

            answers[row] = 2 * features[row * 3];
        }

        return new TrainingData(Tensor.From(new Shape(count, 3), features), Tensor.From(new Shape(count, 1), answers));
    }
}

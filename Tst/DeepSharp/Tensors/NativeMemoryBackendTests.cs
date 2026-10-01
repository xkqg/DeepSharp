// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// An engine of the tests' own that differs from the light one in the two ways libtorch does: what it makes lives in memory
/// allocated outside .NET, let go of only when nothing holds it any more, and it adds its totals up in single precision.
/// What every engine is held to runs on it from the contract; this is what is its own — and what a run keeps of an
/// engine's memory, which only such an engine can count.
/// </summary>
public class NativeMemoryBackendTests
{
    // Two to the twenty-fourth: from here a float's next value up is two away, so a one added to it is lost.
    private const float Large = 16777216f;

    [Fact]
    public void TheEngine_SaysWhichOneItIs()
    {
        Assert.Equal("nativememory", new NativeMemoryBackend().Name);
    }

    [Fact]
    public void TheEngine_NamesItsVersionAndTheDeviceItWorksOn_AsAnEngineOfSomebodyElsesCan()
    {
        INamesItsVersionAndDevice engine = new NativeMemoryBackend();

        Assert.Equal(typeof(NativeMemoryBackend).Assembly.GetName().Version!.ToString(3), engine.Version);
        Assert.Equal("cpu", engine.Device);
    }

    [Fact]
    public void ItsTotals_AreAddedUpInSinglePrecision_AsLibtorchsAre_WhereTheLightEngineKeepsThemInDouble()
    {
        // Two to the twenty-fourth, then a one and a one: a single-precision total loses each one as it takes it in, a
        // double one keeps both — in a column's sum, in a product's inner sum, and where folded patches land on one place.
        var native = new NativeMemoryBackend();
        var light = new CpuBackend();
        var column = Tensor.From(new Shape(3, 1), [Large, 1f, 1f]);
        var row = Tensor.From(new Shape(1, 3), [Large, 1f, 1f]);
        var ones = Tensor.From(new Shape(3, 1), [1f, 1f, 1f]);

        // A window three tall walks three places down an image five tall: the middle row is covered by all three.
        var patches = Tensor.From(new Shape(3, 3), [0f, 0f, Large, 0f, 1f, 0f, 1f, 0f, 0f]);
        var images = new Shape(1, 5, 1, 1);
        var window = new Window(3, 1);

        Assert.Equal(Large, native.SumRows(column).Values[0]);
        Assert.Equal(Large + 2, light.SumRows(column).Values[0]);
        Assert.Equal(Large, native.MatMul(row, ones).Values[0]);
        Assert.Equal(Large + 2, light.MatMul(row, ones).Values[0]);
        Assert.Equal(Large, native.Fold(patches, images, window).Values[2]);
        Assert.Equal(Large + 2, light.Fold(patches, images, window).Values[2]);
    }

    [Fact]
    public void WhatItMakes_StandsOnMemoryOfItsOwn_AndIsReadBackHere()
    {
        var engine = new NativeMemoryBackend();

        var sum = engine.Add(Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]), Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]));

        Assert.Equal(4, Assert.IsType<NativeMemoryStorage>(sum.Storage).Count);
        Assert.Equal<float[]>([11f, 22f, 33f, 44f], sum.Values.ToArray());
        Assert.Equal(1, engine.Made);
    }

    [Fact]
    public void ATensorFromThisMachinesMemory_IsTakenInOnce_HoweverOftenItIsHanded()
    {
        var engine = new NativeMemoryBackend();
        var rows = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);

        engine.Add(rows, rows);
        engine.MatMul(rows, rows);
        engine.Transpose(rows);

        Assert.Equal(1, engine.TakenIn);

        engine.Add(rows, Tensor.From(new Shape(2, 2), [5f, 6f, 7f, 8f]));

        Assert.Equal(2, engine.TakenIn);
    }

    [Fact]
    public void AReshape_StandsOnTheStorageItWasHanded_WithoutACopy()
    {
        var engine = new NativeMemoryBackend();
        var made = engine.Fill(new Shape(2, 3), 0.5f);

        var laidOut = engine.Reshape(made, new Shape(3, 2));

        Assert.Same(made.Storage, laidOut.Storage);
        Assert.Equal(new Shape(3, 2), laidOut.Shape);
        Assert.Equal(1, engine.Made);
    }

    [Fact]
    public void ATensorHandedToAnotherEngine_AndBack_IsReadBackOnce_AndTakenInOnce()
    {
        var native = new NativeMemoryBackend();
        var counting = new CopyCountingBackend();
        var start = native.Fill(new Shape(2), 1.5f);

        var there = counting.Add(start, start);
        var back = native.Multiply(there, there);

        Assert.Equal<float[]>([9f, 9f], back.Values.ToArray());
        Assert.Equal(2, counting.TakenIn);
        Assert.Equal(1, native.TakenIn);
        Assert.Equal(1, Assert.IsType<CopyCountingStorage>(there.Storage).Copies);
    }

    [Fact]
    public void WhatItMade_IsLetGoOf_OnceNothingHoldsIt()
    {
        var engine = new NativeMemoryBackend();

        MadeAndDropped(engine, 1000);
        Collected();

        Assert.Equal(1000, engine.Made);
        Assert.Equal(0, engine.Live);
    }

    [Fact]
    public void WhatARunKeepsOfAnEnginesMemory_IsItsSlotsItsOptimizersMemoryItsBestEpochAndTheCheckpointsKept_AtItsTenthEpochAndItsFiftieth()
    {
        // The passenger list through the pipeline, fifty epochs of twenty steps on the tests' own native engine, the best
        // epoch kept to end on, and every epoch's checkpoint handed out — every fifth kept, every other one dropped. At the
        // tenth and the fiftieth, after a forced collection, what is still alive of what the engine made is exactly what
        // something still holds: the network's slots, what Adam remembers of each, the best epoch's slots and the
        // checkpoints kept, each with the best epoch's slots as they stood when it was taken, which a run going on from it
        // ends holding — every intermediate of a thousand steps, and every checkpoint dropped, let go of.
        var engine = new NativeMemoryBackend();
        var adam = new Adam(0.01);
        var compiled = new Sequential().Dense(16).Relu().Dense(1).Compile(adam, new BinaryCrossEntropy());
        var kept = new List<Checkpoint>();
        var standing = new Dictionary<int, Standing>();

        compiled.Fit(WikiTitanic.In(WikiTitanic.DataFolder).Run(), new FitOptions(20260929)
        {
            Backend = engine,
            Epochs = 50,
            EarlyStopping = new EarlyStopping { Patience = 50, RestoreBest = true },
            Checkpoints = new Checkpoints(checkpoint =>
            {
                if (checkpoint.Epochs % 5 == 0)
                {
                    kept.Add(checkpoint);
                }

                if (checkpoint.Epochs is 10 or 50)
                {
                    standing[checkpoint.Epochs] = Measured(checkpoint);
                }
            }),
        });

        Assert.Equal([10, 50], standing.Keys.Order());
        Assert.All(standing, pair =>
        {
            // Four slots and Adam's two running means of each are twelve tensors an epoch, all new every step: each kept
            // checkpoint holds its epoch's twelve, the last kept one the network's own; and the four slots of each best epoch
            // the run or a kept checkpoint holds are four more where that epoch's own checkpoint was dropped.
            Assert.Equal((12 * (pair.Key / 5)) + (4 * pair.Value.BestsDropped), pair.Value.Held);
            Assert.Equal(pair.Value.Held, pair.Value.Live);
        });

        Standing Measured(Checkpoint checkpoint)
        {
            Collected();

            var slots = compiled.Network.Slots().Select(named => named.Slot.Value);
            var remembered = compiled.Network.Parameters().Select(adam.MomentsOf).SelectMany(moments => new[] { moments.Average, moments.SquaredAverage });
            var judged = kept.Select(each => each.State.Judgement!).Append(checkpoint.State.Judgement!).ToArray();
            var bests = judged.SelectMany(judgement => judgement.BestSlots!.Values);
            var checkpoints = kept.SelectMany(each => each.State.Slots.Values.Concat(each.State.Memory.Values.SelectMany(memory => memory.Tensors.Values)));
            var held = slots.Concat(remembered).Concat(bests).Concat(checkpoints).Select(tensor => tensor.Storage).Distinct(ReferenceEqualityComparer.Instance);
            var bestsDropped = judged.Select(judgement => judgement.BestEpoch).Distinct().Count(epoch => (epoch + 1) % 5 != 0);

            return new Standing(engine.Live, held.Count(), bestsDropped);
        }
    }

    // A forced collection: whatever nothing holds is found, its storage let go of, and what that left is collected too.
    private static void Collected()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    // Tensors made and dropped in a frame of their own, so nothing of this method's keeps them.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void MadeAndDropped(NativeMemoryBackend engine, int count)
    {
        for (var made = 0; made < count; made++)
        {
            engine.Fill(new Shape(256), 1f);
        }
    }

    // How many storages of the engine's are alive after a forced collection, how many something still holds, and how many
    // of the best epochs held are epochs whose own checkpoint was dropped.
    private readonly record struct Standing(int Live, int Held, int BestsDropped);
}

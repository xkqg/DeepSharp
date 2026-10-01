// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Torch;
using TorchSharp;

namespace DeepSharp.Tests.Backends.TorchGpu;

/// <summary>
/// The engine on a graphics card: what it makes lives on the card and is read back here; a tensor made on the card is
/// moved to the processor's engine once, and one made on the processor to the card's; and a network trained on the card
/// serves on the processor alike. Run where the suite is built with -p:Libtorch=cuda on a machine with an NVIDIA card, and
/// skipped everywhere else.
/// </summary>
public class CardTests
{
    [Fact]
    public void WhatItMakes_LivesOnTheCard_AndIsReadBackHere()
    {
        var engine = Card.First();

        var sum = engine.Add(Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]), Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]));

        var storage = Assert.IsType<TorchStorage>(sum.Storage);
        Assert.True(storage.IsOn("cuda:0"));
        Assert.False(storage.IsOn("cpu"));
        Assert.Equal<float[]>([11f, 22f, 33f, 44f], sum.Values.ToArray());
    }

    [Fact]
    public void ATensorMadeOnTheCard_IsMovedToTheProcessorsEngineOnce_AndOneMadeThere_ToTheCards()
    {
        var card = Card.First();
        var processor = TorchBackend.OnCpu();
        var onCard = card.Fill(new Shape(2, 3), 1.5f);

        var moved = processor.Reshape(onCard, new Shape(3, 2));
        var again = processor.Reshape(onCard, new Shape(6));
        var back = card.Multiply(processor.Add(moved, moved), processor.Fill(new Shape(3, 2), 2f));

        Assert.Same(moved.Storage, again.Storage);
        Assert.True(Assert.IsType<TorchStorage>(moved.Storage).IsOn("cpu"));
        Assert.True(Assert.IsType<TorchStorage>(back.Storage).IsOn("cuda:0"));
        Assert.Equal<float[]>([6f, 6f, 6f, 6f, 6f, 6f], back.Values.ToArray());
    }

    [Fact]
    public void ACardNumberedBeyondTheOnesThere_IsRefused_NamingHowManyThereAre()
    {
        Assert.SkipUnless(Card.Present, Card.Absent);

        var cards = torch.cuda.device_count();
        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => TorchBackend.OnGpu(cards));

        Assert.Equal("index", refused.ParamName);
        Assert.Equal(Libtorch.NoCard(cards, cards).Message, refused.Message);
    }

    [Fact]
    public void ACheckpointOfARunOnTheProcessor_GoesOnOnTheCardNoMore_RefusedNamingBothDevices()
    {
        // The same libtorch on another device adds its totals up otherwise: to a checkpoint it is another engine.
        var card = Card.First();
        var processor = TorchBackend.OnCpu();
        var prepared = Passengers.Prepared();
        var compiled = new Sequential().Dense(16).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());
        var files = new List<string>();
        compiled.Fit(prepared, new FitOptions(seed: 7)
        {
            Backend = processor, Epochs = 2, Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))),
        });
        var resumed = CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), prepared);

        var wrong = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(prepared, new FitOptions(seed: 7) { Backend = card, Epochs = 4, ResumeFrom = resumed.Checkpoint }));

        Assert.Equal("cuda:0", card.Device);
        Assert.Equal(
            $"The checkpoint was taken of a run on the engine 'torch' {processor.Version} on cpu, and going on under the engine 'torch' {card.Version} on cuda:0 would round every step otherwise. (Parameter 'options')",
            wrong.Message);
    }

    [Fact]
    public void ANetworkTrainedOnTheCard_ServesOnTheProcessorAlike()
    {
        var card = Card.First();
        var trained = new Sequential().Dense(16).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(Passengers.Prepared(), new FitOptions(seed: 20260929) { Backend = card, Epochs = 5 });

        var onCard = trained.Predict(Passengers.Served(), card).Answers;
        var onProcessor = trained.Predict(Passengers.Served(), TorchBackend.OnCpu()).Answers;

        Assert.All(trained.Network.Slots(), named => Assert.True(Assert.IsType<TorchStorage>(named.Slot.Value.Storage).IsOn("cuda:0")));

        for (var row = 0; row < onCard.Count; row++)
        {
            Assert.True(
                Math.Abs(onCard[row][0] - onProcessor[row][0]) <= 1e-5 + (1.3e-6 * Math.Abs(onProcessor[row][0])),
                string.Create(CultureInfo.InvariantCulture, $"row {row}: {onCard[row][0]} on the card against {onProcessor[row][0]}"));
        }
    }
}

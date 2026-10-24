// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the checkpoint DeepSharp 0.14.0 - the packages on nuget.org, not this source - writes of a run on the libtorch engine on the
// processor, once, so the tests can read it as a person's application reads what an earlier release left. Nothing here is a real flock:
// every number is worked out from the row's place, so the files can be made again byte for byte.
//
// What the release after 0.14.0 adds to an engine's name - the width it keeps its work in - is written only when it says something, so
// a run at full width must be written, read and gone on from exactly as 0.14.0 wrote it.
//
// Copy this file into an empty folder outside the repository (the repository's own build settings do not apply to it) and run:
//
//     dotnet run make-0.14.0.torch.cs -- <folder the files go to>
//
// The files it leaves are those named torch-0.14.0.*; the tests in FilesWrittenBy0140OnLibtorchTests read them. Nothing in this folder
// is part of the build.

#:package DeepSharp@0.14.0
#:package DeepSharp.Pipelines@0.14.0
#:package DeepSharp.Learners.Networks@0.14.0
#:package DeepSharp.Backends.TorchSharp@0.14.0
#:package libtorch-cpu-win-x64@2.10.0
#:property PublishAot=false

using System.Globalization;
using System.Text;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

var target = Path.GetFullPath(args.Length > 0 ? args[0] : ".");
Directory.CreateDirectory(target);
Directory.SetCurrentDirectory(target);   // a relative path in a pipeline is read from where the program runs

const string Stem = "torch-0.14.0";
var text = new StringBuilder("Farm,Heavy,Animals,Age\n");

for (var row = 0; row < 240; row++)
{
    var animals = 20000 + (row * 37 % 900);
    var age = 28 + (row * 5 % 14);

    text.Append(CultureInfo.InvariantCulture, $"{"ABCD"[row % 4]},{(age > 34 ? "True" : "False")},{animals},{age}\n");
}

File.WriteAllText(Path.Join(target, $"{Stem}.csv"), text.ToString(), new UTF8Encoding(false));

var pipeline = Pdd.Create()
    .ReadCsv($"{Stem}.csv")
    .Declare(schema => schema.Category("Farm").Boolean("Heavy").Number("Animals", "Age"))
    .ScaleGiven(scale => scale.Between("Animals", 15000, 25000))
    .SplitStratified("Heavy", train: 0.70, validation: 0.15, seed: 7)
    .Target("Heavy")
    .EncodeCategories()
    .Normalise("Age")
    .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16))
    .Build();
var prepared = pipeline.Run();
var checkpointed = string.Empty;
var compiled = new Sequential().Dense(4).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());

compiled.Fit(prepared, new FitOptions(seed: 20261017)
{
    Epochs = 3,
    BatchSize = 16,
    Backend = TorchBackend.OnCpu(),
    Checkpoints = new Checkpoints(checkpoint => checkpointed = checkpoint.Epochs == 2 ? CheckpointFile.Write(compiled, prepared, checkpoint) : checkpointed),
});

File.WriteAllText(Path.Join(target, $"{Stem}.pipeline.json"), pipeline.Declaration.ToJson(), new UTF8Encoding(false));
File.WriteAllText(Path.Join(target, $"{Stem}.checkpoint.json"), checkpointed, new UTF8Encoding(false));

Console.WriteLine($"wrote {Directory.GetFiles(target, $"{Stem}.*").Length} files to {target}");

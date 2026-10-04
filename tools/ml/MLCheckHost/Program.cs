// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// What an application that brought only the packages finds. Built twice: with the package that trains, where it fits a
// tree of the Titanic pipeline and writes the model beside its pipeline as one file; and without it, where that file is
// read back, said back and plainly cannot be trained from, because the door is not there to call.
using DeepSharp.Learners.ML;
using DeepSharp.Pipelines;
#if TRAINS
using DeepSharp.Learners.MLNet;
#endif

var what = args.Length > 0 ? args[0] : "reads";
var passengers = args.Length > 1 ? args[1] : "titanic.csv";
var file = args.Length > 2 ? args[2] : "titanic.mlmodel.json";

#if TRAINS
if (what == "trains")
{
    var trained = Pdd.Create()
        .ReadCsv(passengers)
        .Declare(schema => schema.Integer("survived", "sibsp").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
        .SplitStratified("survived", train: 0.70, validation: 0.15)
        .FillMissing(fill => fill.Median("age"))
        .EncodeCategories()
        .Normalise("age", "fare")
        .Target("survived")
        .WithML(trainer => trainer.FastTree(trees: 20))
        .Build()
        .TrainWithML();

    File.WriteAllText(file, trained.ToJson());

    Console.WriteLine(
        $"    trained a {trained.Trainer.Kind} on {trained.TrainedOn.Rows} rows of {trained.TrainedOn.Features.Count} features, "
        + $"predicting '{trained.TrainedOn.Answer}', and wrote it beside its pipeline as one file of {new FileInfo(file).Length} bytes");

    return 0;
}
#endif

if (what == "trains")
{
    Console.Error.WriteLine("::error::this build brought no trainer, so it cannot train");

    return 1;
}

var read = MLModelFile.FromJson(File.ReadAllText(file));
var carriesMLNet = AppDomain.CurrentDomain.GetAssemblies()
    .Concat(typeof(MLModelFile).Assembly.GetReferencedAssemblies().Select(name => (object)name))
    .Any(each => each.ToString()!.Contains("Microsoft.ML", StringComparison.Ordinal));

Console.WriteLine(
    $"    read the model file: a {read.Trainer.Kind} of {read.Rows} rows, written by ML.NET {read.Library} on {read.Processor}, "
    + $"carrying its pipeline at version {read.Carried.Version} with digest {read.Carried.Digest[..12]}");
Console.WriteLine($"    the model itself is {read.Model.Length} bytes of ML.NET's own archive, which only ML.NET opens");

if (carriesMLNet)
{
    Console.Error.WriteLine("::error::this build was supposed to carry nothing of ML.NET, and it does");

    return 1;
}

Console.WriteLine("    and this application carries nothing of ML.NET at all");

return 0;

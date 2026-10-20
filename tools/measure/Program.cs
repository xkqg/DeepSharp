// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using DeepSharp.Measure;
using DeepSharp.Tensors;

// What the light engine costs, written as Markdown: `dotnet run --project tools/measure -c Release -f net10.0 -- all`.
// "products" times the matrix products only, "training" the runs, "all" both. Nothing here is a test: a time is the machine's.
var what = args.Length > 0 ? args[0] : "all";
var named = new CpuBackend();

Console.WriteLine($"# light engine, {RuntimeInformation.FrameworkDescription}, DeepSharp {named.Version}, {Environment.ProcessorCount} logical processors, vector width {System.Numerics.Vector<float>.Count} floats");
Console.WriteLine();

if (what is "products" or "all")
{
    Console.WriteLine("## matrix products, 0.13.0's arithmetic against this build's");
    Console.WriteLine();
    Console.Write(new ProductWorkloads(new Measurement()).Table());
    Console.WriteLine();
}

if (what is "training" or "all")
{
    Console.WriteLine("## training runs, alone and five at a time in one process");
    Console.WriteLine();
    Console.Write(new TrainingWorkloads(repeats: 3).Table());
}

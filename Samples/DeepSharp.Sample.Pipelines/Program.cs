// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

// A pipeline reached the way an application reaches everything else: a builder, its services, and the app
// that comes out of it. Nothing here is required to use the library -- the last few lines do the same thing
// with no host anywhere -- but this is the shape a .NET application already has, so the library fits into it.
var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddDeepSharpPipelines();

using var app = builder.Build();

var pipelines = app.Services.GetRequiredService<IPipelineFactory>();

// Declaring, not doing: no file is opened by any of this. The path need not even exist yet.
var declaration = pipelines.Create()
    .ReadCsv("btceur-1d.csv")
    .SplitByTime("timestamp", train: 0.70, validation: 0.15, test: 0.15)
    .FillMissing("trades", With.Mean)
    .Declaration;

Console.WriteLine("Declared:");
Console.WriteLine($"  {declaration}");
Console.WriteLine();

var json = declaration.ToJson();

Console.WriteLine("As a file:");
Console.WriteLine(json);
Console.WriteLine();

// The other direction, which is the property the whole design rests on: what was built in C# reads back as
// the same declaration, so a pipeline written by hand and one written in code reach equally far.
var returned = PipelineDeclaration.FromJson(json);

Console.WriteLine($"Read back:  {returned}");
Console.WriteLine($"Identical:  {returned.Equals(declaration)}");
Console.WriteLine();

// And the same thing again, with no container in sight.
Console.WriteLine($"Without a host: {Pdd.Create().ReadCsv("btceur-1d.csv").Declaration}");

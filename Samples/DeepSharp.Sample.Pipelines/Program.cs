// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Sample.Pipelines;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

// A pipeline reached the way an application reaches everything else: a builder, its services, and the app
// that comes out of it. None of it is required -- Pdd.Create() starts the same pipeline with no host anywhere, and the
// sample's last lines run one again from its file with none -- but this is the shape a .NET application already has, so
// the library fits into it.
var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddDeepSharpPipelines();

using var app = builder.Build();

PipelineSample.Run(
    app.Services.GetRequiredService<IPipelineFactory>(),
    Path.GetFullPath(Path.Join(AppContext.BaseDirectory, "..", "..", "..", "..", "data")),
    Console.Out);

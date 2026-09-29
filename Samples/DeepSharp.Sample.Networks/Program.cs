// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Sample.Networks;

// The three walked networks, each behind its pipeline: trained, measured, saved, read back and served. The charts land
// beside the program, as SVG, for a browser to open.
var data = Path.Join(AppContext.BaseDirectory, "..", "..", "..", "..", "data");
var charts = Directory.CreateDirectory(Path.Join(AppContext.BaseDirectory, "charts")).FullName;

NetworkSample.Run(data, Console.Out, charts);

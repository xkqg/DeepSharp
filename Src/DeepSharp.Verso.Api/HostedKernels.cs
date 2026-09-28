// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What became of an open notebook's kernels, as Verso's editors show it beside the notebook's name.</summary>
/// <param name="Restarts">
/// How many times one of its kernels was started afresh since the notebook opened — by a stop, or by Verso's Restart
/// Kernel — each of which clears the notebook's variables, the pipeline handed to C# cells among them. Run All starts
/// the kernels afresh too, but says nothing of it, as Verso's editors say nothing.
/// </param>
/// <param name="Restarting">Whether one is being started afresh now.</param>
/// <param name="Fault">
/// Why the last start afresh failed — the kernel is gone meanwhile — until a kernel is started afresh or a cell begins, as
/// Verso's editors stop showing it once a cell runs; nothing when none failed.
/// </param>
public readonly record struct HostedKernels(long Restarts, bool Restarting, string? Fault);

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>A number a file holds for one slot of a network: the slot's path, the tensor, and where in the file it stands.</summary>
/// <param name="Path">The path of the slot it is for, dotted as the network names it: <c>1.weight</c>, <c>1.running_mean</c>.</param>
/// <param name="Value">The tensor the slot is to hold, laid out as the slot keeps it.</param>
/// <param name="Source">
/// Where the file holds it, in the file's own words — a tensor's name, a dataset's path, a line — so a fault is named where
/// somebody can find it.
/// </param>
/// <remarks>What a reader of a file hands <see cref="Network.Load"/>, one for every slot of the network.</remarks>
public readonly record struct SlotEntry(string Path, Tensor Value, string Source);

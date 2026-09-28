// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// What a notebook says of itself, as Verso's Metadata panel shows it: its title, which names what it is exported as, the
/// kernel a code cell that names no language runs in, when it was made and last saved, and the version of its format.
/// </summary>
/// <param name="Title">Its title; nothing while it has none.</param>
/// <param name="DefaultKernel">The language of the kernel a code cell that names none runs in; nothing while it names none.</param>
/// <param name="Created">When it was made; nothing when its file does not say.</param>
/// <param name="Modified">When it was last saved; nothing when its file does not say.</param>
/// <param name="FormatVersion">The version of Verso's format it is held in.</param>
public readonly record struct HostedMetadata(string? Title, string? DefaultKernel, DateTimeOffset? Created, DateTimeOffset? Modified, string FormatVersion);

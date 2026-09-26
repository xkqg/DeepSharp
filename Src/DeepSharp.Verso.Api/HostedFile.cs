// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A file a toolbar button handed over, for whoever pressed it to save.</summary>
/// <param name="Name">The name it is saved under.</param>
/// <param name="ContentType">What it holds.</param>
/// <param name="Bytes">Its bytes.</param>
/// <remarks>Two looks at a file are equal while they hold the same, whichever array the bytes came in.</remarks>
public readonly record struct HostedFile(string Name, string ContentType, byte[] Bytes)
{
    /// <inheritdoc />
    public bool Equals(HostedFile other) => Name == other.Name && ContentType == other.ContentType && Bytes.AsSpan().SequenceEqual(other.Bytes);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Name, ContentType, Bytes.Length);
}

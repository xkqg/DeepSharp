// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What a field of a properties panel takes.</summary>
public enum FieldKind
{
    /// <summary>A line of text.</summary>
    Text = 0,

    /// <summary>A number.</summary>
    Number = 1,

    /// <summary>On or off.</summary>
    Toggle = 2,

    /// <summary>One of its choices.</summary>
    Select = 3,

    /// <summary>Any of its choices.</summary>
    MultiSelect = 4,

    /// <summary>A colour.</summary>
    Color = 5,

    /// <summary>Words of the person's own choosing.</summary>
    Tags = 6,
}

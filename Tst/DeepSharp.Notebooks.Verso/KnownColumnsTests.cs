// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Notebooks.Verso;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The columns a key can name are those of a kind its step works on — one rule, read by the form and by the editor
/// alike, so the two never offer different columns for the same key.
/// </summary>
public class KnownColumnsTests
{
    [Fact]
    public void TheNamesOfAKind_AreEachNamedOnce_InTheOrderTheyCome()
    {
        KnownColumn[] known =
        [
            new("age", ColumnKind.Number, Surely: false),
            new("sex", ColumnKind.Category, Surely: true),
            new("fare", ColumnKind.Number, Surely: true),
            new("age", ColumnKind.Number, Surely: true),
            new("pclass", ColumnKind.Integer, Surely: true),
        ];

        Assert.Equal(["age", "fare", "pclass"], known.NamesOf(ColumnKinds.Fillable));
        Assert.Equal(["sex"], known.NamesOf([ColumnKind.Category]));
        Assert.Empty(known.NamesOf([ColumnKind.Timestamp]));
    }
}

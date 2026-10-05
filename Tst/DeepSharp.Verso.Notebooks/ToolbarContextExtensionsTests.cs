// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// Where a cell stands among the notebook's cells, as a button that writes blocks asks it after each block it makes. A cell
/// the notebook no longer holds is refused in words that say so, rather than answered with a place that is none.
/// </summary>
public sealed class ToolbarContextExtensionsTests
{
    /// <summary>A toolbar context that holds the cells it is given and answers nothing else.</summary>
    public class Context : DispatchProxy
    {
        /// <summary>The cells the context holds.</summary>
        public IReadOnlyList<CellModel> Cells { get; set; } = [];

        /// <inheritdoc />
        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) =>
            targetMethod?.Name == $"get_{nameof(IToolbarActionContext.NotebookCells)}"
                ? Cells
                : throw new NotSupportedException(targetMethod?.Name);
    }

    private static IToolbarActionContext Holding(params CellModel[] cells)
    {
        var context = DispatchProxy.Create<IToolbarActionContext, Context>();

        ((Context)context).Cells = cells;

        return context;
    }

    [Fact]
    public void ACellThatStands_IsFoundAtItsPlace_CountingFromNought()
    {
        var cells = new[] { new CellModel(), new CellModel(), new CellModel() };

        Assert.Equal(0, Holding(cells).IndexOf(cells[0].Id));
        Assert.Equal(2, Holding(cells).IndexOf(cells[2].Id));
    }

    [Fact]
    public void ACellThatIsNoLongerThere_IsRefused_InWordsThatSaySo()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Holding(new CellModel()).IndexOf(Guid.NewGuid()));

        Assert.Contains("no longer in it", refused.Message, StringComparison.Ordinal);
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Notebooks.Verso;

/// <summary>The words a parameter takes, as an editor offers them.</summary>
/// <param name="Inside">What goes between the quotes of a value or of an item of a list.</param>
/// <param name="Outside">Whole values, written where a value goes.</param>
internal readonly record struct Words(IReadOnlyList<string> Inside, IReadOnlyList<string> Outside)
{
    /// <summary>Nothing to offer.</summary>
    public static Words None { get; } = new([], []);
}

/// <summary>
/// The words each kind of parameter takes, read from the parameter itself, and the columns a key naming one can name.
/// </summary>
/// <param name="scope">The columns known, each with a kind it holds; none before anything showed the pipeline.</param>
/// <param name="written">The items the list under the cursor already holds; a set of columns names each once.</param>
/// <remarks>
/// A projection of the kinds like the schema and the reference page: the words an editor offers are the ones the
/// reader accepts, taken from the same place, so the two cannot disagree about a word. A key naming a column is
/// offered the columns of a kind it works on; a new column's name is a person's own.
/// </remarks>
internal sealed class StepWords(IReadOnlyList<KnownColumn> scope, IReadOnlyList<string> written) : IStepParameterVisitor<Words>
{
    public Words Visit(TextParameter parameter) => Words.None;

    public Words Visit(FilePathParameter parameter) => Words.None;

    public Words Visit(ColumnParameter parameter)
    {
        var names = Named(parameter.Accepts);

        return new(names, [.. names.Select(Quote)]);
    }

    public Words Visit(NewColumnParameter parameter) => Words.None;

    public Words Visit(ColumnsParameter parameter) =>
        new(parameter.Repeatable ? Named(parameter.Accepts) : [.. Named(parameter.Accepts).Except(written, StringComparer.Ordinal)], []);

    public Words Visit(NumberParameter parameter) => Words.None;

    public Words Visit(WholeNumberParameter parameter) => Words.None;

    public Words Visit(TrueOrFalseParameter parameter) => new([], ["true", "false"]);

    public Words Visit(ShareParameter parameter) => Words.None;

    public Words Visit<TEnum>(OneOfParameter<TEnum> parameter)
        where TEnum : struct, Enum => new(parameter.Choices, [.. parameter.Choices.Select(Quote)]);

    public Words Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
        where TEnum : struct, Enum => new(parameter.Choices, []);

    // A strategy carrying a number is an object rather than a word, so it is offered whole or not at all.
    public Words Visit(FillStrategyParameter parameter) =>
        new(
            [.. parameter.Allowed.Where(name => !With.TakesAValue(name))],
            [.. parameter.Allowed.Select(name => With.TakesAValue(name)
                ? $"{{\"{FillStrategyParameter.KindKey}\": \"{name}\", \"{FillStrategyParameter.ValueKey}\": 0}}"
                : Quote(name))]);

    public Words Visit(SplitSharesParameter parameter) => Words.None;

    public Words Visit(ColumnDeclarationsParameter parameter) => Words.None;

    private IReadOnlyList<string> Named(IReadOnlyList<ColumnKind> accepts) => scope.NamesOf(accepts);

    private static string Quote(string word) => $"\"{word}\"";
}

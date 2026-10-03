// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How a category is written down as numbers.</summary>
public enum As
{
    /// <summary>One column per category, one of them a one and the rest nothing.</summary>
    OneHot,

    /// <summary>One column holding the category's place in the list.</summary>
    Ordinal,
}

/// <summary>What happens to a category the training rows never held.</summary>
public enum Unseen
{
    /// <summary>
    /// A place is kept for it, so an unfamiliar value has somewhere to go: written one column per category, the column
    /// named after the encoded one and <c>other</c>, so a category the training rows hold under that name is refused.
    /// </summary>
    Reserve,

    /// <summary>Stop, and say the data holds something this model has never seen.</summary>
    Refuse,
}

/// <summary>
/// Writes a category down as numbers, using the categories the training rows held.
/// </summary>
/// <remarks>
/// The list of categories is learned, which is what makes this a step that belongs after the split. An
/// unfamiliar value will turn up in production sooner or later, so what happens then is declared: a place
/// kept for it, or a refusal saying the data holds something this model has never seen.
/// </remarks>
public sealed record EncodeStep : IFittedStep, IPipelineStep<EncodeStep>, IDescribesColumns, IMeetsANeed, IEncodesCategories
{
    // What a gap is written as when the categories are handed over as their places: no place at all.
    private const double NoPlace = -1;

    // What the column kept for a category the training rows never held is named after, beside the column's own name.
    private const string Reserved = "other";

    // What the column marking where the cell was empty is named after, beside the column's own name.
    private const string Marked = "was_missing";

    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column of words to write down as numbers.", "column", ColumnKinds.Any);

    private static readonly OneOfParameter<As> AsKey = new(
        "as", "How the categories are written down: one column per category, or one column of places.", As.OneHot);

    private static readonly OneOfParameter<Unseen> UnseenKey = new(
        "unseen", "What happens to a category the training rows never held: a place kept for it, or a refusal.", Unseen.Reserve);

    /// <summary>Declares that a column of words is written down as numbers.</summary>
    /// <param name="column">The column of words.</param>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public EncodeStep(string column, As how = As.OneHot, Unseen unseen = Unseen.Reserve)
    {
        Column = ColumnKey.Require(column);
        How = AsKey.Require(how);
        Unseen = UnseenKey.Require(unseen);
    }

    /// <inheritdoc />
    public static StepParameters<EncodeStep> Parameters { get; } = new StepParameters<EncodeStep>()
        .With(ColumnKey, step => step.Column)
        .With(AsKey, step => step.How)
        .With(UnseenKey, step => step.Unseen);

    /// <summary>The column of words.</summary>
    public string Column { get; }

    /// <summary>How the categories are written down.</summary>
    public As How { get; }

    /// <summary>What happens to a category the training rows never held.</summary>
    public Unseen Unseen { get; }

    /// <summary>The column written beside an encoded one, saying where the cell was empty.</summary>
    public string MarkerColumn => $"{Column}_{Marked}";

    /// <inheritdoc />
    /// <remarks>
    /// Every learner but one that takes categories itself. For that one, the same categories are learned from the training
    /// rows, under the same entry, and each is handed over as its place in their list.
    /// </remarks>
    public bool NeededBy(Needs needs) => !needs.TakesCategories();

    /// <inheritdoc />
    IFittedStep? IMeetsANeed.Instead => new CategoriesAsPlaces(this);

    /// <inheritdoc />
    IEnumerable<string> IEncodesCategories.Encoded(ColumnState before) => [Column];

    /// <inheritdoc />
    IReadOnlyDictionary<string, IReadOnlyList<string>> IEncodesCategories.CategoriesIn(FittedStepValues fitted) =>
        new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal) { [Column] = fitted.List("categories") };

    /// <inheritdoc />
    /// <remarks>
    /// One column per category is a family: which categories there are is known once the training rows have
    /// been seen, so its members are known by the start of their names. One column of places keeps the name.
    /// </remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return How == As.Ordinal
            ? PlacesAfter(before)
            : before.Without(Column).WithFamily($"{Column}_", Form.Unit).With(MarkerColumn, ColumnKind.Number, Form.Unit);
    }

    /// <summary>The columns after this one is written down as the places of its categories, beside where it was empty.</summary>
    /// <param name="before">The columns before it.</param>
    /// <returns>The columns with the places, numbers that land in no range, and the marker.</returns>
    internal ColumnState PlacesAfter(ColumnState before) =>
        before.Without(Column).With(Column, ColumnKind.Number).With(MarkerColumn, ColumnKind.Number, Form.Unit);

    /// <inheritdoc />
    public static string Name => "encode";

    /// <inheritdoc />
    public static string Purpose => "Writes a column of words down as numbers, using the categories the training rows held.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">
    /// Every training row of the column is a gap; or, written one column per category, a category the training rows hold
    /// would take a column of this step's own: 'other', the column kept for a category they never held, or 'was_missing',
    /// the column that marks a gap.
    /// </exception>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var categories = table.CategoriesOf(Column, row => parts[row] == Part.Train);

        if (categories.Count == 0)
        {
            throw new InvalidOperationException(
                $"Every training row of '{Column}' is a gap, so there are no categories to learn.");
        }

        ThrowIfACategoryTakesAColumnOfThisStep(categories);

        var learned = new FittedStepValues();
        learned.Learned("categories", categories);

        return learned;
    }

    // Written one column per category, a category's column is named after it, beside the column kept for a category the
    // training rows never held and the one marking a gap: a category of either name would take that column, and its rows
    // would lose their category without a word. Refused here, where the categories are learned, so every run of the
    // declaration refuses it alike; a replay learns nothing, so a file written before this refusal replays as it was written.
    private void ThrowIfACategoryTakesAColumnOfThisStep(IReadOnlyList<string> categories)
    {
        if (How != As.OneHot)
        {
            return;
        }

        if (Unseen == Unseen.Reserve && categories.Contains(Reserved, StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                $"'{Column}' holds the category '{Reserved}' on its training rows, and written one column per category it would take "
                + $"'{Column}_{Reserved}', the column kept for a category the training rows never held. Declare the encoder with "
                + "unseen: refuse, which keeps no such column, or with as: ordinal, which writes each category as its place.");
        }

        if (categories.Contains(Marked, StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                $"'{Column}' holds the category '{Marked}' on its training rows, and written one column per category it would take "
                + $"'{MarkerColumn}', the column that marks where the cell was empty. Declare the encoder with as: ordinal, which "
                + "writes each category as its place.");
        }
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var categories = fitted.List("categories");
        var places = PlacesIn(table, categories);

        table.Remove(Column);

        // An empty cell is not a category and not an unfamiliar one either, so it becomes no category at
        // all -- every slot nothing -- and the marking column remembers that it was empty. Leaving a gap
        // in the encoded columns instead would only move the problem to whoever hands the rows over.
        var marker = Marker(places);

        if (How == As.Ordinal)
        {
            // Declared this way, a gap takes the first place, as it always has, and only its mark tells it apart.
            table.Put(new Column<double>(Column, ColumnKind.Number, places.Select(place => (double?)(place ?? 0))));
            table.Put(marker);

            return;
        }

        var slots = Unseen == Unseen.Reserve ? categories.Count + 1 : categories.Count;

        for (var slot = 0; slot < slots; slot++)
        {
            var label = slot < categories.Count ? categories[slot] : Reserved;
            var here = slot;

            table.Put(new Column<double>(
                $"{Column}_{label}", ColumnKind.Number,
                places.Select(place => (double?)(place == here ? 1 : 0))));
        }

        table.Put(marker);
    }

    /// <summary>
    /// Writes the column down as the place of each category in the list the training rows held — the place kept for one
    /// they never held, and no place, minus one, for a gap — beside the column that says where the cell was empty.
    /// </summary>
    /// <param name="table">The data, changed in place.</param>
    /// <param name="categories">The categories the training rows held, in the order of their places.</param>
    /// <exception cref="InvalidOperationException">A row holds a category the training rows never held, and such a category is refused.</exception>
    internal void PutPlaces(Table table, IReadOnlyList<string> categories)
    {
        var places = PlacesIn(table, categories);

        table.Remove(Column);
        table.Put(new Column<double>(Column, ColumnKind.Number, places.Select(place => (double?)(place ?? NoPlace))));
        table.Put(Marker(places));
    }

    // Each row's category as its place in the list: the place kept for one the training rows never held, or a refusal that
    // names the row as it was read; nothing for a gap.
    private double?[] PlacesIn(Table table, IReadOnlyList<string> categories)
    {
        var column = table[Column];
        var listed = categories.ToList();
        var places = new double?[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            if (column.IsMissing(row))
            {
                continue;
            }

            var at = listed.IndexOf(column.TextAt(row)!);

            // A refusal names the row as it was read, which is the row a person can find in their file.
            places[row] = at >= 0
                ? at
                : Unseen == Unseen.Refuse
                    ? throw new InvalidOperationException(
                        $"Row {table.Identities[row].ReadAt + 1} of '{Column}' holds '{column.TextAt(row)}', which the training rows never held.")
                    : categories.Count;
        }

        return places;
    }

    private Column<double> Marker(double?[] places) =>
        new(MarkerColumn, ColumnKind.Number, places.Select(place => (double?)(place is null ? 1 : 0)));

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static EncodeStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), AsKey.Read(element), UnseenKey.Read(element));
}

/// <summary>
/// Writes every column that stands for a group down as numbers, each by the categories the training rows held.
/// </summary>
/// <remarks>
/// Which columns are categories is said once, where it is a fact: in the schema, or by the step that made the
/// column. This takes every column that is a category where it stands, so marking one more column a category
/// is the whole of the change — nothing further down has to be told. It was a word in the chain that became
/// one encoding step per category at the moment it was written, so a file could not say it and a column
/// marked a category afterwards was never encoded at all.
/// </remarks>
public sealed record EncodeCategoriesStep : IFittedStep, IPipelineStep<EncodeCategoriesStep>, IDescribesColumns, IMeetsANeed, IEncodesCategories
{
    private static readonly OneOfParameter<As> AsKey = new(
        "as", "How each category is written down: one column per category, or one column of places.", As.OneHot);

    private static readonly OneOfParameter<Unseen> UnseenKey = new(
        "unseen", "What happens to a category the training rows never held: a place kept for it, or a refusal.", Unseen.Reserve);

    /// <summary>Declares that every category column is written down as numbers.</summary>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    public EncodeCategoriesStep(As how = As.OneHot, Unseen unseen = Unseen.Reserve)
    {
        How = AsKey.Require(how);
        Unseen = UnseenKey.Require(unseen);
    }

    /// <summary>How the categories are written down.</summary>
    public As How { get; }

    /// <summary>What happens to a category the training rows never held.</summary>
    public Unseen Unseen { get; }

    /// <inheritdoc />
    /// <remarks>
    /// Every learner but one that takes categories itself. For that one, the same categories are learned from the training
    /// rows, under the same entry, and each is handed over as its place in their list.
    /// </remarks>
    public bool NeededBy(Needs needs) => !needs.TakesCategories();

    /// <inheritdoc />
    IFittedStep? IMeetsANeed.Instead => new CategoriesAsPlaces(this);

    /// <inheritdoc />
    IEnumerable<string> IEncodesCategories.Encoded(ColumnState before) =>
        before.Columns.Where(column => column.Kind == ColumnKind.Category).Select(column => column.Name);

    /// <inheritdoc />
    IReadOnlyDictionary<string, IReadOnlyList<string>> IEncodesCategories.CategoriesIn(FittedStepValues fitted) => fitted.Lists;

    /// <inheritdoc />
    /// <remarks>Each category where it stands, encoded exactly as a step for that one column would be.</remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.Columns
            .Where(column => column.Kind == ColumnKind.Category)
            .Aggregate(before, (state, column) => new EncodeStep(column.Name, How, Unseen).After(state));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Refused where provably no column is a category: none is known, and nothing unnamed may be there. The run
    /// still refuses when none is, for the columns a step from elsewhere may have made.
    /// </remarks>
    public string? Refusal(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.Open || before.Columns.Any(column => column.Kind == ColumnKind.Category)
            ? null
            : "nothing before it declares a category, so this pipeline has no categories to write down as numbers.";
    }

    /// <inheritdoc />
    public static string Name => "encode.categories";

    /// <inheritdoc />
    public static string Purpose => "Writes every column that stands for a group down as numbers, each by the categories the training rows held.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<EncodeCategoriesStep> Parameters { get; } = new StepParameters<EncodeCategoriesStep>()
        .With(AsKey, step => step.How)
        .With(UnseenKey, step => step.Unseen);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">
    /// No column is a category where this step stands; or a column's categories are refused as <see cref="EncodeStep.Fit"/>
    /// refuses them.
    /// </exception>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var categories = table.Columns.Where(column => column.Kind == ColumnKind.Category).ToArray();

        if (categories.Length == 0)
        {
            throw new InvalidOperationException(
                "This pipeline has no categories where the encoder stands, so there is nothing to write down as numbers.");
        }

        var learned = new FittedStepValues();

        // One list per column, under the column's name, each learned exactly as a step for that one column
        // would learn it: from the training rows alone.
        foreach (var column in categories)
        {
            learned.Learned(column.Name, new EncodeStep(column.Name, How, Unseen).Fit(table, parts).List("categories"));
        }

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        // In the order the columns stand on the table, which a run and a replay reach the same way, so both
        // hand the encoded columns over in one order.
        var encoded = table.Columns.Select(column => column.Name).Where(fitted.Lists.ContainsKey).ToArray();

        foreach (var column in encoded)
        {
            var one = new FittedStepValues();
            one.Learned("categories", fitted.List(column));

            new EncodeStep(column, How, Unseen).ApplyTo(table, one);
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static EncodeCategoriesStep ReadFrom(JsonElement element) => new(AsKey.Read(element), UnseenKey.Read(element));
}

/// <summary>A step that learns the categories of columns from the training rows and writes each down as numbers.</summary>
/// <remarks>What an encoder handing its categories over as places needs of the encoder it stands for.</remarks>
internal interface IEncodesCategories : IFittedStep
{
    /// <summary>What happens to a category the training rows never held.</summary>
    Unseen Unseen { get; }

    /// <summary>The columns it writes down, where the columns before it are these.</summary>
    /// <param name="before">The columns before it.</param>
    /// <returns>Their names.</returns>
    IEnumerable<string> Encoded(ColumnState before);

    /// <summary>The categories it learned for each column it writes down, each list in the order of the places.</summary>
    /// <param name="fitted">What it learned.</param>
    /// <returns>The lists, by the column each was learned for.</returns>
    IReadOnlyDictionary<string, IReadOnlyList<string>> CategoriesIn(FittedStepValues fitted);
}

/// <summary>
/// An encoder as a run for a learner that takes categories hands its categories over: each as its place in the list the
/// training rows held, a category they never held at the place kept for it, and a gap as no place, minus one.
/// </summary>
/// <param name="Declared">The encoder as it was declared: what it learns, and how it is written.</param>
/// <remarks>
/// It learns what the encoder learns, under the same entry, so a network's run and this one share the categories learned;
/// and it is written as the encoder is, since a run's file names every step as it was declared. The column keeps its name
/// and holds numbers, so an encoder further down, which takes the categories where it stands, never writes it down again.
/// </remarks>
internal sealed record CategoriesAsPlaces(IEncodesCategories Declared) : IFittedStep, IDescribesColumns
{
    /// <inheritdoc />
    public string Verb => Declared.Verb;

    /// <inheritdoc />
    public IReadOnlyList<ColumnRead> ColumnsRead => Declared.ColumnsRead;

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer) => Declared.WriteTo(writer);

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts) => Declared.Fit(table, parts);

    /// <inheritdoc />
    /// <remarks>In the order the columns stand on the table, which a run and a replay reach the same way.</remarks>
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        var learned = Declared.CategoriesIn(fitted);

        foreach (var column in table.Columns.Select(each => each.Name).Where(learned.ContainsKey).ToArray())
        {
            new EncodeStep(column, As.Ordinal, Declared.Unseen).PutPlaces(table, learned[column]);
        }
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) =>
        Declared.Encoded(before).Aggregate(before, (state, column) => new EncodeStep(column).PlacesAfter(state));
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// The numbers of one split, in the shape anything that learns can take them.
/// </summary>
/// <param name="FeatureNames">The columns, in the order every row lists them.</param>
/// <param name="Features">One row of numbers per row of data.</param>
/// <param name="Labels">The answer for each row, when the pipeline's output names exactly one.</param>
/// <remarks>
/// The column order is part of the handover, not an accident of iteration: a model fed the same numbers in
/// a different order is quietly a different model, and nothing about the numbers themselves would say so.
/// <para>
/// The answers are handed over as rows of numbers too, as many a row as the output names — seventy for a
/// histogram of weights — and in the order it names them, for the same reason. An output of one answer hands
/// it over both ways: as the labels, one number a row, and as the answers.
/// </para>
/// <para>
/// A batch also says which part it was handed over from and which row each is, by its key, so what a model predicts
/// for it is measured against the rows it was made for.
/// </para>
/// </remarks>
public readonly record struct Batch(
    IReadOnlyList<string> FeatureNames,
    IReadOnlyList<double[]> Features,
    IReadOnlyList<double>? Labels)
{
    /// <summary>
    /// The columns that hold the answer, in the order each row of <see cref="Answers"/> lists them; nothing when no
    /// answer is handed over.
    /// </summary>
    public IReadOnlyList<string>? AnswerNames { get; init; }

    /// <summary>The answers of each row, as many numbers as <see cref="AnswerNames"/> names; nothing when no answer is handed over.</summary>
    public IReadOnlyList<double[]>? Answers { get; init; }

    /// <summary>The part these rows were handed over from; nothing, for a batch <see cref="Handover.Batch(PreparedData, Pipelines.Part, Needs)"/> did not make.</summary>
    public Part? Part { get; init; }

    /// <summary>
    /// The key of each row, in the order the rows are handed over: a digest of the record it was read from; nothing, for a
    /// batch <see cref="Handover.Batch(PreparedData, Pipelines.Part, Needs)"/> did not make.
    /// </summary>
    /// <remarks>
    /// What a model predicts for these rows is measured against them only when the predictions are for these rows in this
    /// order, which the keys are how to tell: predictions in another order would be measured against other rows' answers.
    /// </remarks>
    public IReadOnlyList<RowKey>? Keys { get; init; }

    /// <summary>
    /// Each feature the run handed over as the places of its categories, with the categories its training rows held, in
    /// the order of their places: empty when it handed none over so; nothing, for a batch
    /// <see cref="Handover.Batch(PreparedData, Pipelines.Part, Needs)"/> did not make.
    /// </summary>
    /// <remarks>
    /// Such a feature holds, for each row, the place of its category in that list, from nought; the place after the last —
    /// as many as the list holds — for a category the training rows never held; and minus one, no place, for a gap, which its
    /// <c>_was_missing</c> column marks as well. Only a run for a learner that takes categories hands a feature over so
    /// (<see cref="Needs.Categories"/>): a learner that splits on a feature's categories needs to know which features are
    /// categories and how many each holds, and a place read as a number would be an order nobody meant.
    /// </remarks>
    public IReadOnlyDictionary<string, IReadOnlyList<string>>? Categories { get; init; }

    /// <summary>How many rows this batch holds.</summary>
    public int RowCount => Features.Count;

    /// <summary>How many numbers each row holds.</summary>
    public int Width => FeatureNames.Count;
}

/// <summary>
/// Rows served through a trained pipeline, in the shape the training rows were handed over in.
/// </summary>
/// <param name="FeatureNames">The columns, in the order every row lists them — the training order.</param>
/// <param name="Features">One row of numbers per served row.</param>
/// <param name="HandedInAt">For each served row, which of the handed-in rows it is, counting from nought.</param>
/// <remarks>
/// No answers: a served row is asked for one. Which handed-in row each is, because a replay can drop rows
/// and put them in order, and a prediction has to find its way back to the row it was made for.
/// </remarks>
public readonly record struct ServedBatch(
    IReadOnlyList<string> FeatureNames,
    IReadOnlyList<double[]> Features,
    IReadOnlyList<int> HandedInAt)
{
    /// <summary>
    /// The key of each served row: a digest of the record it was read from, as it was handed in — an answer the rows do not
    /// carry counted as a gap, and one they carry as the cell they carry.
    /// </summary>
    /// <remarks>
    /// The same record has the same key in any hand-in, whatever the order, which is how a way back that needs the
    /// rows finds each one again. A row handed in with its answer is another record than the same row without it, so the
    /// two are keyed apart, though both are served alike. Nothing, for a batch
    /// <see cref="Handover.Served(PreparedData, IRowSource, Needs)"/> did not make.
    /// </remarks>
    public IReadOnlyList<RowKey>? Keys { get; init; }

    /// <summary>
    /// Each feature the run hands over as the places of its categories, with the categories its training rows held, in the
    /// order of their places, as <see cref="Batch.Categories"/> says them for the training rows: empty when it hands none
    /// over so; nothing, for a batch <see cref="Handover.Served(PreparedData, IRowSource, Needs)"/> did not make.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>>? Categories { get; init; }

    /// <summary>How many rows were served.</summary>
    public int RowCount => Features.Count;
}

/// <summary>What a learner needs of the features it is handed, said where they are handed over and where a pipeline is run for it.</summary>
/// <remarks>
/// One need a learner, each its own value: what each says is written once, by <see cref="NeedsExtensions"/>, so a learner
/// that comes to need something new gets a value of its own rather than a combination nothing else understands. A value no
/// need is named by is refused wherever a need is stated.
/// </remarks>
public enum Needs
{
    /// <summary>Numbers only, each on the scale its steps declare: the run takes every step, as a linear model takes it.</summary>
    Numbers,

    /// <summary>Every feature on one scale, between minus one and one, as a network takes them.</summary>
    OneScale,

    /// <summary>Numbers of any size, as a tree takes them: a step that only scales a feature can be left out.</summary>
    NoScale,

    /// <summary>
    /// Each category as its place in the list the training rows held, with numbers of any size, as a boosted tree that splits
    /// on categories takes them.
    /// </summary>
    Categories,
}

/// <summary>
/// Handing prepared data over to whatever learns from it.
/// </summary>
/// <remarks>
/// This is where the pipeline stops. Everything up to here is declared once, whatever is going to learn from the result:
/// rows of numbers, their column names, and the answers when the pipeline names an output. A learner says what it needs of
/// them, and the pipeline is run for it (<see cref="Pipeline.RunFor(Needs)"/>) — a scale left out for a learner indifferent
/// to scale, each category as its place for one that takes categories — so each learner is handed the same declaration, the
/// same rows in the same parts and the same things learned, in the form it takes them; and only a run it can take. A network
/// written here, a trainer from an established .NET library and something a caller wrote are all handed over here and
/// measured by the pipeline's report on the same rows — which is the only reason two of them can honestly be compared.
/// </remarks>
public static class Handover
{
    extension(PreparedData prepared)
    {
        /// <summary>The numbers of one part, ready for something that learns.</summary>
        /// <param name="part">Which part of it to hand over.</param>
        /// <returns>The rows of that split, and their answers when the pipeline names an output.</returns>
        /// <exception cref="ArgumentException">The part asked for is the gap a split keeps apart.</exception>
        /// <exception cref="InvalidOperationException">
        /// The pipeline holds no rows, as one read from its file holds none; the run left out a step a learner of numbers needs,
        /// a column still holds words, an answer column is not there, a value
        /// is a gap or not a finite number, or the output refuses a row's answers.
        /// </exception>
        /// <remarks>Handed over for a learner of numbers, <see cref="Needs.Numbers"/>, which takes every step as declared.</remarks>
        public Batch Batch(Part part) => prepared.Batch(part, Needs.Numbers);

        /// <summary>The numbers of one part, ready for a learner that needs them as it says.</summary>
        /// <param name="part">Which part of it to hand over.</param>
        /// <param name="needs">What the learner needs of its features.</param>
        /// <returns>
        /// The rows of that split, their answers when the pipeline names an output, and each feature handed over as the places of
        /// its categories.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">No need is named by the value.</exception>
        /// <exception cref="ArgumentException">The part asked for is the gap a split keeps apart.</exception>
        /// <exception cref="InvalidOperationException">
        /// The pipeline holds no rows, as one read from its file holds none; the run left out a step this learner needs, every
        /// such step named; a column still holds words, an answer column is
        /// not there, a value is a gap or not a finite number, the output refuses a row's answers, or — for a learner that takes
        /// every feature on one scale — a feature is not declared to land between minus one and one.
        /// </exception>
        /// <remarks>
        /// A run of every step goes to every learner; a run for a learner goes to one that does without every step it left out
        /// (<see cref="PreparedData.Skipped"/>). Where a feature lands is read from the steps the run took, as they say, not from
        /// the rows: a feature that happens to lie between minus one and one on these rows and is declared to land nowhere would
        /// not on the next ones.
        /// </remarks>
        public Batch Batch(Part part, Needs needs)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            _ = needs.Named();

            if (part == Part.Gap)
            {
                throw new ArgumentException(
                    "The rows a split keeps apart are fitted on by nothing and handed to nothing: that is what keeps them apart.",
                    nameof(part));
            }

            var rows = RowsOf(prepared, part);
            var features = FeaturesOf(prepared, prepared.Table, rows, needs);
            var answered = AnswersOf(prepared, prepared.Table, rows);

            // The labels are one number a row, so they carry an output of one answer.
            return new Batch(features.Names, features.Rows, answered.Names is [_] ? [.. answered.Answers!.Select(each => each[0])] : null)
            {
                AnswerNames = answered.Names,
                Answers = answered.Answers,
                Part = part,
                Keys = features.Keys,
                Categories = features.Categories,
            };
        }

        /// <summary>Rows that arrived after training, replayed and handed over in the shape the training rows were.</summary>
        /// <param name="rows">The rows to serve, without the answer — it is what is being asked.</param>
        /// <returns>Their numbers in the training order, and which handed-in row each is.</returns>
        /// <exception cref="InvalidOperationException">
        /// The run left out a step a learner of numbers needs, or a served row is refused for what a training row would be: a
        /// column still holding words, a gap, a value that is not a finite number.
        /// </exception>
        /// <remarks>
        /// Nothing is fitted: the rows are replayed with the numbers the training rows produced, which is the
        /// only way a model sees tomorrow's rows the way it saw the ones it learned from.
        /// </remarks>
        public ServedBatch Served(IRowSource rows) => prepared.Served(rows, Needs.Numbers);

        /// <summary>Rows that arrived after training, replayed and handed over to a learner that needs them as it says.</summary>
        /// <param name="rows">The rows to serve, without the answer — it is what is being asked.</param>
        /// <param name="needs">What the learner needs of its features.</param>
        /// <returns>Their numbers in the training order, which handed-in row each is, and each feature handed over as places.</returns>
        /// <exception cref="ArgumentOutOfRangeException">No need is named by the value.</exception>
        /// <exception cref="InvalidOperationException">
        /// The run left out a step this learner needs; a served row is refused for what a training row would be; or — for a
        /// learner that takes every feature on one scale — a feature is not declared to land between minus one and one, which a
        /// pipeline loaded from its file says as the one that was fitted does.
        /// </exception>
        /// <remarks>The steps are replayed as the run took them, so a served row is handed over as the training rows were.</remarks>
        public ServedBatch Served(IRowSource rows, Needs needs)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(rows);
            _ = needs.Named();

            var table = prepared.Replay(rows);
            var all = Enumerable.Range(0, table.RowCount).ToArray();
            var features = FeaturesOf(prepared, table, all, needs);

            return new ServedBatch(features.Names, features.Rows, [.. all.Select(row => table.Identities[row].ReadAt)])
            {
                Keys = features.Keys,
                Categories = features.Categories,
            };
        }
    }

    /// <summary>The answers of one part and the key of each of its rows, and nothing of its features.</summary>
    /// <param name="prepared">The data as the pipeline left it.</param>
    /// <param name="part">Which part of it.</param>
    /// <returns>The answers the output names, each row's, and the keys of the rows, in the order the part hands them over.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline holds no rows, as one read from its file holds none; an answer column is not there, an answer is a gap or
    /// not a finite number, or the output refuses a row's answers.
    /// </exception>
    /// <remarks>
    /// What the report measures a learner's predictions against, whatever the learner was handed: it reads none of the
    /// features, so it measures a learner that took categories as places, or a part a feature of which holds words, as it
    /// measures any other.
    /// </remarks>
    internal static HandedAnswers AnswersOf(PreparedData prepared, Part part) => AnswersOf(prepared, prepared.Table, RowsOf(prepared, part));

    // The rows of one part, by their place in the table; refused whole for a pipeline that holds no rows.
    private static int[] RowsOf(PreparedData prepared, Part part)
    {
        prepared.RequireRows();

        return [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part)];
    }

    // The features half: the steps the learner needs first, then words, then one scale over the steps the run took, then the
    // numbers, and what each feature handed over as places holds.
    private static HandedFeatures FeaturesOf(PreparedData prepared, Table table, int[] rows, Needs needs)
    {
        if (prepared.Course.Missed(needs) is [_, ..] missed)
        {
            throw new InvalidOperationException(
                $"This run left out steps a learner stating Needs.{needs} cannot do without: "
                + $"{string.Join("; ", missed.Select(at => $"step {at + 1}, '{prepared.Declaration.Steps[at].Verb}'"))}. "
                + $"Hand it a run of every step, Run(), or the run made for it, RunFor(Needs.{needs}).");
        }

        var answers = AnswersNamed(prepared, table);

        // Every answer the output names is left out of what a model is shown, not only the first: the others would
        // reach it as features.
        var features = table.Columns
            .Where(column => !answers.Contains(column.Name, StringComparer.Ordinal))
            .ToArray();

        var words = features.FirstOrDefault(column => column is TextColumn);

        if (words is not null)
        {
            // Words are not numbers, and turning them into one quietly is how a category becomes an order
            // nobody meant. Encode it, or leave it out of the schema.
            throw new InvalidOperationException(
                $"'{words.Name}' still holds words. Encode it, or do not declare it.");
        }

        if (needs.NeedsOneScale())
        {
            // Where each feature lands, as the steps the run took say: every one of them at once.
            var declared = prepared.Course.ColumnsBefore(prepared.Course.Steps.Count);
            string[] off = [.. features.Where(column => declared.LandsOf(column.Name) is null).Select(column => $"'{column.Name}'")];

            if (off.Length > 0)
            {
                throw new InvalidOperationException(
                    "A learner that takes every feature on one scale needs each between minus one and one, and these are not declared to "
                    + $"land there: {string.Join(", ", off)}. Scale each with midrange, min-max or max-abs, write a moment on a circle in a "
                    + "form, or encode a category one-hot.");
            }
        }

        var values = features.Select(column => table.NumbersOf(column.Name)).ToArray();
        var handed = new List<double[]>(rows.Length);
        var madeFrom = MadeFrom(prepared);

        foreach (var row in rows)
        {
            // A refusal names the row as it was read, which is the row a person can find in their file.
            var readAt = table.Identities[row].ReadAt;
            var line = new double[features.Length];

            for (var at = 0; at < features.Length; at++)
            {
                line[at] = Handed(values[at][row], features[at].Name, readAt, madeFrom);
            }

            handed.Add(line);
        }

        string[] names = [.. features.Select(column => column.Name)];

        return new HandedFeatures(
            names,
            handed,
            [.. rows.Select(row => table.Identities[row].Key)],
            prepared.Course.CategoriesHanded(prepared.Fitted).Where(each => names.Contains(each.Key)).ToDictionary(StringComparer.Ordinal));
    }

    // The answers half: each row's answers, as many as the output names, refused as a feature is, and by the output.
    private static HandedAnswers AnswersOf(PreparedData prepared, Table table, int[] rows)
    {
        var answers = AnswersNamed(prepared, table);
        IReadOnlyList<RowKey> keys = [.. rows.Select(row => table.Identities[row].Key)];

        if (prepared.Declaration.Output is not { } output)
        {
            return new HandedAnswers(null, null, keys);
        }

        var known = answers.Select(answer => table.NumbersOf(answer)).ToArray();
        var answered = new List<double[]>(rows.Length);
        var madeFrom = MadeFrom(prepared);

        foreach (var row in rows)
        {
            var readAt = table.Identities[row].ReadAt;
            var rowAnswers = new double[answers.Count];

            for (var at = 0; at < answers.Count; at++)
            {
                rowAnswers[at] = Handed(known[at][row], answers[at], readAt, madeFrom);
            }

            if (output.Refusal(rowAnswers) is { } refusal)
            {
                throw new InvalidOperationException($"Row {readAt + 1} is refused by the output '{output.Verb}': {refusal}");
            }

            answered.Add(rowAnswers);
        }

        return new HandedAnswers(answers, answered, keys);
    }

    // The answers the output names, each of which reached the end of the pipeline.
    private static IReadOnlyList<string> AnswersNamed(PreparedData prepared, Table table)
    {
        var answers = prepared.Declaration.Output?.Answers ?? [];

        return answers.FirstOrDefault(answer => !table.Has(answer)) is { } vanished
            ? throw new InvalidOperationException($"This pipeline predicts '{vanished}', and no column of that name reached the end of it.")
            : answers;
    }

    /// <summary>For each column a step worked out from others, the columns it was worked out from.</summary>
    /// <param name="prepared">The run.</param>
    /// <returns>The columns a step added, each with the columns that step reads; empty for a pipeline that adds none.</returns>
    /// <remarks>
    /// Worked out from the declaration rather than kept anywhere: a step that adds columns says which it reads through
    /// its own parameters, and what it leaves behind is the difference its <see cref="IDescribesColumns.After"/> makes.
    /// </remarks>
    private static IReadOnlyDictionary<string, IReadOnlyList<string>> MadeFrom(PreparedData prepared)
    {
        var made = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
        var standing = ColumnState.None;

        foreach (var step in prepared.Course.Steps)
        {
            var after = step is IDescribesColumns describes ? describes.After(standing) : standing;

            if (step is IAddsColumns && step.ColumnsRead is { Count: > 0 } read)
            {
                foreach (var column in after.Columns.Select(known => known.Name).Except(
                    standing.Columns.Select(known => known.Name), StringComparer.Ordinal))
                {
                    made[column] = [.. read.Select(each => each.Column)];
                }
            }

            standing = after;
        }

        return made;
    }

    /// <summary>A value as it may be handed to something that learns, or the reason it may not.</summary>
    private static double Handed(
        double? value, string column, int readAt, IReadOnlyDictionary<string, IReadOnlyList<string>> madeFrom) => value switch
    {
        // A column worked out from another carries that column's gaps, and filling what it was made from below the
        // split does not reach back into it: the feature was worked out above the split, from the columns as they
        // stood there. So the refusal names where the gap came from rather than only that it is one.
        null => throw new InvalidOperationException(
            madeFrom.TryGetValue(column, out var sources) && sources.Count > 0
                ? $"Row {readAt + 1} of '{column}' is still a gap. It is worked out from "
                  + $"{string.Join(" and ", sources.Select(source => $"'{source}'"))}, so a gap there is a gap here: "
                  + "settle those rows above the step that works it out, or leave the column out."
                : $"Row {readAt + 1} of '{column}' is still a gap. Fill it, drop it, or leave the column out."),

        // A model handed an infinity or a not-a-number learns nothing from that row and says nothing about
        // it. Either is a fault upstream, and the verb for it is fill.nan.
        { } number when !double.IsFinite(number) => throw new InvalidOperationException(string.Create(
            CultureInfo.InvariantCulture,
            $"Row {readAt + 1} of '{column}' is not a finite number ({number}). Declare fill.nan for it, or leave the column out.")),

        { } number => number,
    };
}

/// <summary>What a learner is handed of its features: their names, each row's numbers, each row's key, and the features handed over as places.</summary>
/// <param name="Names">The features, in the order every row lists them.</param>
/// <param name="Rows">Each row's numbers.</param>
/// <param name="Keys">Each row's key, in the order of the rows.</param>
/// <param name="Categories">Each feature handed over as the places of its categories, with the categories.</param>
internal readonly record struct HandedFeatures(
    IReadOnlyList<string> Names,
    IReadOnlyList<double[]> Rows,
    IReadOnlyList<RowKey> Keys,
    IReadOnlyDictionary<string, IReadOnlyList<string>> Categories);

/// <summary>The answers of some rows, and their keys: what the report measures predictions against.</summary>
/// <param name="Names">The answers the output names, in its order; nothing when the pipeline names no output.</param>
/// <param name="Answers">Each row's answers, in the same order; nothing when the pipeline names no output.</param>
/// <param name="Keys">Each row's key, in the order of the rows.</param>
internal readonly record struct HandedAnswers(IReadOnlyList<string>? Names, IReadOnlyList<double[]>? Answers, IReadOnlyList<RowKey> Keys);

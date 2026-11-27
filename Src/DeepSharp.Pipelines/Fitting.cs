// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// A pipeline that has been run: the data, where every row landed, and what each step learned.
/// </summary>
public sealed class PreparedData
{
    /// <summary>A pipeline that has been run, or one loaded back from the file it was saved as.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <param name="table">The data, as the last step left it.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <param name="fitted">What each step learned, by its position in the declaration.</param>
    /// <remarks>
    /// Public because a saved pipeline has to be usable without the data it was fitted on: a serving host
    /// loads the declaration and what it learned, hands in a row, and gets it prepared the way the training
    /// rows were. See <see cref="FromJson(string, StepCatalog)"/>. A run made again with the evidence it produced says it
    /// beside these: <c>new PreparedData(declaration, table, parts, fitted) { Evidence = evidence }</c>. Every declared step
    /// is taken to have been run as it was declared: a run that left steps out for a learner is made by
    /// <see cref="Pipeline.RunFor(Needs)"/>, or read back from the file it wrote.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// A step that learns has nothing it learned here, or something is here for a step that learns nothing.
    /// </exception>
    public PreparedData(
        PipelineDeclaration declaration,
        Table table,
        IReadOnlyList<Part> parts,
        IReadOnlyDictionary<int, FittedStepValues> fitted)
        : this(Course.Whole(declaration), new Walked(table, parts, new Dictionary<int, Evidence>()), fitted)
    {
    }

    /// <summary>A pipeline that has been run, with the evidence its run produced.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <param name="table">The data, as the last step left it.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <param name="fitted">What each step learned, by its position in the declaration.</param>
    /// <param name="evidence">What each step that produces evidence produced, by its position.</param>
    /// <exception cref="ArgumentException">
    /// A step that learns has nothing it learned here, or something is here for a step that learns nothing.
    /// </exception>
    /// <remarks>
    /// Every declared step is taken to have been run as it was declared: a run that left steps out for a learner is made by
    /// <see cref="Pipeline.RunFor(Needs)"/>, or read back from the file it wrote.
    /// </remarks>
    [Obsolete("Use new PreparedData(declaration, table, parts, fitted) { Evidence = evidence }.")]
    public PreparedData(
        PipelineDeclaration declaration,
        Table table,
        IReadOnlyList<Part> parts,
        IReadOnlyDictionary<int, FittedStepValues> fitted,
        IReadOnlyDictionary<int, Evidence> evidence)
        : this(Course.Whole(declaration), new Walked(table, parts, evidence), fitted)
    {
    }

    /// <summary>A pipeline run through a course: the steps as it took them, what the walk left, and what each step learned.</summary>
    /// <param name="course">The steps as the run took them, and which it left out.</param>
    /// <param name="walked">The data as the last step left it, where each row landed, and the evidence.</param>
    /// <param name="fitted">What each step learned, by its position in the declaration.</param>
    /// <exception cref="ArgumentException">
    /// A step the run took that learns has nothing it learned here, or something is here for a step that learned nothing in
    /// this run: a step that learns nothing, or one the run left out.
    /// </exception>
    internal PreparedData(Course course, Walked walked, IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        ArgumentNullException.ThrowIfNull(walked.Table, "table");
        ArgumentNullException.ThrowIfNull(walked.Parts, "parts");
        ArgumentNullException.ThrowIfNull(fitted);
        ArgumentNullException.ThrowIfNull(walked.Evidence, "evidence");

        ThrowIfTheFitsDoNotMatch(course, fitted);

        Course = course;
        Declaration = course.Declaration;
        Table = walked.Table;
        Parts = walked.Parts;
        Fitted = fitted;
        Evidence = walked.Evidence;
    }

    /// <summary>Refuses fits that are not exactly one per step the run took that learns, and one for the split.</summary>
    /// <remarks>
    /// A step that learns and has nothing it learned used to be skipped when the pipeline was replayed, so a
    /// saved declaration without its fitted half served every value unscaled and said nothing about it. The
    /// split's entry is what the fit saw, and a fitted half without it is one nobody can trace to its data. The fits
    /// are held to the steps as the run took them: a step it left out learned nothing.
    /// </remarks>
    private static void ThrowIfTheFitsDoNotMatch(Course course, IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        foreach (var at in fitted.Keys)
        {
            if (at < 0 || at >= course.Steps.Count || !WritesAnEntry(course.Steps[at]))
            {
                throw new ArgumentException(
                    at < 0 || at >= course.Steps.Count
                        ? $"There is no step {at + 1} for anything to have been learned at."
                        : NothingToLearnAt(at, course.Steps[at]),
                    nameof(fitted));
            }
        }

        if (FitsMissing(course, fitted).ToArray() is [var missing, ..])
        {
            throw new ArgumentException(missing.ToString(), nameof(fitted));
        }
    }

    /// <summary>Every step the run took that writes an entry and has none among these fits.</summary>
    /// <param name="course">The steps as the run took them.</param>
    /// <param name="fitted">What was learned, by position.</param>
    /// <returns>A fault for each step whose entry is missing, in the order of the steps.</returns>
    /// <remarks>One rule for this constructor and for the file a pipeline is read from, which says where each is.</remarks>
    internal static IEnumerable<DeclarationFault> FitsMissing(Course course, IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        for (var at = 0; at < course.Steps.Count; at++)
        {
            var step = course.Steps[at];

            if (WritesAnEntry(step) && !fitted.ContainsKey(at))
            {
                yield return new DeclarationFault(
                    at,
                    step.Verb,
                    step is ISplitStep
                        ? "divides the rows, and what it saw of them is not here. Fit the pipeline again."
                        : "learns from the data, and nothing it learned is here. Fit the pipeline again.");
            }
        }
    }

    /// <summary>What is said of a fit that is there for a step that learns nothing.</summary>
    /// <param name="at">The step's place in the course, counting from nought.</param>
    /// <param name="step">The step.</param>
    /// <returns>The sentence, naming the step as a person counts them.</returns>
    internal static string NothingToLearnAt(int at, IPipelineStep step) =>
        $"Step {at + 1}, '{step.Verb}', learns nothing, so nothing it learned can be here.";

    /// <summary>Whether a fit writes an entry for this step: each step that learns, and the split it learned behind.</summary>
    /// <param name="step">The step.</param>
    /// <returns><see langword="true"/> for a step that learns and for the split.</returns>
    internal static bool WritesAnEntry(IPipelineStep step) => step is IFittedStep or ISplitStep;

    /// <summary>Loads a saved pipeline, reading its steps with the verbs a given catalog knows.</summary>
    /// <param name="json">The file the pipeline was written as.</param>
    /// <param name="catalog">The verbs that may appear in it, another package's included.</param>
    /// <returns>The pipeline, with no data and everything it learned.</returns>
    /// <exception cref="PipelineFileException">
    /// Anything in the file is wrong, every fault named at its line and column: a step that cannot be read, a
    /// rule the steps break, a step that learned and has no entry, or an entry learned behind other steps than
    /// these — which is what a declaration edited after its fit looks like.
    /// </exception>
    /// <remarks>
    /// The table that comes back is empty, because a saved pipeline carries no data — that is the point of
    /// saving it. Hand rows to <see cref="Replay"/> and they are prepared exactly as the training rows were.
    /// The catalog is handed over rather than assumed, because a verb another package brings has to be as
    /// readable here as anywhere, or a pipeline could be saved and never served.
    /// </remarks>
    public static PreparedData FromJson(string json, StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(catalog);

        return Loaded(PipelineDocument.ReadPipeline(json, catalog));
    }

    /// <summary>Loads a saved pipeline that stands inside a larger file, as the value of one of the keys at its top.</summary>
    /// <param name="json">The larger file: one JSON object, holding the pipeline's file under one of its keys.</param>
    /// <param name="catalog">The verbs that may appear in the pipeline, another package's included.</param>
    /// <param name="property">The key the pipeline stands under.</param>
    /// <returns>The pipeline, with no data and everything it learned.</returns>
    /// <exception cref="ArgumentException">The key is empty.</exception>
    /// <exception cref="PipelineFileException">
    /// The larger file is not JSON, not one object, holds the key twice or not at all, or anything in the pipeline is wrong
    /// as <see cref="FromJson(string, StepCatalog)"/> finds it — every fault named at its line and column in the larger file.
    /// </exception>
    /// <remarks>
    /// For a file that carries a pipeline beside something else, such as a trained network and the pipeline it was trained
    /// behind. The pipeline is read exactly as its own file is, and a fault is placed where it stands in the file a person
    /// opens, not in the pipeline cut out of it: a fault on that file's line 313 is on line 313. What the rest of the file
    /// holds is left to whatever reads it.
    /// </remarks>
    public static PreparedData FromJson(string json, StepCatalog catalog, string property)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentException.ThrowIfNullOrWhiteSpace(property);

        return Loaded(PipelineDocument.ReadPipelineIn(json, catalog, property));
    }

    // A saved pipeline as it was read: the steps as its run took them, what they learned, and no rows.
    private static PreparedData Loaded(SavedPipeline saved) =>
        new(Course.Of(saved.Declaration, saved.Skipped), new Walked(new Table([]), [], new Dictionary<int, Evidence>()), saved.Fitted);

    /// <summary>The steps, exactly as they were declared.</summary>
    public PipelineDeclaration Declaration { get; }

    /// <summary>
    /// The places of the steps the run left out for the learner it was run for, in ascending order, counting from nought:
    /// each a step that learner does without, left out or taken in the form it offers; empty for a run of every step.
    /// </summary>
    /// <remarks>
    /// What <see cref="Pipeline.RunFor(Needs)"/> decided from the declaration and the need, kept with the run and written into
    /// its file, so a replay leaves out the same steps and a learner is handed the run only when it does without each of them.
    /// </remarks>
    public IReadOnlyList<int> Skipped => Course.Skipped;

    /// <summary>The steps as the run took them: every one as it was declared, or as the learner it was run for takes it.</summary>
    internal Course Course { get; }

    /// <summary>The data, as the last step left it.</summary>
    public Table Table { get; }

    /// <summary>Which part of the data each row belongs to, in row order.</summary>
    public IReadOnlyList<Part> Parts { get; }

    /// <summary>What each step learned, by its position in the declaration.</summary>
    /// <remarks>
    /// Keyed by position rather than by verb, because a declaration may hold the same step twice and two
    /// identical steps are indistinguishable by value.
    /// </remarks>
    public IReadOnlyDictionary<int, FittedStepValues> Fitted { get; }

    /// <summary>The columns the ways back need, as the rows were read: kept by a run, and by nothing else.</summary>
    internal ColumnsAsRead AsRead { get; init; } = ColumnsAsRead.None;

    /// <summary>What each step that produces evidence produced when the pipeline was run, by its position.</summary>
    /// <remarks>
    /// Output, not the pipeline: it is kept with the run and never written into the pipeline's file, so a
    /// pipeline loaded from one has none, and a replay produces none. A run made again from its parts is handed the
    /// evidence it produced here; one made without it has none.
    /// </remarks>
    /// <exception cref="ArgumentNullException">The evidence handed in is nothing.</exception>
    public IReadOnlyDictionary<int, Evidence> Evidence
    {
        get;
        init => field = value ?? throw new ArgumentNullException(nameof(Evidence));
    }

    /// <summary>How many rows landed in one split.</summary>
    /// <param name="part">The part to count.</param>
    /// <returns>The number of rows.</returns>
    public int CountIn(Part part) => Parts.Count(each => each == part);

    /// <summary>
    /// The columns a column of words was written down as, one for each category: what a layer that reads a category from its
    /// columns reads.
    /// </summary>
    /// <param name="column">The column of words, as it was named before it was written down.</param>
    /// <returns>
    /// The names of its columns, in order: each category the training rows held, in the order the encoder learned them; the
    /// column kept for a category they never held, where the encoder keeps one; and the column that marks a gap. A row holds a
    /// one in exactly one of them.
    /// </returns>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <exception cref="InvalidOperationException">No step this run took wrote the column down one column per category.</exception>
    /// <remarks>
    /// The encoder's own naming is the contract, so a layer reads the block by the rule that wrote it rather than by the start of
    /// a name another column could share.
    /// </remarks>
    public IReadOnlyList<string> CategoryBlock(string column)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        return Course.CategoryBlock(column, Fitted)
            ?? throw new InvalidOperationException(
                $"No step of this pipeline writes '{column}' down one column per category, so it has no block of categories to read: encode it with as: oneHot.");
    }

    /// <summary>Refuses a pipeline that holds no rows, as one read from its file holds none: it has no part to hand over and nothing to measure.</summary>
    /// <exception cref="InvalidOperationException">The pipeline holds no rows.</exception>
    /// <remarks>The one refusal for it, wherever a part's rows are asked for — handed to a learner, or measured by the report.</remarks>
    internal void RequireRows()
    {
        if (Table.RowCount == 0)
        {
            throw new InvalidOperationException(
                "This pipeline holds no rows: one read from its file keeps what its steps learned, not the rows they learned it from, "
                + "so it has no part to hand over and nothing to measure. Run its declaration over the rows — "
                + "new Pipeline(declaration, rows, folder).Run() — and hand that run over; rows that arrive later are served through this one.");
        }
    }

    /// <summary>Runs the same declaration over new rows, with the same numbers it learned before.</summary>
    /// <param name="rows">The rows to prepare — one of them, or a million.</param>
    /// <returns>The new rows, prepared exactly as the training data was.</returns>
    /// <exception cref="InvalidOperationException">The declaration names no columns.</exception>
    /// <remarks>
    /// This is what serving is. Nothing is fitted again and nothing is learned: the means, the fill values
    /// and the category lists are the ones the training rows produced, so a row arriving a year from now
    /// meets exactly the numbers the model was trained on. It is also why a model without its pipeline
    /// cannot be used at all. The steps are replayed as the run took them: a step it left out for its learner
    /// (<see cref="Skipped"/>) is left out here too, so a served row is handed over as the training rows were.
    /// </remarks>
    public Table Replay(IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        return new Walk(Course, new ReplayWhatWasFitted(Fitted), SourceFolder.WorkingDirectory).Through(rows).Table;
    }

    /// <summary>Puts predictions back into the units the target was read in.</summary>
    /// <param name="predictions">What a model said, in the units it was trained on.</param>
    /// <returns>The same numbers, in the units of the column the pipeline started from.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline names no answer, or several, or a step on the way to it cannot be undone.
    /// </exception>
    /// <remarks>
    /// The steps that touched the answer are walked backwards, each undoing what it did, and on through the
    /// column an answer was made from. A step that cannot be undone — one that clipped, or blanked, or threw
    /// information away — says so rather than quietly handing back a number in the wrong units, which is the
    /// failure this exists to prevent. So does a step whose way back needs the row a number belongs to: hand the
    /// rows over with the predictions.
    /// </remarks>
    public IReadOnlyList<double> BackToOriginal(IEnumerable<double> predictions)
    {
        ArgumentNullException.ThrowIfNull(predictions);

        var target = Declaration.Output switch
        {
            { Answers: [var only] } => only,
            null => throw NothingToComeBackTo(),
            var output => throw new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"This pipeline's output names {output.Answers.Count} answers, and one number goes back into the units of one of them.")),
        };

        var chain = UndoChain.For(Declaration, Fitted, target);

        return [.. predictions.Select(value => chain.Back(value))];
    }

    /// <summary>Puts one prediction back into the units the target was read in.</summary>
    /// <param name="prediction">What a model said.</param>
    /// <returns>The number in the units of the column the pipeline started from.</returns>
    public double BackToOriginal(double prediction) => BackToOriginal([prediction])[0];

    /// <summary>Puts predictions for the rows of one part back into the units each answer was read in.</summary>
    /// <param name="predictions">
    /// What a model said for the rows of the part, in the order <see cref="Handover.Batch(PreparedData, Part, Needs)"/> hands them over: one
    /// number for each answer the output names, in its order.
    /// </param>
    /// <param name="part">The part the predictions are for.</param>
    /// <returns>The same numbers, each in the units of the column its answer comes back to.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline names no answer, or a step on the way back cannot be undone.
    /// </exception>
    /// <exception cref="ArgumentException">There is not one prediction per row, or not one number per answer.</exception>
    /// <remarks>
    /// Each prediction comes back with the row it was made for as it was read, so a way back that needs more than
    /// the number — a share of the birds in the row, a return on its price — comes back too.
    /// </remarks>
    public IReadOnlyList<double[]> BackToOriginal(IReadOnlyList<double[]> predictions, Part part)
    {
        ArgumentNullException.ThrowIfNull(predictions);

        RowAsRead[] rows = [.. this.ReadAtOf(part).Select(readAt => new RowAsRead(AsRead, readAt))];

        return Back(predictions, rows);
    }

    /// <summary>Puts predictions for served rows back into the units each answer was read in.</summary>
    /// <param name="predictions">
    /// What a model said for each served row, in the order they were served: one number for each answer the output
    /// names, in its order.
    /// </param>
    /// <param name="served">What <see cref="Handover.Served(PreparedData, IRowSource, Needs)"/> handed over for these rows.</param>
    /// <param name="rows">The rows that were handed in to be served, in the order they were handed in.</param>
    /// <returns>The same numbers, each in the units of the column its answer comes back to.</returns>
    /// <exception cref="InvalidOperationException">
    /// The rows handed in are not the rows that were served, the pipeline names no answer, or a step on the way back
    /// cannot be undone.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// The served rows carry no keys, there is not one prediction per row, or not one number per answer.
    /// </exception>
    /// <remarks>
    /// Each served row is found again among the rows handed in by where it was handed in, and its key is checked:
    /// rows handed in again in another order would otherwise be answered with another row's numbers.
    /// </remarks>
    public IReadOnlyList<double[]> BackToOriginal(IReadOnlyList<double[]> predictions, ServedBatch served, IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(predictions);

        return Back(predictions, Found(served, rows));
    }

    // The served rows found again among the rows handed in, each by where it was handed in and checked by its key.
    private RowAsRead[] Found(ServedBatch served, IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        if (served.Keys is not { } keys || keys.Count != served.HandedInAt.Count)
        {
            throw new ArgumentException("These served rows carry no keys to find them by: hand over what Served returned.", nameof(served));
        }

        // Read the way the replay read them, so each row is keyed as it was when it was served.
        var read = new Walk(Course, new ReplayWhatWasFitted(Fitted), SourceFolder.WorkingDirectory).Bound(rows);
        var asRead = ColumnsAsRead.Of(Declaration, read);
        var found = new RowAsRead[keys.Count];

        for (var at = 0; at < found.Length; at++)
        {
            var handedIn = served.HandedInAt[at];

            if (handedIn >= read.RowCount || read.Identities[handedIn].Key != keys[at])
            {
                throw new InvalidOperationException(
                    $"Served row {at + 1} was handed in as row {handedIn + 1}, and the rows handed in now hold something else "
                    + "there. Hand in the rows that were served, in the order they were served.");
            }

            found[at] = new RowAsRead(asRead, handedIn);
        }

        return found;
    }

    /// <summary>Puts several members' predictions for served rows back into the units each answer was read in, and combines them there.</summary>
    /// <param name="members">For each member, what it said for each served row, in the order they were served.</param>
    /// <param name="served">What <see cref="Handover.Served(PreparedData, IRowSource, Needs)"/> handed over for these rows.</param>
    /// <param name="rows">The rows that were handed in to be served, in the order they were handed in.</param>
    /// <param name="blend">How the members' answers become one.</param>
    /// <returns>One answer for each served row, in the units of the column each answer comes back to.</returns>
    /// <exception cref="ArgumentException">There are no members; or, as <see cref="BackToOriginal(IReadOnlyList{double[]}, ServedBatch, IRowSource)"/> refuses them.</exception>
    /// <exception cref="InvalidOperationException">
    /// As <see cref="BackToOriginal(IReadOnlyList{double[]}, ServedBatch, IRowSource)"/> refuses; or the members vote on answers that are amounts.
    /// </exception>
    /// <remarks>The rows are walked once, whatever the number of members.</remarks>
    public IReadOnlyList<double[]> BackToOriginal(IReadOnlyList<IReadOnlyList<double[]>> members, ServedBatch served, IRowSource rows, Blend blend)
    {
        blend.ThrowIfVotedOnAmounts([this]);

        return blend.CombinedBack([.. BackEach(members, served, rows).Select(back => new ServedAnswers(back, served))], Declaration.Output?.Ones ?? 0).Answers;
    }

    /// <summary>Puts several members' predictions for the same served rows back into the units each answer was read in, each apart.</summary>
    /// <param name="members">For each member, what it said for each served row, in the order they were served.</param>
    /// <param name="served">What was handed over for these rows.</param>
    /// <param name="rows">The rows that were handed in to be served, in the order they were handed in.</param>
    /// <returns>For each member, one answer for each served row, in the units of the column each answer comes back to.</returns>
    /// <exception cref="ArgumentException">There are no members; or, as <see cref="BackToOriginal(IReadOnlyList{double[]}, ServedBatch, IRowSource)"/> refuses them.</exception>
    /// <exception cref="InvalidOperationException">As <see cref="BackToOriginal(IReadOnlyList{double[]}, ServedBatch, IRowSource)"/> refuses.</exception>
    /// <remarks>The rows are walked once, whatever the number of members.</remarks>
    internal IReadOnlyList<IReadOnlyList<double[]>> BackEach(IReadOnlyList<IReadOnlyList<double[]>> members, ServedBatch served, IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(members);

        if (members.Count == 0)
        {
            throw new ArgumentException("An ensemble's answers are combined from at least one member's.", nameof(members));
        }

        var found = Found(served, rows);

        return [.. members.Select(member => Back(member, found))];
    }

    private IReadOnlyList<double[]> Back(IReadOnlyList<double[]> predictions, RowAsRead[] rows)
    {
        var answers = Declaration.Output?.Answers ?? throw NothingToComeBackTo();

        if (predictions.Count != rows.Length)
        {
            throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"There are {predictions.Count} predictions for {rows.Length} rows, and each row has one."),
                nameof(predictions));
        }

        UndoChain[] chains = [.. answers.Select(answer => UndoChain.For(Declaration, Fitted, answer))];
        var back = new double[rows.Length][];

        for (var row = 0; row < rows.Length; row++)
        {
            var said = predictions[row];

            if (said.Length != chains.Length)
            {
                throw new ArgumentException(
                    string.Create(CultureInfo.InvariantCulture, $"Row {row + 1} has {said.Length} predictions, and the output names {chains.Length} answers."),
                    nameof(predictions));
            }

            back[row] = [.. chains.Select((chain, at) => chain.Back(said[at], rows[row]))];
        }

        return back;
    }

    private static InvalidOperationException NothingToComeBackTo() =>
        new("This pipeline names no answer, so there is nothing to put back into any units.");

    /// <summary>
    /// Measures what a model predicted by this pipeline's report: on each part it names, in the answer's own units, each
    /// measure beside the same measure of predicting the training rows' average answer.
    /// </summary>
    /// <param name="predictions">
    /// For every part the report names, what the model predicted for the batch <see cref="Handover.Batch(PreparedData, Part, Needs)"/>
    /// handed over for it: one row of predictions per row, in the batch's order, in the units the answers were handed over in.
    /// </param>
    /// <returns>The measures, part by part and measure by measure in the order the report names them.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline declares no report; it holds no rows, as one read from its file holds none; a part it names holds no rows,
    /// or the training part holds none; R² is asked of a part of one row; or, where the report counts classes, a row's answer
    /// is neither nought nor one, or holds another number of ones than its output says.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// A part the report names has no predictions, or two sets; predictions are for a part it does not name, for a batch the
    /// pipeline did not hand over, or for other rows or another order than the part's; there is not one prediction per row
    /// and per answer; a prediction comes back as a number that is not finite; or what the predictions say is unfamiliar is
    /// not said of one row each.
    /// </exception>
    /// <remarks>
    /// The predictions and the answers they are compared with both come back through the way back, because in the units
    /// they were handed over in every error is small and every model looks excellent. Beside each measure stands what
    /// predicting, for every row, the average of the training rows' answers measures — brought back and measured the same
    /// way — so no number stands alone; and beside a part's rows, when the predictions say it, how many of them the model
    /// learned nothing about (<see cref="PartPredictions.Unfamiliar"/>). What is measured is output, kept with whatever
    /// measured it and never written into the pipeline's file. Only the rows' keys and answers are read, never the features a
    /// learner was handed, so a learner handed each category as its place is measured on the same rows as any other.
    /// </remarks>
    public Measures Measure(IReadOnlyList<PartPredictions> predictions)
    {
        ArgumentNullException.ThrowIfNull(predictions);

        return Measurement.Of(this, predictions);
    }

    /// <summary>
    /// Measures again, by this pipeline's report and on this run, what a model predicted for the parts the report names,
    /// handed over as the text <see cref="Measures.PredictionsToJson"/> writes.
    /// </summary>
    /// <param name="predictions">
    /// The text: for each part, the key of each row, what was predicted for it in the units the answers were handed over in,
    /// and what the model said it learned nothing about.
    /// </param>
    /// <returns>The measures, part by part and measure by measure in the order the report names them.</returns>
    /// <exception cref="InvalidOperationException">
    /// As <see cref="Measure"/> refuses: the pipeline declares no report, a part holds no rows, R² of one row, or answers
    /// that are not the classes the report counts.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// The text is not what <see cref="Measures.PredictionsToJson"/> writes, or a newer DeepSharp wrote it; the predictions
    /// answer other answers than this pipeline's output names; they were made behind another fit than this one, behind a
    /// pipeline of a version of the pipeline file no comparison knows, or behind a run that took a step this run left out; or,
    /// as <see cref="Measure"/> refuses them, they are for other rows than this run's parts hand over — known by the rows'
    /// keys — or in another order, or are not one for each row and answer.
    /// </exception>
    /// <remarks>
    /// This is how predictions made where this library's types are other types — a notebook's C# cell — are measured where
    /// they are shown: the one measurement <see cref="Measure"/> makes, on the run that reads the text, and only beside the
    /// very fit they were made behind. That is judged as a network's file is held to its pipeline: both pipelines through one
    /// writer, the version each names left out (<see cref="PipelineText"/>), so predictions made behind other steps, or behind
    /// numbers learned from other rows — a feature scaled another way, or the answer — are refused before anything is
    /// measured, as are those of another output, by the names of its answers. Predictions made behind a run for a learner
    /// that left steps out (<see cref="Pipeline.RunFor(Needs)"/>) say which, and are measured again on a run of every step of
    /// the same declaration, which learned all they were made behind, as well as on the very run they were made behind.
    /// </remarks>
    public Measures MeasureAgain(string predictions)
    {
        ArgumentNullException.ThrowIfNull(predictions);

        var written = WrittenPredictions.FromJson(predictions);

        if (Declaration.Output is { } output && !output.Answers.SequenceEqual(written.Answers))
        {
            throw new ArgumentException(
                $"These predictions answer {written.Answers.Quoted()}, and this pipeline answers {output.Answers.Quoted()}: they were made for another pipeline's output.",
                nameof(predictions));
        }

        written.Behind.ThrowIfNotBehind(this);

        return Measurement.Of(this, [.. written.Parts.Select(part => part.ToPartPredictions())]);
    }

    /// <summary>
    /// Measures several members' predictions as one ensemble, and each member alone, by this pipeline's report: every member is
    /// brought back into the answer's own units first, and the ensemble is combined there.
    /// </summary>
    /// <param name="members">
    /// For each member, what it predicted for every part the report names, as <see cref="Measure"/> takes one member's: in the
    /// units the answers were handed over in. At least two.
    /// </param>
    /// <param name="blend">How the members' answers become one: averaged, unless said.</param>
    /// <param name="interval">The interval the ensemble was given, so the report's coverage and width are measured of it; none, and they are left out.</param>
    /// <returns>The ensemble's measures beside each member's, on the very same rows.</returns>
    /// <exception cref="ArgumentException">There are fewer than two members; or a member's predictions are refused as <see cref="Measure"/> refuses them.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The blend is none of the blends.</exception>
    /// <exception cref="InvalidOperationException">As <see cref="Measure"/> refuses; or the members vote on answers that are amounts.</exception>
    /// <remarks>
    /// A way back is not a straight line — a log is undone by an exponential, and the mean of exponentials is not the exponential
    /// of the mean — so the members meet in the answer's own units and not before. An ensemble's measures hold what the members
    /// said together, which no one text holds: each member's measures cross as text, the ensemble's do not.
    /// </remarks>
    public EnsembleMeasures MeasureEnsemble(IReadOnlyList<IReadOnlyList<PartPredictions>> members, Blend blend = Blend.Average, ConformalInterval? interval = null)
    {
        ArgumentNullException.ThrowIfNull(members);

        blend.ThrowIfVotedOnAmounts([this]);

        return Measurement.OfEnsemble([.. members.Select(member => new MemberPredictions(this, member))], blend, interval);
    }

    /// <summary>The interval a model's predictions for the calibration rows allow, at a level: split conformal prediction.</summary>
    /// <param name="calibration">What the model predicted for the rows the pipeline set aside to calibrate on, as <see cref="Measure"/> takes predictions: in the units the answers were handed over in.</param>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <returns>A margin for each answer, in its own units.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">The pipeline sets no rows aside; its answers are classes; or the rows hold no margin that keeps the promise.</exception>
    /// <exception cref="ArgumentException">The predictions are not for the calibration rows, or not one for each of them.</exception>
    /// <remarks>
    /// The margin is the ⌈(n+1)(1−α)⌉-th smallest of the errors on the n calibration rows, in the answer's own units — the errors are
    /// worked out after each prediction has come back through the way back. A row the model has not seen then lands within the
    /// margin of what the model says with a chance of at least 1−α, whatever the model is, as long as the rows are exchangeable with
    /// the calibration rows. It says nothing of any one row, and a series whose later rows differ from its earlier ones breaks the
    /// promise it rests on.
    /// </remarks>
    public ConformalInterval IntervalOf(PartPredictions calibration, double alpha) =>
        Conformal.Of([new MemberPredictions(this, [calibration])], Blend.Average, alpha);

    /// <summary>The interval several members\' predictions for the calibration rows allow, once combined, at a level.</summary>
    /// <param name="members">What each member predicted for the calibration rows, in the units the answers were handed over in. At least one.</param>
    /// <param name="blend">How the members\' answers become one.</param>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <returns>A margin for each answer, in its own units, of the combined answer.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">As <see cref="IntervalOf(PartPredictions, double)"/> refuses; or the members vote on answers that are amounts.</exception>
    /// <exception cref="ArgumentException">As <see cref="IntervalOf(PartPredictions, double)"/> refuses, or there are no members.</exception>
    /// <remarks>Each member comes back into the answer\'s own units first and the errors are those of what they say together.</remarks>
    public ConformalInterval IntervalOf(IReadOnlyList<PartPredictions> members, Blend blend, double alpha)
    {
        ArgumentNullException.ThrowIfNull(members);

        blend.ThrowIfVotedOnAmounts([this]);

        return Conformal.Of([.. members.Select(member => new MemberPredictions(this, [member]))], blend, alpha);
    }

    /// <summary>
    /// This run's file as a run of the same declaration that left out these steps would have written it: each of them that
    /// learns nothing in the form the run takes it has no entry, and they are named under <c>skipped</c>.
    /// </summary>
    /// <param name="skipped">The places of the steps left out, which <see cref="Course.FaultsIn"/> finds nothing wrong with, and among which every step this run left out stands.</param>
    /// <returns>The text, as <see cref="ToJson"/> writes a run's.</returns>
    /// <remarks>
    /// A run for a learner learns what the run of every step learns, but for what it left out, so predictions made behind it
    /// are held to the run of every step through this text.
    /// </remarks>
    internal string TextWithout(IReadOnlyList<int> skipped)
    {
        var course = Course.Of(Declaration, skipped);

        return PipelineDocument.Write(course, Fitted.Where(entry => WritesAnEntry(course.Steps[entry.Key])).ToDictionary());
    }

    /// <summary>
    /// Writes the whole pipeline: the version, what was declared, what the fit learned, and — for a run that left steps out
    /// for its learner — which it left out.
    /// </summary>
    /// <returns>The pipeline as one JSON document.</returns>
    /// <remarks>
    /// Both halves in one file, because a model without what its pipeline learned cannot be used: the
    /// numbers reaching it would not be the numbers it was trained on. Each entry of the fitted half names
    /// the step that learned it and the key of the steps it was learned behind
    /// (<see cref="PipelineDeclaration.KeyAt"/>), which is what keeps a fit from being served under steps
    /// that changed after it. A run that left steps out names each under <c>skipped</c>, by its verb and the same key, so a
    /// replay leaves out the same steps; a run of every step writes nothing more than it always did.
    /// </remarks>
    public string ToJson() => PipelineDocument.Write(Course, Fitted);
}

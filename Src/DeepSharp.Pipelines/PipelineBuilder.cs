// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// A pipeline being written, before it has been told how to split.
/// </summary>
/// <remarks>
/// Everything offered here is arithmetic on a row: where the data comes from, which columns are derived
/// from which. Nothing here learns anything from the data as a whole — the operations that do are not
/// methods on this type, and the one door that takes a step from another package refuses a step that says
/// it learns. They arrive with <see cref="FittingBuilder"/>, which is only reachable by splitting.
/// </remarks>
public sealed class PipelineBuilder
{
    private readonly List<IPipelineStep> _steps = [];
    private bool _split;
    private double _predict;
    private IRowSource? _rows;

    internal PipelineBuilder()
    {
    }

    /// <summary>What has been declared so far.</summary>
    public PipelineDeclaration Declaration => new(_steps);

    /// <summary>Adds a declared step. Every verb, including one from another package, comes through here.</summary>
    /// <param name="step">The step to declare. It must not be one that learns from the data.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="InvalidOperationException">
    /// The step learns from the data, or this builder has already been split and is finished.
    /// </exception>
    public PipelineBuilder Add(IPipelineStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        ThrowIfSplit();

        // Refused where it is written, by the same rules the declaration keeps — a step that learns, above a
        // split not written yet, among them — rather than at the moment somebody asks for the declaration and
        // has to work out which line made it wrong.
        PipelineDeclaration.ThrowIfFaulty([.. _steps, step]);

        _steps.Add(step);

        return this;
    }

    /// <summary>Declares that the rows are handed in rather than opened by the pipeline.</summary>
    /// <param name="rows">The rows, for this run.</param>
    /// <param name="description">What they are, for whoever reads the saved file later.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// The escape hatch that keeps the list of readers short: anything a package can turn into rows and a
    /// declared schema comes in here, and the file records what it was rather than pretending it can open
    /// it again by itself.
    /// </remarks>
    public PipelineBuilder Read(IRowSource rows, string description)
    {
        ArgumentNullException.ThrowIfNull(rows);

        _rows = rows;

        return Add(new ReadRowsStep(description));
    }

    /// <summary>Declares which columns take part, what they hold, and what becomes of the rest.</summary>
    /// <param name="schema">Names the columns, in the order they should reach a model.</param>
    /// <param name="remainder">What becomes of the columns the schema does not name.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">There are no columns, or one is declared twice.</exception>
    public PipelineBuilder Declare(Action<SchemaBuilder> schema, Remainder remainder = Remainder.Drop)
    {
        ArgumentNullException.ThrowIfNull(schema);

        var builder = new SchemaBuilder();
        schema(builder);

        return Add(new DeclareStep(builder.Columns, remainder));
    }

    /// <summary>Puts the rows in order by one or more columns, smallest first.</summary>
    /// <param name="columns">The columns to order by, the one that decides first first.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// What a moving average, a warm-up drop and a gap filled with the value before it need above them: the
    /// order they read is then the one declared here, whichever order the file or the query handed over.
    /// </remarks>
    public PipelineBuilder OrderBy(params string[] columns) => Add(new OrderByStep(columns));

    /// <summary>Adds a column worked out from two others.</summary>
    /// <param name="name">What the new column is called.</param>
    /// <param name="left">The column on the left.</param>
    /// <param name="arithmetic">What to do with them.</param>
    /// <param name="right">The column on the right.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public PipelineBuilder AddFeature(string name, string left, Arithmetic arithmetic, string right) =>
        Add(new AddFeatureStep(name, left, arithmetic, right));

    /// <summary>Writes a moment in time as a place on a circle, so its ends meet.</summary>
    /// <param name="column">The column holding the moment.</param>
    /// <param name="period">Which cycle to place it on.</param>
    /// <param name="form">How to write the two values down.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public PipelineBuilder Cyclical(string column, Period period, Form form = Form.Signed) =>
        Add(new CyclicalStep(column, period, form));

    /// <summary>Pulls a column into another shape, by arithmetic that learns nothing.</summary>
    /// <param name="column">The column to reshape.</param>
    /// <param name="maths">Which shape: a logarithm, a root, a reciprocal.</param>
    /// <param name="into">What to call the result; the same column, unless you say otherwise.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// The variance-stabilising transformations. They belong here, before the split, because the logarithm
    /// of a number does not depend on any other number — and a column of money or of volume usually wants
    /// one, because there the ratio carries the meaning and the difference does not.
    /// </remarks>
    public PipelineBuilder Reshape(string column, Maths maths, string? into = null) =>
        Add(new MathsStep(column, maths, into));

    /// <summary>Takes a moment in time apart into the pieces people reason with.</summary>
    /// <param name="column">The column holding the moment.</param>
    /// <param name="parts">Which pieces to take out of it.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// The pieces arrive as categories, because a month is not a quantity: March is not three of anything,
    /// and December is not twelve times January. <see cref="FittingBuilder.EncodeCategories"/> then turns them into the
    /// numbers a model can take. Use <see cref="TimePartsAsNumbers"/> where the order is the point.
    /// </remarks>
    public PipelineBuilder TimeParts(string column, params TimePart[] parts) =>
        Add(new TimePartsStep(column, parts));

    /// <summary>Takes a moment in time apart, as numbers rather than as groups.</summary>
    /// <param name="column">The column holding the moment.</param>
    /// <param name="parts">Which pieces to take out of it.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// For a piece where the order is the point — a year across a long span, where a column per year would
    /// be refused the first time next year arrives.
    /// </remarks>
    public PipelineBuilder TimePartsAsNumbers(string column, params TimePart[] parts) =>
        Add(new TimePartsStep(column, parts, asCategories: false));

    /// <summary>Drops the rows at the start that no column can speak for yet.</summary>
    /// <param name="atMost">The most rows this is allowed to drop; beyond it the run stops.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// What you want as soon as there are indicators on the data. A twenty-period average says nothing
    /// about the first nineteen rows, and filling them would invent measurements nobody took: with
    /// indicators, 506 rows are 487 rows of data and nineteen rows of not-yet.
    /// </remarks>
    public PipelineBuilder DropWarmUp(int atMost = 1000) => Add(new DropWarmUpStep(atMost));

    /// <summary>Leaves columns out from here on.</summary>
    /// <param name="columns">The columns to leave out.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// For a column the schema has to name — because a step reads it first, or made it — and nobody wants
    /// handed to a model. A column nobody needs at all is simply left out of the schema.
    /// </remarks>
    public PipelineBuilder Drop(params string[] columns) => Add(new DropColumnsStep(columns));

    /// <summary>Profiles the columns where it stands, on the rows the split trains on.</summary>
    /// <param name="columns">The columns to profile; none, for every column here.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>Evidence, declared before the numbers exist; the run keeps it, the file does not.</remarks>
    public PipelineBuilder Profile(params string[] columns) => Add(new ProfileStep(columns));

    /// <summary>Sets out the rows a correlation between columns is drawn from, on the rows the split trains on.</summary>
    /// <param name="columns">Two or more columns holding numbers.</param>
    /// <param name="shown">Drawn, or as the numbers themselves.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public PipelineBuilder Correlation(IEnumerable<string> columns, Shown shown = Shown.Drawn) =>
        Add(new CorrelationStep(columns, shown));

    /// <summary>Drops every row that has a gap in any of these columns.</summary>
    /// <param name="columns">The columns a row may not have a gap in.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>
    /// Here, above the split, where every part loses the row alike. Below it the parts would shrink without
    /// knowing, which is why the split is only offered after this.
    /// </remarks>
    public PipelineBuilder DropGaps(params string[] columns) => Add(new DropGapsStep(columns));

    /// <summary>Holds a share of the rows back, to predict on once a model has been trained.</summary>
    /// <param name="share">How much to hold back, as a fraction or as a percentage.</param>
    /// <returns>This builder, so the split can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The share is not a share.</exception>
    /// <exception cref="InvalidOperationException">A share has already been held back, or this builder has been split.</exception>
    /// <remarks>
    /// A fourth part, outside the three a model is trained and measured with: <c>Predict(10)</c> keeps a
    /// tenth of the rows out of everything, and training, validation and test are then the ninety that
    /// remain. Nothing is fitted on it and nothing is measured on it — it is there to be run through the
    /// trained network, the way the data that arrives tomorrow will be.
    /// </remarks>
    public PipelineBuilder Predict(double share)
    {
        ThrowIfSplit();

        if (!double.IsFinite(share) || share <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(share), share, "A share to predict on is more than none of the data.");
        }

        if (_predict > 0)
        {
            throw new InvalidOperationException(
                $"This pipeline already holds {_predict} back to predict on, and a second share would "
                + "quietly replace the first.");
        }

        _predict = share;

        return this;
    }

    /// <summary>Splits the rows by where they sit in time, and opens the half of the chain that learns.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models; none, unless you say.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <exception cref="ArgumentException">The column has no name, or the shares ask for more than there is.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share.</exception>
    /// <exception cref="InvalidOperationException">This builder has already been split.</exception>
    /// <remarks>
    /// The earliest rows to learn from, the latest to be measured on, which is the only honest division for
    /// data that arrives in order. The share to be measured on is never written down: it is what is left
    /// once training, validation and anything held back to predict on have been taken, so nothing can add
    /// up to more than everything there is. <c>80, 10</c> and <c>0.80, 0.10</c> say the same thing.
    /// </remarks>
    public FittingBuilder SplitByTime(string column, double train, double validation = 0) =>
        Split(new SplitByTimeStep(column, SplitShares.Of(train, validation, _predict)));

    /// <summary>Splits the rows by where they sit in time, keeping the last moments of every part apart.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models; nought for none.</param>
    /// <param name="gap">How many of the last moments of every part are kept apart: at least as many as the rows an answer reads ahead.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <exception cref="ArgumentException">The column has no name, or the shares ask for more than there is.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share, or the gap is below nought.</exception>
    /// <exception cref="InvalidOperationException">This builder has already been split.</exception>
    /// <remarks>
    /// For an answer read from later rows: without a gap, the last training rows learn their answers from the rows a
    /// model is measured on. The rows kept apart are fitted on by nothing and handed to nothing.
    /// </remarks>
    public FittingBuilder SplitByTime(string column, double train, double validation, int gap) =>
        Split(new SplitByTimeStep(column, SplitShares.Of(train, validation, _predict), gap));

    /// <summary>Splits the rows at random, and opens the half of the chain that learns.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models; none, unless you say.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <remarks>
    /// The right split for rows that do not depend on one another. The share to be measured on is worked
    /// out rather than written, as it is everywhere here.
    /// </remarks>
    public FittingBuilder SplitAtRandom(double train, double validation = 0, int seed = 20260923) =>
        Split(new SplitAtRandomStep(SplitShares.Of(train, validation, _predict), seed));

    /// <summary>Splits at random while keeping the mixture of one column the same in every part.</summary>
    /// <param name="column">The column whose mixture is kept.</param>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models; none, unless you say.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <remarks>The right split when an answer is rare enough that a plain shuffle could lose it.</remarks>
    public FittingBuilder SplitStratified(
        string column, double train, double validation = 0, int seed = 20260923) =>
        Split(new SplitStratifiedStep(column, SplitShares.Of(train, validation, _predict), seed));

    /// <summary>Finishes the pipeline, so it can be run.</summary>
    /// <returns>The declaration with the means to carry it out.</returns>
    /// <exception cref="InvalidOperationException">A share was held back to predict on and nothing splits the rows.</exception>
    public Pipeline Build()
    {
        // A share held back by a pipeline that never divides anything is a promise nothing keeps: the
        // split is what carries it, so without one the rows would all come out as training.
        if (_predict > 0)
        {
            throw new InvalidOperationException(
                $"This pipeline holds {_predict} back to predict on but never splits the rows, and the "
                + "split is what sets that share aside.");
        }

        return new Pipeline(Declaration, _rows);
    }

    private FittingBuilder Split(ISplitStep step)
    {
        ThrowIfSplit();
        PipelineDeclaration.ThrowIfFaulty([.. _steps, step]);

        _steps.Add(step);
        _split = true;

        return new FittingBuilder(_steps, _rows);
    }

    private void ThrowIfSplit()
    {
        // Holding on to this builder after splitting used to let a feature be declared into the same list,
        // after the split — the leak arriving from the side — and a second split made one declaration that
        // claimed to divide the rows twice. Two pipelines are two chains, said out loud.
        if (_split)
        {
            throw new InvalidOperationException(
                "This pipeline has been split; carry on with the builder the split handed back, "
                + "or start another pipeline for a second arrangement.");
        }
    }
}

/// <summary>
/// A pipeline being written, after it has been told how to split.
/// </summary>
/// <remarks>
/// This is where everything that learns from the data lives, and each of those things is fitted on the
/// training rows alone. There is no way back to <see cref="PipelineBuilder"/>, and the builder the chain
/// started with is finished the moment it is split, so nothing can be added on the far side of the line.
/// </remarks>
public sealed class FittingBuilder
{
    private readonly List<IPipelineStep> _steps;
    private readonly IRowSource? _rows;

    internal FittingBuilder(List<IPipelineStep> steps, IRowSource? rows)
    {
        _steps = steps;
        _rows = rows;
    }

    /// <summary>What has been declared so far.</summary>
    public PipelineDeclaration Declaration => new(_steps);

    /// <summary>Adds a declared step, which may be one that is fitted on the training rows.</summary>
    /// <param name="step">The step to declare.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Add(IPipelineStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        PipelineDeclaration.ThrowIfFaulty([.. _steps, step]);

        _steps.Add(step);

        return this;
    }

    /// <summary>Fills the gaps in a column, the named way.</summary>
    /// <param name="column">The column with gaps in it.</param>
    /// <param name="strategy">What to put in them — <see cref="With"/> has the names.</param>
    /// <param name="refuseAbove">
    /// The share of the training rows that may be gaps and still be filled; above it the column is left out and
    /// the column saying where the gaps were speaks for it. Nothing, for no limit.
    /// </param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">The column has no name, or the strategy is not one of the names.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The share is not one.</exception>
    public FittingBuilder FillMissing(string column, FillStrategy strategy, double? refuseAbove = null) =>
        Add(FillMissingStep.Of(column, strategy, refuseAbove));

    /// <summary>Brings a column onto a comparable scale, by numbers learned from the training rows.</summary>
    /// <param name="column">The column to scale.</param>
    /// <param name="scale">Which kind of scaling.</param>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Normalise(
        string column, Scale scale = Scale.Standard, OutOfRange outOfRange = OutOfRange.Pass) =>
        Add(new NormaliseStep(column, scale, outOfRange));

    /// <summary>Brings several columns onto a comparable scale, by numbers learned from the training rows.</summary>
    /// <param name="columns">The columns to scale, each on its own.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Normalise(params string[] columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        foreach (var column in columns)
        {
            Add(new NormaliseStep(column));
        }

        return this;
    }

    /// <summary>Writes a column of words down as numbers, using the categories the training rows held.</summary>
    /// <param name="column">The column of words.</param>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Encode(string column, As how = As.OneHot, Unseen unseen = Unseen.Reserve) =>
        Add(new EncodeStep(column, how, unseen));

    /// <summary>Brings each row onto a comparable scale, learning nothing.</summary>
    /// <param name="norm">How the row's size is measured.</param>
    /// <param name="columns">The columns that make up the row.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder NormaliseRow(Norm norm, params string[] columns) =>
        Add(new NormaliseRowStep(columns, norm));

    /// <summary>Holds the extreme values of a column to bounds learned from the training rows.</summary>
    /// <param name="column">The column to hold.</param>
    /// <param name="bounds">How the bounds are worked out: by quantile, by spread, or by the middle half.</param>
    /// <param name="at">How far out they sit: a share for a quantile, a multiple otherwise.</param>
    /// <param name="outlier">What happens to a value outside them.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder ClipOutliers(
        string column, Bounds bounds = Bounds.Iqr, double at = 1.5, Outlier outlier = Outlier.Clip) =>
        Add(new ClipOutliersStep(column, bounds, at, outlier));

    /// <summary>Writes every column that stands for a group down as numbers.</summary>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="InvalidOperationException">Nothing before it declares a category.</exception>
    /// <remarks>
    /// Which columns are categories was said once, where the data was declared or by the step that made the
    /// column. This is one step that takes every category where it stands, so marking one more column a
    /// category does not mean remembering to add a line down here as well.
    /// </remarks>
    public FittingBuilder EncodeCategories(As how = As.OneHot, Unseen unseen = Unseen.Reserve) =>
        Add(new EncodeCategoriesStep(how, unseen));

    /// <summary>Leaves columns out from here on.</summary>
    /// <param name="columns">The columns to leave out.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>The column that says where a gap was is written always, and this is how it is left out.</remarks>
    public FittingBuilder Drop(params string[] columns) => Add(new DropColumnsStep(columns));

    /// <summary>Profiles the columns where it stands, on the training rows.</summary>
    /// <param name="columns">The columns to profile; none, for every column here.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Profile(params string[] columns) => Add(new ProfileStep(columns));

    /// <summary>Sets out the rows a correlation between columns is drawn from, on the training rows.</summary>
    /// <param name="columns">Two or more columns holding numbers.</param>
    /// <param name="shown">Drawn, or as the numbers themselves.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Correlation(IEnumerable<string> columns, Shown shown = Shown.Drawn) =>
        Add(new CorrelationStep(columns, shown));

    /// <summary>Says what happens to a value in a column that is not a number.</summary>
    /// <param name="column">The column to watch.</param>
    /// <param name="strategy">What to do; refusing is the default and usually the right answer.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder FillNaN(string column, FillStrategy strategy = default) =>
        Add(new FillNaNStep(column, strategy));

    /// <summary>Names the column a model is being asked to predict.</summary>
    /// <param name="column">The column holding the answer.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <remarks>The answer is handed over separately, never among the numbers a model is shown.</remarks>
    public FittingBuilder Target(string column) => Add(new TargetStep(column));

    /// <summary>Names the columns a model is asked to predict as one answer: how a whole is divided among them.</summary>
    /// <param name="columns">The columns, in their order: at least two.</param>
    /// <param name="scaleBy">
    /// The column saying how many the shares are shares of, so predictions come back as how many fell in each; nothing
    /// to have them come back as shares.
    /// </param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
    /// <remarks>Every row's shares sum to one: divide each row by its sum above this, with <see cref="NormaliseRow"/> and L1.</remarks>
    public FittingBuilder Distribution(IEnumerable<string> columns, string? scaleBy = null) =>
        Add(new DistributionStep(columns, scaleBy));

    /// <summary>Finishes the pipeline, so it can be run.</summary>
    /// <returns>The declaration with the means to carry it out.</returns>
    public Pipeline Build() => new(Declaration, _rows);
}

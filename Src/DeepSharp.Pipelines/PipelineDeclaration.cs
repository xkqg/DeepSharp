// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>
/// The ordered steps of a pipeline, as they were declared.
/// </summary>
/// <remarks>
/// This is the half of a saved pipeline that a person writes. The other half — what each step learned while
/// being fitted — is written by the fit and kept apart from it, which is what allows the same declaration to
/// be re-fitted on fresh data and two runs to be compared by their declarations alone.
/// <para>
/// It is also where the rules live. A declaration is the single thing the chain, the extension point, a
/// hand-written file and a notebook all become, so refusing a step that learns before the data has been
/// split — or a second source, schema, split or target, or rows dropped after the split — is done here once
/// rather than in four places that would have to agree.
/// </para>
/// </remarks>
public sealed class PipelineDeclaration : IEquatable<PipelineDeclaration>
{
    // Every rule a declaration keeps, whichever door it came through. Each is a small type of its own, so a
    // rule is added by writing one and listing it here, and the declaration itself does not change shape.
    private static readonly IDeclarationRule[] Rules =
    [
        new EveryStepActs(),
        new AtMostOne<IOpensRows>("source"),
        new TheSourceComesFirst(),
        new AtMostOne<IBindsColumns>("schema"),
        new ColumnsAreDeclaredFirst(),
        new AtMostOne<ISplitStep>("split"),
        new NothingLearnsBeforeTheSplit(),
        new RowsAreSettledBeforeTheSplit(),
        new AtMostOne<IOrdersRows>("order"),
        new RowOrderIsDeclaredBeforeItIsRead(),
        new AtMostOne<TargetStep>("target"),
        new ColumnsAreThereWhereTheyAreRead(),
    ];

    // The label the chain of keys starts from. Fixed for good, because the keys are written into files.
    private const string KeyLabel = "deepsharp.declaration";

    private readonly IPipelineStep[] _steps;

    // Worked out once, when a key is first asked for: the steps never change, so neither do their keys.
    private string[]? _keys;

    /// <summary>A declaration of exactly these steps, in this order.</summary>
    /// <param name="steps">The steps, in the order they were written.</param>
    /// <exception cref="DeclarationException">
    /// The steps break a rule every declaration keeps: a step that does nothing, two of something there is one
    /// of, a step above the columns it works on, a step that learns from the data before the split, rows dropped
    /// after it. Every fault is named at once.
    /// </exception>
    public PipelineDeclaration(IEnumerable<IPipelineStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        _steps = [.. steps];

        ThrowIfFaulty(_steps);

        ColumnsAt = Array.FindIndex(_steps, step => step is IBindsColumns);
        SplitAt = Array.FindIndex(_steps, step => step is ISplitStep);
    }

    /// <summary>Everything wrong with these steps as a declaration, without refusing them.</summary>
    /// <param name="steps">The steps, in the order they were written.</param>
    /// <returns>Every fault, each with the step it is at; none when the steps make a declaration.</returns>
    /// <remarks>
    /// For whatever writes a declaration a piece at a time and wants to show where it is wrong before it is
    /// finished — a notebook puts each fault on the block it belongs to. The constructor refuses exactly
    /// these faults, so the two can never disagree.
    /// </remarks>
    public static IReadOnlyList<DeclarationFault> FaultsIn(IEnumerable<IPipelineStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        IPipelineStep[] written = [.. steps];

        return [.. Rules.SelectMany(rule => rule.FaultsIn(written)).OrderBy(fault => fault.At)];
    }

    /// <summary>Refuses steps that break a rule, naming every fault at once.</summary>
    /// <param name="steps">The steps, in the order they were written.</param>
    /// <exception cref="DeclarationException">At least one rule is broken.</exception>
    internal static void ThrowIfFaulty(IReadOnlyList<IPipelineStep> steps)
    {
        var faults = FaultsIn(steps);

        if (faults.Count > 0)
        {
            throw new DeclarationException(faults);
        }
    }

    /// <summary>The version of the pipeline file this library writes, and the newest one it reads.</summary>
    /// <remarks>
    /// Every file names the version it was written against, and a file that names none is read as the first.
    /// The number goes up when a verb comes to mean something else — as the splits did when they began to divide
    /// rows by what they hold rather than by where they stand — and a step is refused from a file older than the
    /// meaning it has now, rather than read as something it never meant. A property rather than a constant, so a
    /// package compiled against this version reads the number the running library has.
    /// </remarks>
    public static int Version => 2;

    /// <summary>The steps, in the order they were written.</summary>
    public IReadOnlyList<IPipelineStep> Steps => _steps;

    /// <summary>The key of the steps up to one: what a fit at that step was learned behind.</summary>
    /// <param name="position">The step's place, counting from nought.</param>
    /// <returns>Sixty-four hexadecimal digits, the same for the same steps in every process on every machine.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There is no step at that place.</exception>
    /// <remarks>
    /// A chain of SHA-256 digests: each step's key is the digest of the key before it and the step as the step
    /// itself writes it, the first seeded with a fixed label. So a key covers its step and every step above it
    /// and nothing below; it does not depend on how a file happened to space, order or spell what a step was
    /// read from, since the step is written again from what it holds; and two identical steps at two places
    /// have two keys. The fitted half of a saved pipeline files every entry under it, so a fit is never served
    /// under steps that changed above it, and survives any edit below it.
    /// </remarks>
    public string KeyAt(int position)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(position, _steps.Length);

        return (_keys ??= ChainOfKeys(_steps))[position];
    }

    /// <summary>
    /// The key of the rows at one step: of where the step stands, and of every step those rows are worked out from
    /// — the steps down to it, and on down to the split when the split stands below it, since the split below
    /// places the rows above it.
    /// </summary>
    /// <param name="position">The step's place, counting from nought.</param>
    /// <returns>Sixty-four hexadecimal digits, the same for the same view in every process on every machine.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There is no step at that place.</exception>
    /// <remarks>
    /// What a front end marks a view with: while the key and the source's bytes stay, the rows at the step are the
    /// same rows, and when either changes they may not be. Every step above the split is worked out from the same
    /// steps and shows rows of its own, so the place is in the key as well. Never a cell's own identity, which a
    /// copy of the cell would share.
    /// </remarks>
    public string ViewKeyAt(int position)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(position, _steps.Length);

        var walked = KeyAt(WalkedFor(position + 1) - 1);

        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(
            string.Create(CultureInfo.InvariantCulture, $"{walked}@{position}"))));
    }

    /// <summary>How many steps the rows after the first so many are worked out through.</summary>
    /// <param name="steps">How many steps from the start.</param>
    /// <returns>Those, or the steps down to the split when it stands below them, since it places the rows above it.</returns>
    internal int WalkedFor(int steps) => Math.Max(steps, SplitAt + 1);

    private static string[] ChainOfKeys(IPipelineStep[] steps)
    {
        var keys = new string[steps.Length];
        var previous = Encoding.UTF8.GetBytes(KeyLabel);

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        for (var at = 0; at < steps.Length; at++)
        {
            hash.AppendData(previous);
            hash.AppendData(steps[at].Canonical());
            previous = hash.GetHashAndReset();
            keys[at] = Convert.ToHexStringLower(previous);
        }

        return keys;
    }

    /// <summary>Where the step that declares the columns stands, or -1 when none does.</summary>
    internal int ColumnsAt { get; }

    /// <summary>Where the split stands, or -1 when nothing divides the rows.</summary>
    internal int SplitAt { get; }

    /// <summary>The columns there are before a step: the ones a block standing there can pick from.</summary>
    /// <param name="position">The step's place, counting from nought; the number of steps for after the last.</param>
    /// <returns>The columns, followed from the schema down.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There is no such place in this declaration.</exception>
    public ColumnState ColumnsBefore(int position)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(position, _steps.Length);

        return ColumnFlow.Follow(_steps, ColumnState.None, from: 0, until: position, declared: false).State;
    }

    /// <summary>Writes the declaration as JSON: the machine's copy, which diffs and travels.</summary>
    /// <returns>The version it is written against and the steps, as one JSON document.</returns>
    public string ToJson() => PipelineDocument.Write(this, fitted: null);

    /// <summary>Reads a declaration back, using the verbs a given catalog knows.</summary>
    /// <param name="json">The file a declaration was written as. What a fit learned, when the file holds it, is left to the fit.</param>
    /// <param name="catalog">The verbs that may appear in it.</param>
    /// <returns>The declaration the file describes.</returns>
    /// <exception cref="PipelineFileException">
    /// Anything in the file is wrong, every fault named at its line and column: text that is not JSON, a key
    /// written twice, a newer version, a step no registered verb reads or that cannot be read, a step whose
    /// meaning changed after the file was written, a rule the steps break.
    /// </exception>
    /// <remarks>
    /// There is no reading a file without saying which verbs it may hold. A reader that reached for a catalog
    /// of its own could not know another package's verbs, and would call one a misspelling; a file naming a verb
    /// the catalog handed over does not know is refused, saying which package brings it when it is one of this
    /// library's own. A fitted half that no longer matches the steps is not refused here, because reading the
    /// steps alone is how a pipeline is fitted again.
    /// </remarks>
    public static PipelineDeclaration FromJson(string json, StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(catalog);

        return PipelineDocument.ReadDeclaration(json, catalog);
    }

    /// <summary>The declaration as it is spoken: <c>read.csv -&gt; split.byTime -&gt; fill.missing</c>.</summary>
    /// <returns>The verbs in order, which is what a person checks first.</returns>
    public override string ToString() => string.Join(" -> ", _steps.Select(step => step.Verb));

    /// <summary>Whether two declarations say the same thing, step for step.</summary>
    /// <param name="other">The declaration to compare with.</param>
    /// <returns><see langword="true"/> when both declare the same steps in the same order.</returns>
    public bool Equals(PipelineDeclaration? other) =>
        other is not null && _steps.SequenceEqual(other._steps);

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as PipelineDeclaration);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var step in _steps)
        {
            hash.Add(step);
        }

        return hash.ToHashCode();
    }

    /// <summary>Whether two declarations say the same thing.</summary>
    /// <param name="left">One declaration.</param>
    /// <param name="right">The other.</param>
    /// <returns><see langword="true"/> when both declare the same steps in the same order.</returns>
    /// <remarks>
    /// Every step is a record and so answers to <c>==</c>. A declaration that answered only to the method
    /// would compare by reference here and by value one line away, which is a trap nobody looks for twice.
    /// </remarks>
    public static bool operator ==(PipelineDeclaration? left, PipelineDeclaration? right) =>
        left is null ? right is null : left.Equals(right);

    /// <summary>Whether two declarations say different things.</summary>
    /// <param name="left">One declaration.</param>
    /// <param name="right">The other.</param>
    /// <returns><see langword="true"/> when they differ.</returns>
    public static bool operator !=(PipelineDeclaration? left, PipelineDeclaration? right) => !(left == right);
}

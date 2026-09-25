// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Which verbs a declaration may use, and how each one is read back.
/// </summary>
/// <remarks>
/// A catalog is handed to whatever reads a file; nothing reaches for a shared one of its own accord.
/// <see cref="BuiltIn"/> is a factory and hands back a fresh catalog every time, so a program that teaches
/// one an extra verb cannot change what another part of the same process understands.
/// </remarks>
public sealed class StepCatalog
{
    /// <summary>The key every step's verb is written under, in a file and in a notebook's block.</summary>
    public const string StepKey = "step";

    // Concurrent because one catalog is registered once and then read by everything in the process, and
    // because a host may still teach it a verb after that. It also makes the one rule atomic: TryAdd either
    // takes the slot or refuses it, so two packages registering the same verb cannot both believe they won.
    private readonly ConcurrentDictionary<string, Verb> _verbs = new(StringComparer.Ordinal);

    /// <summary>The verbs this library ships with, in a catalog of their own.</summary>
    /// <returns>A new catalog; the caller owns it.</returns>
    public static StepCatalog BuiltIn()
    {
        var catalog = new StepCatalog();

        catalog.Register<ReadCsvStep>();
        catalog.Register<ReadRowsStep>();
        catalog.Register<DeclareStep>();
        catalog.Register<SplitByTimeStep>();
        catalog.Register<SplitAtRandomStep>();
        catalog.Register<SplitStratifiedStep>();
        catalog.Register<FillMissingStep>();
        catalog.Register<AddFeatureStep>();
        catalog.Register<CyclicalStep>();
        catalog.Register<NormaliseStep>();
        catalog.Register<NormaliseRowStep>();
        catalog.Register<EncodeStep>();
        catalog.Register<TargetStep>();
        catalog.Register<DistributionStep>();
        catalog.Register<LabelsStep>();
        catalog.Register<AheadStep>();
        catalog.Register<FillNaNStep>();
        catalog.Register<DropWarmUpStep>();
        catalog.Register<TimePartsStep>();
        catalog.Register<MathsStep>();
        catalog.Register<ClipOutliersStep>();
        catalog.Register<DropColumnsStep>();
        catalog.Register<EncodeCategoriesStep>();
        catalog.Register<OrderByStep>();
        catalog.Register<DropGapsStep>();
        catalog.Register<ProfileStep>();
        catalog.Register<CorrelationStep>();

        return catalog;
    }

    /// <summary>Teaches this catalog a step, under the name the step itself carries.</summary>
    /// <typeparam name="TStep">The step to register.</typeparam>
    /// <exception cref="InvalidOperationException">
    /// The verb is already known, or the step type does not say what it is called or what it does. One verb,
    /// one reader: a second registration that silently won would make the same file mean different things
    /// depending on which packages happened to be present.
    /// </exception>
    /// <remarks>
    /// The only way into a catalog. The step's parameters come with it, so a verb registered here is one a
    /// file is checked against key by key, the schema and the reference page describe, and a notebook can
    /// offer as a form. A door that took a verb and a reader alone let a verb in that nothing could describe.
    /// </remarks>
    public void Register<TStep>()
        where TStep : IPipelineStep<TStep>
    {
        if (string.IsNullOrWhiteSpace(TStep.Name))
        {
            throw new InvalidOperationException(
                $"{typeof(TStep).Name} does not say which verb it is written under, so no file could ever name it.");
        }

        if (string.IsNullOrWhiteSpace(TStep.Purpose))
        {
            throw new InvalidOperationException(
                $"'{TStep.Name}' does not say what it does, so nothing could describe it to whoever writes it.");
        }

        Add(TStep.Name, new Verb(
            element => TStep.ReadFrom(element),
            new StepDescription(TStep.Name, TStep.Purpose, TStep.Since, TStep.Parameters.All)));
    }

    /// <summary>Whether this catalog can read a step written under that name.</summary>
    /// <param name="verb">The name to look for.</param>
    /// <returns><see langword="true"/> when the verb is registered.</returns>
    public bool Knows(string verb) => _verbs.ContainsKey(verb);

    /// <summary>The verbs DeepSharp's other packages bring, and the package each comes from.</summary>
    /// <remarks>
    /// Kept here so that a file naming one of them in a catalog that was never taught it is told which
    /// package it needs, rather than that it misspelled something it did not. A test holds the list to what
    /// the packages actually register, both ways round.
    /// </remarks>
    public static IReadOnlyDictionary<string, string> VerbsOtherPackagesBring { get; } =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["feature.indicator"] = "DeepSharp.Pipelines.Indicators",
        };

    /// <summary>The package that brings a verb, when it is one of DeepSharp's own packages.</summary>
    /// <param name="verb">The verb.</param>
    /// <returns>The package's name, or nothing when no package of this library brings the verb.</returns>
    public static string? PackageThatBrings(string verb) => VerbsOtherPackagesBring.GetValueOrDefault(verb);

    /// <summary>What every verb in this catalog is, in the order of their names.</summary>
    public IReadOnlyList<StepDescription> Descriptions =>
        [.. _verbs.Values.Select(verb => verb.Description).OrderBy(each => each.Verb, StringComparer.Ordinal)];

    /// <summary>What one verb is: what it does, its parameters, and the step a new block starts with.</summary>
    /// <param name="verb">The verb.</param>
    /// <returns>Its description.</returns>
    /// <exception cref="ArgumentException">The verb is empty.</exception>
    /// <exception cref="NotSupportedException">Nothing in this catalog describes that verb.</exception>
    public StepDescription Describe(string verb)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(verb);

        return _verbs.TryGetValue(verb, out var known)
            ? known.Description
            : throw new NotSupportedException($"Nothing here describes the step '{verb}'.");
    }

    /// <summary>The JSON Schema a pipeline file written with this catalog's verbs keeps.</summary>
    /// <returns>The schema, as an indented JSON document.</returns>
    /// <remarks>
    /// For an editor, which then offers the verbs and their keys as a file is typed and marks what the reader
    /// would refuse. It is written from the same descriptions the reader checks a file against, so the two
    /// cannot disagree about a key, a kind or a word. What no schema can say — shares that make a whole, a
    /// quantile bound further out than half, where a step may stand — is the reader's alone.
    /// </remarks>
    public string JsonSchema() => PipelineFileSchema.Write(Descriptions);

    /// <summary>Every verb in this catalog, written out as a page a person reads.</summary>
    /// <returns>The reference, in Markdown.</returns>
    /// <remarks>
    /// Generated rather than typed, so the page says exactly what the reader reads: a step that changes
    /// changes its page with it, and a test holds the committed copy to what the steps say.
    /// </remarks>
    public string VerbReference() => Pipelines.VerbReference.Write(Descriptions);

    private void Add(string verb, Verb entry)
    {
        if (!_verbs.TryAdd(verb, entry))
        {
            throw new InvalidOperationException(
                $"The verb '{verb}' is already registered, and a verb has exactly one reader.");
        }
    }

    /// <summary>Reads one step out of the JSON object it was written as.</summary>
    /// <param name="element">The object, including its <c>step</c> key.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">
    /// The object does not say which step it is, holds a key the step does not take, holds a value the step
    /// cannot take, or holds one it would not write back as it was written.
    /// </exception>
    /// <exception cref="NotSupportedException">
    /// Nothing has registered that verb. The file is refused rather than read with the step left out: a
    /// pipeline that quietly skipped a step would run, report nothing wrong, and not be the pipeline in the
    /// file.
    /// </exception>
    public IPipelineStep Read(JsonElement element) => Read(element, PipelineDeclaration.Version);

    /// <summary>Reads one step out of the JSON object it was written as, in a file of a given version.</summary>
    /// <param name="element">The object, including its <c>step</c> key.</param>
    /// <param name="writtenAgainst">The version of the pipeline file the step was written against.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There is no such version: below the first, or above this library's.</exception>
    /// <exception cref="FormatException">
    /// The step means something else now than it did in the version it was written against, or the object is not
    /// a step it can read (see <see cref="Read(JsonElement)"/>).
    /// </exception>
    /// <exception cref="NotSupportedException">Nothing has registered that verb.</exception>
    /// <remarks>
    /// The one place a step is read, whether from a file or from a notebook cell. A verb whose meaning changed —
    /// a split that divided rows by their place in a file and now divides them by what they hold — is refused
    /// from anything written before the change, by name: read the new way, the same words would train on other
    /// rows and say nothing about it, and there is no reading them the old way.
    /// </remarks>
    public IPipelineStep Read(JsonElement element, int writtenAgainst)
    {
        ThrowIfNoSuchVersion(writtenAgainst);

        // Asked for a property, a JSON value that is not an object throws about element types and names
        // neither the file nor the step. A declaration is edited by people, so it says what is wrong here.
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new FormatException("Every step in a declaration is an object with a 'step' in it.");
        }

        var verb = element.RequiredString(StepKey);

        if (!_verbs.TryGetValue(verb, out var known))
        {
            throw new NotSupportedException(Unknown(verb));
        }

        var description = known.Description;

        // Before its keys: a step that came to mean something else may have come to take other keys too, and a
        // key it no longer takes is not what is wrong with it.
        if (description.Since > writtenAgainst)
        {
            throw new FormatException(string.Create(
                CultureInfo.InvariantCulture,
                $"'{verb}' is read as it means now from version {description.Since} of the pipeline file, and this was written against version {writtenAgainst}, when it meant something else. It is not read the old way: write the step again, and fit the pipeline again."));
        }

        // A key nobody defined used to be read past and then vanish when the step was written back, so the
        // file said something the pipeline never did. It is refused, and the refusal lists what the step takes.
        foreach (var property in element.EnumerateObject())
        {
            if (property.Name != StepKey && !description.Keys.Contains(property.Name))
            {
                throw new FormatException(
                    $"The step '{verb}' has no parameter called '{property.Name}'. It takes: "
                    + $"{string.Join(", ", description.Parameters.SelectMany(parameter => parameter.Keys))}.");
            }
        }

        IPipelineStep step;

        try
        {
            step = known.Read(element);
        }
        catch (Exception fault)
        {
            // A step refuses its own arguments in the language of a C# parameter, and a kind a value it cannot
            // read in the language of a key; both are right where they are raised and say nothing of which step
            // it was. Reading a file is one kind of fault, named after the step it happened in.
            throw new FormatException($"The step '{verb}' cannot be read: {InTheFilesWords(fault)}", fault);
        }

        if (step.Verb != verb)
        {
            throw new FormatException(
                $"The step written as '{verb}' was read as a '{step.Verb}', so it would be written back "
                + "under a different name than the one it was loaded from.");
        }

        ThrowIfNotKeptAsWritten(verb, description, element, step);

        return step;
    }

    /// <summary>Reads one step written on its own, as text: a notebook block, say.</summary>
    /// <param name="json">The step, as the JSON object it was written as.</param>
    /// <returns>The step.</returns>
    /// <exception cref="PipelineFileException">
    /// The text is not one step this catalog reads: text that is not JSON, a key written twice, something other
    /// than one object, or a step <see cref="Read(JsonElement)"/> refuses. Every fault is at its line and column,
    /// placed by the same rule as a pipeline file's.
    /// </exception>
    public IPipelineStep ReadStep(string json) => ReadStep(json, PipelineDeclaration.Version);

    /// <summary>Reads one step written on its own, as text, against a given version of the pipeline file.</summary>
    /// <param name="json">The step, as the JSON object it was written as.</param>
    /// <param name="writtenAgainst">The version it was written against.</param>
    /// <returns>The step.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There is no such version.</exception>
    /// <exception cref="PipelineFileException">The text is not one step this catalog reads.</exception>
    public IPipelineStep ReadStep(string json, int writtenAgainst)
    {
        ArgumentNullException.ThrowIfNull(json);
        ThrowIfNoSuchVersion(writtenAgainst);

        return PipelineDocument.ReadStep(json, this, writtenAgainst);
    }

    // The versions of the pipeline file there are: from the first to the one this library writes.
    private static void ThrowIfNoSuchVersion(int version)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(version, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(version, PipelineDeclaration.Version);
    }

    /// <summary>Refuses a step that would write a value back other than the one the file holds.</summary>
    /// <remarks>
    /// A step may settle a value it was handed — a name of nothing but spaces left to the step, say — and a
    /// file holding such a value would load, run as something else, and write itself back saying so. Each
    /// value the file wrote is read back from the step written out again, through its own kind.
    /// </remarks>
    private static void ThrowIfNotKeptAsWritten(string verb, StepDescription description, JsonElement written, IPipelineStep step)
    {
        using var rewritten = JsonDocument.Parse(step.Canonical());

        foreach (var parameter in description.Parameters)
        {
            if (parameter.IsWrittenIn(written) && !parameter.KeptAsWritten(written, rewritten.RootElement))
            {
                throw new FormatException(
                    $"The step '{verb}' does not keep '{parameter.Key}' as the file writes it "
                    + $"({parameter.AsWrittenIn(written)}): it would be written back as "
                    + $"{parameter.AsWrittenIn(rewritten.RootElement)}, so the file would say one thing and the "
                    + "pipeline do another.");
            }
        }
    }

    /// <summary>Why a verb cannot be read here: it is another package's, or nothing anywhere knows it.</summary>
    private string Unknown(string verb)
    {
        if (PackageThatBrings(verb) is { } package)
        {
            return $"'{verb}' is a step from {package}, which is not registered here. Reference the package and "
                   + "register its steps with the catalog that reads this file.";
        }

        var nearest = _verbs.Keys.MinBy(known => Distance(verb, known));

        return nearest is null
            ? $"'{verb}' is not a step anything here knows."
            : $"'{verb}' is not a step anything here knows. The nearest one it knows is '{nearest}'.";
    }

    /// <summary>How many single-character edits turn one word into the other.</summary>
    private static int Distance(string one, string other)
    {
        var previous = Enumerable.Range(0, other.Length + 1).ToArray();

        for (var at = 1; at <= one.Length; at++)
        {
            var current = new int[other.Length + 1];
            current[0] = at;

            for (var to = 1; to <= other.Length; to++)
            {
                var change = one[at - 1] == other[to - 1] ? 0 : 1;
                current[to] = Math.Min(Math.Min(current[to - 1] + 1, previous[to] + 1), previous[to - 1] + change);
            }

            previous = current;
        }

        return previous[other.Length];
    }

    // A refusal as a file reads it. The runtime names the C# parameter after the message of an argument refused,
    // and a file has keys rather than parameters — "Train" is not the "train" it holds — so the name is left off,
    // with the value it may add after it, which the runtime writes in the culture of the machine. The words it
    // names a parameter in are asked of the runtime itself, so they are whatever it says.
    private static string InTheFilesWords(Exception fault) =>
        fault is ArgumentException { ParamName: { } parameter }
            ? fault.Message.Split(new ArgumentException(string.Empty, parameter).Message, 2)[0]
            : fault.Message;

    /// <summary>How one verb is read, and what it is.</summary>
    private readonly record struct Verb(Func<JsonElement, IPipelineStep> Read, StepDescription Description);
}

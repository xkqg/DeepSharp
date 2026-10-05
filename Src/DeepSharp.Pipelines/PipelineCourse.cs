// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace DeepSharp.Pipelines;

/// <summary>
/// One step of a course: the verb it is written under, and what has been said of it so far.
/// </summary>
/// <remarks>
/// A step cannot be half made — a reader refuses a step that lacks a key it needs, naming the key — so a course never
/// holds steps. It holds what a person has decided of each verb, and makes the step only once enough is decided: from what
/// was said and the verb's skeleton, read as a file's step is read. What was said is kept as text, with its keys in one
/// order and nothing said as nothing, so two steps that say the same thing in the same words are the same step, and what a
/// caller does to the object it handed in afterwards reaches no step.
/// </remarks>
public sealed record CourseStep
{
    private static readonly JsonSerializerOptions Compact = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private CourseStep(string verb, string said)
    {
        Verb = verb;
        Said = said;
    }

    /// <summary>The verb the step is written under.</summary>
    public string Verb { get; }

    /// <summary>
    /// What has been said of the step: one JSON object on one line, its keys in alphabetical order, holding only the keys
    /// that were decided.
    /// </summary>
    public string Said { get; }

    /// <summary>A step of a course.</summary>
    /// <param name="verb">The verb the step is written under.</param>
    /// <param name="said">
    /// What has been said of it, under the keys the verb takes; nothing when nothing has. A key said as nothing is not said.
    /// </param>
    /// <returns>The step, holding a copy of what was said.</returns>
    /// <exception cref="ArgumentException">The verb is empty, or what was said holds the verb itself as a key.</exception>
    /// <exception cref="ArgumentNullException">There is no verb.</exception>
    public static CourseStep Of(string verb, JsonObject? said = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(verb);

        if (said?.ContainsKey(StepCatalog.StepKey) is true)
        {
            throw new ArgumentException(
                $"A step is written under its verb, so '{StepCatalog.StepKey}' is not a key to say of it.", nameof(said));
        }

        var kept = new JsonObject();

        foreach (var (key, value) in (said ?? []).Where(each => each.Value is not null).OrderBy(each => each.Key, StringComparer.Ordinal))
        {
            kept[key] = value!.DeepClone();
        }

        return new CourseStep(verb, kept.ToJsonString(Compact));
    }

    /// <summary>The keys this step still waits for: those the verb leaves to the person that nobody has said yet.</summary>
    /// <param name="catalog">The verbs that may be named.</param>
    /// <returns>The keys, in the order the verb writes them; none once everything that names something of the person's is said.</returns>
    /// <exception cref="NotSupportedException">The catalog describes no such verb.</exception>
    public IReadOnlyList<string> Waiting(StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        var said = SaidObject();

        return [.. catalog.Describe(Verb).Waiting.Where(key => !said.ContainsKey(key))];
    }

    /// <summary>The step as a block starts with it: the verb's skeleton, with what has been said put in its place.</summary>
    /// <param name="catalog">The verbs that may be named.</param>
    /// <returns>
    /// The step as one JSON object on one line, every key the verb needs present, those still waiting as nothing. It is read
    /// as a step once nothing waits.
    /// </returns>
    /// <exception cref="NotSupportedException">The catalog describes no such verb.</exception>
    public string Skeleton(StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        var description = catalog.Describe(Verb);
        var skeleton = JsonNode.Parse(description.Skeleton)!.AsObject();
        var template = JsonNode.Parse(description.Template)!.AsObject();
        var said = SaidObject();
        var written = new JsonObject();

        // In the order the verb writes its keys, whatever was said of one the skeleton leaves out; a key the verb does not
        // take goes last, where whoever reads the step finds it refused.
        foreach (var (key, _) in template)
        {
            if (said.TryGetPropertyValue(key, out var value) || skeleton.TryGetPropertyValue(key, out value))
            {
                written[key] = value?.DeepClone();
            }
        }

        foreach (var (key, value) in said.Where(pair => !written.ContainsKey(pair.Key)))
        {
            written[key] = value?.DeepClone();
        }

        return written.ToJsonString(Compact);
    }

    /// <summary>What has been said, as an object of its own that the caller may change.</summary>
    /// <returns>A new object holding every key that was decided.</returns>
    internal JsonObject SaidObject() => JsonNode.Parse(Said)!.AsObject();

    /// <summary>The step as far as its verb can judge it now: the step it makes, or what it refuses.</summary>
    /// <param name="catalog">The verbs that may be named.</param>
    /// <param name="version">The version of the pipeline file the step is said to be written against.</param>
    /// <returns>
    /// What the catalog says of a verb it does not know; else, for a key the verb does not take or a file older than the verb,
    /// the verb's own refusal; else nothing while the step still waits, since a value is judged beside the keys it is tied to
    /// and some of those are yet to be said; and once nothing waits, the step the verb makes of what was said, or what it
    /// refuses of it.
    /// </returns>
    /// <remarks>The one rule a course is judged by, at the file, at the declaration and in a notebook's block alike.</remarks>
    internal CourseStepJudgement Judged(StepCatalog catalog, int version)
    {
        if (!catalog.Knows(Verb))
        {
            return new CourseStepJudgement(null, [catalog.Unknown(Verb)]);
        }

        var description = catalog.Describe(Verb);
        var said = SaidObject();
        var untaken = new JsonObject();

        foreach (var (key, value) in said.Where(pair => !description.Keys.Contains(pair.Key)))
        {
            untaken[key] = value?.DeepClone();
        }

        // What a verb refuses before it looks at a value: a key it does not take, and a file written before it meant this.
        if (untaken.Count > 0 || description.Since > version)
        {
            return Made(catalog, untaken, version);
        }

        return Waiting(catalog).Count > 0 ? new CourseStepJudgement(null, []) : Made(catalog, said, version);
    }

    private CourseStepJudgement Made(StepCatalog catalog, JsonObject said, int version)
    {
        try
        {
            return new CourseStepJudgement(catalog.MakeFromSaid(Verb, said, version), []);
        }
        catch (PipelineFileException refused)
        {
            return new CourseStepJudgement(null, [.. refused.Faults.Select(fault => fault.Message)]);
        }
    }
}

/// <summary>A step of a course as far as its verb can judge it.</summary>
/// <param name="Made">The step the verb makes of what was said, once nothing waits and the verb accepts it.</param>
/// <param name="Refusals">What the catalog or the verb refuses, each in its own words; none when the step is made or still waits.</param>
internal readonly record struct CourseStepJudgement(IPipelineStep? Made, IReadOnlyList<string> Refusals);

/// <summary>
/// The steps of a prepared pipeline in the order they belong, each as much said as a person has said of it: a pipeline
/// written from nothing, for a person to fill in rather than remember.
/// </summary>
/// <remarks>
/// A person who starts a pipeline from nothing has to know which verb comes where. A course says it: every step of a
/// prepared pipeline, in order, each waiting for what only its person knows — the file, the columns, the bounds — while
/// what merely settles how starts as its verb starts it. It is a file of its own, as a preset is, read through the same
/// door with the same care, and it is not a pipeline: it holds what has been said of each verb, never a step, because a
/// step cannot be half made.
/// </remarks>
public sealed record PipelineCourse
{
    private PipelineCourse(CourseStep[] steps) => Steps = Array.AsReadOnly(steps);

    /// <summary>
    /// The course for the rows of a table, rows that do not depend on one another: a passenger list, a customer file.
    /// </summary>
    /// <remarks>
    /// It reads the file, declares the columns, settles the gaps and works out the features — everything that is the same for
    /// every row, whoever reads it — and scales what has bounds you know. Only then are the rows divided, at random and
    /// keeping the mixture of the answer; and below the line, the answer is named, what is not needed is dropped, and what
    /// the training rows decide is filled and scaled, before the report and the network. The answer stands right below the
    /// split because a return must be made from its column as it was read, so every answer can stand there and not every
    /// answer can stand lower.
    /// </remarks>
    public static PipelineCourse Table { get; } = Of(
    [
        CourseStep.Of("read.csv"),
        CourseStep.Of("declare"),
        CourseStep.Of("settle.gaps"),
        CourseStep.Of("feature.add"),
        CourseStep.Of("scale.given"),
        CourseStep.Of("split.stratified"),
        CourseStep.Of("target"),
        CourseStep.Of("drop.columns"),
        CourseStep.Of("fill.missing"),
        CourseStep.Of("normalise"),
        CourseStep.Of("evidence.report"),
        CourseStep.Of("learn.network"),
    ]);

    /// <summary>
    /// The course for rows that follow one another in time: prices, readings, anything where yesterday is not independent
    /// of today.
    /// </summary>
    /// <remarks>
    /// The table's course with three differences the rules make: the rows are put in order first, since a series is read in
    /// its order and shuffling it would be a leak; they are divided along the clock rather than at random, with a gap of one
    /// moment kept apart, since the answer is read from the row after; and the answer is the one that reads ahead. The gap
    /// is said here, because an answer that reads ahead refuses a split without one; make it as wide as the answer reads.
    /// </remarks>
    public static PipelineCourse SeriesInTime { get; } = Of(
    [
        CourseStep.Of("read.csv"),
        CourseStep.Of("declare"),
        CourseStep.Of("order.by"),
        CourseStep.Of("settle.gaps"),
        CourseStep.Of("feature.add"),
        CourseStep.Of("scale.given"),
        CourseStep.Of("split.byTime", new JsonObject { ["gap"] = 1 }),
        CourseStep.Of("target.ahead"),
        CourseStep.Of("drop.columns"),
        CourseStep.Of("fill.missing"),
        CourseStep.Of("normalise"),
        CourseStep.Of("evidence.report"),
        CourseStep.Of("learn.network"),
    ]);

    /// <summary>Every course there is: the one a notebook offers a button for each of.</summary>
    public static IReadOnlyList<PipelineCourse> Named { get; } = [Table, SeriesInTime];

    /// <summary>The steps, in the order they belong.</summary>
    public IReadOnlyList<CourseStep> Steps { get; }

    /// <summary>A course of these steps.</summary>
    /// <param name="steps">The steps, in the order they belong.</param>
    /// <returns>The course, holding a copy of the list.</returns>
    /// <exception cref="ArgumentNullException">There is no list.</exception>
    /// <exception cref="ArgumentException">The list holds no step, or one of them is nothing.</exception>
    public static PipelineCourse Of(IEnumerable<CourseStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        CourseStep[] kept = [.. steps];

        if (kept.Length == 0)
        {
            throw new ArgumentException("A course holds at least one step.", nameof(steps));
        }

        if (kept.Any(step => step is null))
        {
            throw new ArgumentException("A course holds steps, and one of these is nothing.", nameof(steps));
        }

        return new PipelineCourse(kept);
    }

    /// <summary>Says more of a verb: what is said now replaces what was said of that key, and the rest stays as it was.</summary>
    /// <param name="verb">The verb the step is written under; the course holds it once.</param>
    /// <param name="said">What is said, under the keys the verb takes. A key said as nothing takes back what was said of it.</param>
    /// <returns>The course with the step said more of; this course is left as it was.</returns>
    /// <exception cref="ArgumentNullException">Nothing is said.</exception>
    /// <exception cref="ArgumentException">
    /// The verb is empty, the course holds no step of that verb, or holds more than one and so cannot tell which is meant.
    /// </exception>
    public PipelineCourse Say(string verb, JsonObject said)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(verb);
        ArgumentNullException.ThrowIfNull(said);

        var places = PlacesOf(verb);

        if (places.Length > 1)
        {
            throw new ArgumentException(
                $"This course holds '{verb}' more than once, at steps {string.Join(" and ", places.Select(place => place + 1))}, so it cannot tell which is meant.",
                nameof(verb));
        }

        var merged = Steps[places[0]].SaidObject();

        foreach (var (key, value) in said)
        {
            if (value is null)
            {
                merged.Remove(key);
            }
            else
            {
                merged[key] = value.DeepClone();
            }
        }

        CourseStep[] steps = [.. Steps];
        steps[places[0]] = CourseStep.Of(verb, merged);

        return new PipelineCourse(steps);
    }

    /// <summary>Adds one more step of a verb the course holds, right after the last one of it.</summary>
    /// <param name="verb">The verb the step is written under; the course holds it at least once.</param>
    /// <param name="said">What is said of the new step, under the keys the verb takes; nothing is said of it until it is.</param>
    /// <returns>The course with one more step of the verb; this course is left as it was.</returns>
    /// <exception cref="ArgumentNullException">Nothing is said.</exception>
    /// <exception cref="ArgumentException">The verb is empty, or the course holds no step of it.</exception>
    /// <remarks>
    /// A settling, a scale and a fill are written once a column, so a course says them one at a time: <see cref="Say"/> the
    /// step the course holds, then add the next, and the next.
    /// </remarks>
    public PipelineCourse Also(string verb, JsonObject said)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(verb);
        ArgumentNullException.ThrowIfNull(said);

        var after = PlacesOf(verb)[^1] + 1;

        return new PipelineCourse([.. Steps.Take(after), CourseStep.Of(verb, said), .. Steps.Skip(after)]);
    }

    /// <summary>Leaves out every step of some verbs: what a pipeline does not need.</summary>
    /// <param name="verbs">The verbs whose steps go; the course holds each at least once.</param>
    /// <returns>The course without them; this course is left as it was.</returns>
    /// <exception cref="ArgumentNullException">No verbs are given.</exception>
    /// <exception cref="ArgumentException">A verb is empty, the course holds no step of it, or nothing would be left.</exception>
    public PipelineCourse Without(params string[] verbs)
    {
        ArgumentNullException.ThrowIfNull(verbs);

        foreach (var verb in verbs)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(verb);
            _ = PlacesOf(verb);
        }

        CourseStep[] kept = [.. Steps.Where(step => !verbs.Contains(step.Verb))];

        return kept.Length > 0
            ? new PipelineCourse(kept)
            : throw new ArgumentException($"Without {string.Join(", ", verbs.Select(verb => $"'{verb}'"))} this course would hold no step.", nameof(verbs));
    }

    // The places of the steps written under a verb; the course's own words when it holds none.
    private int[] PlacesOf(string verb)
    {
        int[] places = [.. Enumerable.Range(0, Steps.Count).Where(place => Steps[place].Verb == verb)];

        return places.Length > 0
            ? places
            : throw new ArgumentException(
                $"This course holds no step '{verb}'. It holds: {string.Join(", ", Steps.Select(step => step.Verb).Distinct(StringComparer.Ordinal))}.",
                nameof(verb));
    }

    /// <summary>What this course still waits for: every step that names something of the person's that nobody has said.</summary>
    /// <param name="catalog">The verbs the course may name.</param>
    /// <returns>
    /// A fault for each step that waits, at its place, with the keys it waits for; and for each step whose verb the catalog
    /// does not know, the catalog's own words about it. None for a course filled in all the way.
    /// </returns>
    public IReadOnlyList<DeclarationFault> Waiting(StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        List<DeclarationFault> waiting = [];

        for (var at = 0; at < Steps.Count; at++)
        {
            var step = Steps[at];

            if (!catalog.Knows(step.Verb))
            {
                waiting.Add(new DeclarationFault(at, step.Verb, catalog.Unknown(step.Verb)));

                continue;
            }

            var keys = step.Waiting(catalog);

            if (keys.Count > 0)
            {
                waiting.Add(new DeclarationFault(at, step.Verb, $"waits for what only you can say: {string.Join(", ", keys.Select(key => $"'{key}'"))}."));
            }
        }

        return waiting;
    }

    /// <summary>The pipeline this course names, once it is filled in all the way.</summary>
    /// <param name="catalog">The verbs the course may name.</param>
    /// <returns>The steps each verb makes of what was said and what its own template supplies, in the order of the course.</returns>
    /// <exception cref="DeclarationException">
    /// A step still waits for something of the person's, a verb is not one the catalog knows, a key is not its verb's, a value
    /// said is one its verb refuses, or the steps break a rule every declaration keeps: every fault, each at its step.
    /// </exception>
    /// <remarks>
    /// Each step is made from what was said and the verb's skeleton, read as a file's step is read, so a verb's own rules hold
    /// here as they do in a file. What a person has to say is never taken from the verb's template, which would put an
    /// example where a decision belongs, and a name the verb may leave out stays out.
    /// </remarks>
    public PipelineDeclaration ToDeclaration(StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        List<DeclarationFault> faults = [.. Waiting(catalog)];
        List<IPipelineStep> steps = [];

        for (var at = 0; at < Steps.Count; at++)
        {
            // A verb the catalog does not know has been said so above, in the catalog's words.
            if (!catalog.Knows(Steps[at].Verb))
            {
                continue;
            }

            var judged = Steps[at].Judged(catalog, PipelineDeclaration.Version);

            faults.AddRange(judged.Refusals.Select(message => new DeclarationFault(at, Steps[at].Verb, message)));

            if (judged.Made is { } made)
            {
                steps.Add(made);
            }
        }

        return faults.Count > 0
            ? throw new DeclarationException([.. faults.OrderBy(fault => fault.At)])
            : new PipelineDeclaration(steps);
    }

    /// <summary>Writes the course as JSON: the version, then each step as its verb and what has been said of it.</summary>
    /// <returns>The file, one key to a line, with a line feed between lines on every system.</returns>
    /// <remarks>
    /// A file of its own, as a preset is — not a pipeline's file — so what it says of a verb is read as a person writes it by
    /// hand: a key left out is a key nobody has said, and a key written as nothing is the same.
    /// </remarks>
    public string ToJson() => PipelineDocument.Write(this);

    /// <summary>Reads a course back, using the verbs a given catalog knows.</summary>
    /// <param name="json">The file a course was written as.</param>
    /// <param name="catalog">The verbs its steps may name.</param>
    /// <returns>The course the file describes.</returns>
    /// <exception cref="ArgumentNullException">There is no file or no catalog.</exception>
    /// <exception cref="PipelineFileException">
    /// Anything in the file is wrong, every fault named at its line and column: text that is not JSON, a key written twice, no
    /// version or a newer one, a key a course file does not have, no list of steps, or a step that names no verb the catalog
    /// knows, a key its verb does not take, or a value its verb refuses.
    /// </exception>
    public static PipelineCourse FromJson(string json, StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(catalog);

        return PipelineDocument.ReadCourse(json, catalog);
    }

    /// <summary>Whether two courses say the same steps in the same order.</summary>
    /// <param name="other">The course to compare with.</param>
    /// <returns><see langword="true"/> when both hold the same steps, each saying the same, in the same order.</returns>
    public bool Equals(PipelineCourse? other) => other is not null && Steps.SequenceEqual(other.Steps);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var step in Steps)
        {
            hash.Add(step);
        }

        return hash.ToHashCode();
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
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
    // Concurrent because one catalog is registered once and then read by everything in the process, and
    // because a host may still teach it a verb after that. It also makes the one rule atomic: TryAdd either
    // takes the slot or refuses it, so two packages registering the same verb cannot both believe they won.
    private readonly ConcurrentDictionary<string, Func<JsonElement, IPipelineStep>> _readers = new(StringComparer.Ordinal);

    /// <summary>The verbs this library ships with, in a catalog of their own.</summary>
    /// <returns>A new catalog; the caller owns it.</returns>
    public static StepCatalog BuiltIn()
    {
        var catalog = new StepCatalog();

        catalog.Register<ReadCsvStep>();
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
        catalog.Register<FillNaNStep>();

        return catalog;
    }

    /// <summary>Teaches this catalog a step, under the name the step itself carries.</summary>
    /// <typeparam name="TStep">The step to register.</typeparam>
    /// <exception cref="InvalidOperationException">The verb is already known.</exception>
    public void Register<TStep>()
        where TStep : IPipelineStep<TStep> =>
        Register(TStep.Name, element => TStep.ReadFrom(element));

    /// <summary>Teaches this catalog a verb.</summary>
    /// <param name="verb">The name the step is written under.</param>
    /// <param name="read">How to read the step back out of its JSON object.</param>
    /// <exception cref="InvalidOperationException">
    /// The verb is already known. One verb, one reader: a second registration that silently won would make
    /// the same file mean different things depending on which packages happened to be present.
    /// </exception>
    public void Register(string verb, Func<JsonElement, IPipelineStep> read)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(verb);
        ArgumentNullException.ThrowIfNull(read);

        if (!_readers.TryAdd(verb, read))
        {
            throw new InvalidOperationException(
                $"The verb '{verb}' is already registered, and a verb has exactly one reader.");
        }
    }

    /// <summary>Whether this catalog can read a step written under that name.</summary>
    /// <param name="verb">The name to look for.</param>
    /// <returns><see langword="true"/> when the verb is registered.</returns>
    public bool Knows(string verb) => _readers.ContainsKey(verb);

    /// <summary>Reads one step out of the JSON object it was written as.</summary>
    /// <param name="element">The object, including its <c>step</c> key.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">The object does not say which step it is.</exception>
    /// <exception cref="NotSupportedException">
    /// Nothing has registered that verb. The file is refused rather than read with the step left out: a
    /// pipeline that quietly skipped a step would run, report nothing wrong, and not be the pipeline in the
    /// file.
    /// </exception>
    public IPipelineStep Read(JsonElement element)
    {
        // Asked for a property, a JSON value that is not an object throws about element types and names
        // neither the file nor the step. A declaration is edited by people, so it says what is wrong here.
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new FormatException("Every step in a declaration is an object with a 'step' in it.");
        }

        var verb = element.RequiredString("step");

        if (!_readers.TryGetValue(verb, out var read))
        {
            throw new NotSupportedException(
                $"Nothing here knows the step '{verb}'. Either it is misspelled, or the package that brings it is not referenced.");
        }

        IPipelineStep step;

        try
        {
            step = read(element);
        }
        catch (Exception fault) when (fault is not FormatException)
        {
            // A step refuses its own arguments in the language of a C# parameter, which is right at a call
            // site and useless at a file-loading boundary. Reading a file is one kind of fault, named after
            // the step it happened in, and this is also where a position in the file will later attach.
            throw new FormatException($"The step '{verb}' cannot be read: {fault.Message}", fault);
        }

        if (step.Verb != verb)
        {
            throw new FormatException(
                $"The step written as '{verb}' was read as a '{step.Verb}', so it would be written back "
                + "under a different name than the one it was loaded from.");
        }

        return step;
    }
}

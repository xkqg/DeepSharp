// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>One kind of word a declared network is written in, registered together with what a part of it becomes.</summary>
/// <typeparam name="TLower">What makes the thing a part of this kind names: its layer, its optimizer, its loss, its schedule or its stopping.</typeparam>
/// <param name="Name">The name, as a declaration and a network's file write it under 'kind'.</param>
/// <param name="Purpose">What a part of this kind is, for whoever writes one.</param>
/// <param name="Settings">The settings the kind takes, in the order they are written.</param>
/// <param name="Lower">What makes the thing a part of this kind names.</param>
/// <remarks>
/// The name, what it takes and what it becomes are written in one place, so no kind is listed without what it becomes and
/// nothing is lowered under a name no list holds.
/// </remarks>
internal readonly record struct KindWord<TLower>(string Name, string Purpose, IReadOnlyList<StepParameter> Settings, TLower Lower)
    where TLower : Delegate;

/// <summary>
/// One family of the words a declared network is written in — its layers, its optimizers, its losses, its schedules or its
/// stopping — each kind with what it becomes.
/// </summary>
/// <typeparam name="TLower">What makes the thing a part of this family names.</typeparam>
/// <remarks>
/// The list a step publishes, which its schema, the reference page and a notebook's form read, is made from the same
/// registrations a declared part is lowered by, so the two cannot drift: a kind is added by writing it once.
/// </remarks>
internal sealed class KindVocabulary<TLower>
    where TLower : Delegate
{
    private readonly Dictionary<string, TLower> _lowered = new(StringComparer.Ordinal);
    private readonly Dictionary<string, PartKind> _kinds = new(StringComparer.Ordinal);
    private readonly IReadOnlyDictionary<string, string> _refused;

    /// <summary>A family of the words, in the order they are listed.</summary>
    /// <param name="words">The kinds, each with what it becomes.</param>
    /// <param name="refused">The names left out on purpose, each with why and with the word that says it instead.</param>
    /// <exception cref="ArgumentException">A kind is registered without what it becomes, or two under one name.</exception>
    public KindVocabulary(IReadOnlyList<KindWord<TLower>> words, IReadOnlyDictionary<string, string> refused)
    {
        _refused = refused;

        foreach (var word in words)
        {
            if (word.Lower is null)
            {
                throw new ArgumentException($"'{word.Name}' is registered without what a part of it becomes.", nameof(words));
            }

            if (!_lowered.TryAdd(word.Name, word.Lower))
            {
                throw new ArgumentException($"'{word.Name}' is registered twice in one family of words.", nameof(words));
            }

            _kinds.Add(word.Name, new PartKind(word.Name, word.Purpose, word.Settings));
        }

        Kinds = [.. words.Select(word => _kinds[word.Name])];
    }

    /// <summary>The kinds, in the order they were registered: the list a step's parameter publishes.</summary>
    public IReadOnlyList<PartKind> Kinds { get; }

    /// <summary>The kind a name stands for.</summary>
    /// <param name="name">The name.</param>
    /// <returns>The kind.</returns>
    /// <exception cref="KeyNotFoundException">No kind of this family goes by that name.</exception>
    public PartKind Kind(string name) => _kinds[name];

    /// <summary>What makes the thing a part names, looked up by the name it gives.</summary>
    /// <param name="part">The part.</param>
    /// <returns>What makes it.</returns>
    /// <exception cref="NotSupportedException">It gives a name these words do not know; one left out on purpose says why.</exception>
    public TLower Lowered(PartDeclaration part) =>
        _lowered.TryGetValue(part.Kind, out var lower)
            ? lower
            : throw new NotSupportedException(_refused.TryGetValue(part.Kind, out var why)
                ? $"'{part.Kind}' is not a word a declared network is written in: {why}"
                : $"'{part.Kind}' is not a word a declared network is written in: {Kinds.Select(kind => kind.Name).Quoted()}.");
}

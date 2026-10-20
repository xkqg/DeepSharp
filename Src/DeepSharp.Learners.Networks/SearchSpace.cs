// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Globalization;
using DeepSharp.Networks;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// One thing a search may vary, under a name.
/// </summary>
/// <remarks>
/// A dimension draws its value from the numbers it is handed and from nothing else, so the same numbers give the same
/// value on every machine and in every order the draws are made. The kinds are the four there are —
/// <see cref="NumberRange"/>, <see cref="LogRange"/>, <see cref="WholeRange"/> and <see cref="Choices"/> — and only this
/// package makes more, so a sampler written elsewhere can tell them apart by type.
/// </remarks>
public abstract class Dimension
{
    private protected Dimension(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Name = name;
    }

    /// <summary>What the dimension is called; a candidate reads its value under this name.</summary>
    public string Name { get; }

    /// <summary>
    /// The word of an earlier choice the dimension is there for, and only then — a momentum for the optimizer that takes one,
    /// as a space Optuna writes by running has it; there for every candidate, unless said.
    /// </summary>
    public Chosen? OnlyWhen { get; init; }

    /// <summary>Whether the dimension is there for a candidate, by what was drawn before it.</summary>
    /// <param name="drawn">The settings drawn for the candidate so far, or all of them.</param>
    /// <returns>Whether it is there: always, unless it hangs on a choice, and then when that choice was drawn as its word.</returns>
    public bool IsThereAmong(IReadOnlyList<Setting> drawn)
    {
        ArgumentNullException.ThrowIfNull(drawn);

        return OnlyWhen is not { } chosen || drawn.Any(setting => setting.Name == chosen.Dimension && setting.Choice == chosen.Word);
    }

    /// <summary>Draws a value.</summary>
    /// <param name="draws">The numbers to draw it from.</param>
    /// <returns>The value, under this dimension's name.</returns>
    public abstract Setting Draw(Draws draws);

    /// <summary>Why the dimension cannot take a setting a sampler gave it; nothing when it can.</summary>
    /// <param name="setting">The setting, which is for this dimension.</param>
    /// <returns>The reason, as the words that follow the dimension's name; nothing for a value it takes.</returns>
    internal abstract string? Refusal(Setting setting);

    private protected static string? RefusalOfANumber(Setting setting, double low, double high) =>
        setting.Choice is not null ? $"is a number, and was given the word {setting.Choice}"
        : !double.IsFinite(setting.Number) ? string.Create(CultureInfo.InvariantCulture, $"is a number between {low} and {high}, and was given {setting.Number}")
        : setting.Number < low || setting.Number > high ? string.Create(CultureInfo.InvariantCulture, $"is a number between {low} and {high}, and was given {setting.Number}")
        : null;

    private protected static void ThrowIfNotARange(string name, double low, double high)
    {
        if (!double.IsFinite(low) || !double.IsFinite(high) || low >= high)
        {
            throw new ArgumentOutOfRangeException(
                name,
                string.Create(CultureInfo.InvariantCulture, $"{low} to {high}"),
                $"'{name}' varies between two numbers, the first below the second, both numbers.");
        }
    }
}

/// <summary>A number anywhere between two bounds, every part of the range as likely as another.</summary>
public sealed class NumberRange : Dimension
{
    /// <summary>Declares the range.</summary>
    /// <param name="name">What the dimension is called.</param>
    /// <param name="low">The smallest value.</param>
    /// <param name="high">The value it stays below.</param>
    /// <exception cref="ArgumentOutOfRangeException">The bounds are not numbers, or the first is not below the second.</exception>
    public NumberRange(string name, double low, double high)
        : base(name)
    {
        ThrowIfNotARange(name, low, high);

        Low = low;
        High = high;
    }

    /// <summary>The smallest value.</summary>
    public double Low { get; }

    /// <summary>The value it stays below.</summary>
    public double High { get; }

    /// <inheritdoc />
    public override Setting Draw(Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);

        return new Setting(Name, Low + (draws.NextDouble() * (High - Low)), null);
    }

    /// <inheritdoc />
    internal override string? Refusal(Setting setting) => RefusalOfANumber(setting, Low, High);
}

/// <summary>A number between two bounds above nought, every decade of the range as likely as another: a learning rate.</summary>
public sealed class LogRange : Dimension
{
    /// <summary>Declares the range.</summary>
    /// <param name="name">What the dimension is called.</param>
    /// <param name="low">The smallest value, above nought.</param>
    /// <param name="high">The value it stays below.</param>
    /// <exception cref="ArgumentOutOfRangeException">The bounds are not numbers above nought, or the first is not below the second.</exception>
    public LogRange(string name, double low, double high)
        : base(name)
    {
        ThrowIfNotARange(name, low, high);

        if (low <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(low), low, $"'{name}' spreads across the decades between its bounds, which are numbers above nought.");
        }

        Low = low;
        High = high;
    }

    /// <summary>The smallest value.</summary>
    public double Low { get; }

    /// <summary>The value it stays below.</summary>
    public double High { get; }

    /// <inheritdoc />
    public override Setting Draw(Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);

        var logLow = Math.Log(Low);
        var value = Math.Exp(logLow + (draws.NextDouble() * (Math.Log(High) - logLow)));

        return new Setting(Name, Math.Clamp(value, Low, High), null);
    }

    /// <inheritdoc />
    internal override string? Refusal(Setting setting) => RefusalOfANumber(setting, Low, High);
}

/// <summary>A whole number between two bounds, both of them included.</summary>
public sealed class WholeRange : Dimension
{
    /// <summary>Declares the range.</summary>
    /// <param name="name">What the dimension is called.</param>
    /// <param name="low">The smallest value.</param>
    /// <param name="high">The largest value.</param>
    /// <exception cref="ArgumentOutOfRangeException">The first bound is not below the second.</exception>
    public WholeRange(string name, int low, int high)
        : base(name)
    {
        ThrowIfNotARange(name, low, high);

        Low = low;
        High = high;
    }

    /// <summary>The smallest value.</summary>
    public int Low { get; }

    /// <summary>The largest value.</summary>
    public int High { get; }

    /// <inheritdoc />
    public override Setting Draw(Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);

        // Over a range of the whole of int the count does not fit an int; a range that wide is no layer's width.
        var count = checked(High - Low + 1);

        return new Setting(Name, Low + draws.NextInt(count), null);
    }

    /// <inheritdoc />
    internal override string? Refusal(Setting setting) =>
        RefusalOfANumber(setting, Low, High) is { } refused ? refused
        : setting.Number != Math.Floor(setting.Number) ? string.Create(CultureInfo.InvariantCulture, $"is a whole number between {Low} and {High}, and was given {setting.Number}")
        : null;
}

/// <summary>One of a few words: an optimizer, an activation.</summary>
public sealed class Choices : Dimension
{
    /// <summary>Declares the words.</summary>
    /// <param name="name">What the dimension is called.</param>
    /// <param name="options">The words, at least two, none empty and none twice.</param>
    /// <exception cref="ArgumentException">There are fewer than two words, a word is empty, or a word is there twice.</exception>
    public Choices(string name, IEnumerable<string> options)
        : base(name)
    {
        ArgumentNullException.ThrowIfNull(options);

        Options = [.. options];

        if (Options.Count < 2 || Options.Any(string.IsNullOrWhiteSpace) || Options.Distinct(StringComparer.Ordinal).Count() != Options.Count)
        {
            throw new ArgumentException($"'{name}' is one of at least two different words, none of them empty.", nameof(options));
        }
    }

    /// <summary>The words, in the order they were given.</summary>
    public IReadOnlyList<string> Options { get; }

    /// <inheritdoc />
    public override Setting Draw(Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);

        var at = draws.NextInt(Options.Count);

        return new Setting(Name, at, Options[at]);
    }

    /// <inheritdoc />
    internal override string? Refusal(Setting setting)
    {
        var at = setting.Choice is null ? -1 : Options.ToList().IndexOf(setting.Choice);

        return at < 0 ? $"is one of {string.Join(", ", Options)}, and was given {(setting.Choice is null ? "a number" : setting.Choice)}"
            : setting.Number != at ? string.Create(CultureInfo.InvariantCulture, $"is one of {string.Join(", ", Options)}, and was given the word {setting.Choice} as the place {setting.Number}, which is place {at}")
            : null;
    }
}

/// <summary>A word of a choice that a dimension is there for.</summary>
/// <param name="Dimension">The name of the choice, a dimension of words earlier in the space.</param>
/// <param name="Word">The word that, chosen, puts the dimension there.</param>
public readonly record struct Chosen(string Dimension, string Word);

/// <summary>
/// One value a search gave to one dimension.
/// </summary>
/// <param name="Name">The dimension it is for.</param>
/// <param name="Number">The number: the value itself for a range, and the place of the word among the options for a choice.</param>
/// <param name="Choice">The word, for a choice; nothing for a number.</param>
public readonly record struct Setting(string Name, double Number, string? Choice);

/// <summary>
/// What a search varies: its dimensions, each under a name of its own.
/// </summary>
public sealed class SearchSpace : IReadOnlyList<Dimension>
{
    private readonly Dimension[] _dimensions;

    /// <summary>Declares the space.</summary>
    /// <param name="dimensions">
    /// The dimensions, at least one, no two under one name, in the order they are drawn: a dimension that is there only when a
    /// word is chosen stands after the choice it hangs on.
    /// </param>
    /// <exception cref="ArgumentException">
    /// There is none, two share a name, or one hangs on no earlier dimension of words, or on a word that dimension does not offer.
    /// </exception>
    public SearchSpace(IEnumerable<Dimension> dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);

        _dimensions = [.. dimensions];

        if (_dimensions.Length == 0)
        {
            throw new ArgumentException("A search space with no dimension has nothing to vary.", nameof(dimensions));
        }

        var twice = _dimensions.GroupBy(each => each.Name, StringComparer.Ordinal).FirstOrDefault(group => group.Count() > 1);

        if (twice is not null)
        {
            throw new ArgumentException($"Two dimensions are called '{twice.Key}': a candidate reads each by its name.", nameof(dimensions));
        }

        for (var at = 0; at < _dimensions.Length; at++)
        {
            if (_dimensions[at].OnlyWhen is { } chosen
                && !_dimensions.Take(at).OfType<Choices>().Any(choice => choice.Name == chosen.Dimension && choice.Options.Contains(chosen.Word, StringComparer.Ordinal)))
            {
                throw new ArgumentException(
                    $"'{_dimensions[at].Name}' is there only when '{chosen.Word}' is chosen for '{chosen.Dimension}', and no dimension of words before it offers that word: "
                    + "a dimension hangs on a choice drawn before it.",
                    nameof(dimensions));
            }
        }
    }

    /// <inheritdoc />
    public int Count => _dimensions.Length;

    /// <inheritdoc />
    public Dimension this[int index] => _dimensions[index];

    /// <inheritdoc />
    public IEnumerator<Dimension> GetEnumerator() => ((IEnumerable<Dimension>)_dimensions).GetEnumerator();

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>
/// One point of a search space: a value for each dimension, read by name.
/// </summary>
public sealed class Candidate : IEquatable<Candidate>
{
    /// <summary>Gathers the settings of one point.</summary>
    /// <param name="settings">The settings, no two under one name.</param>
    /// <exception cref="ArgumentException">Two settings are for one name.</exception>
    public Candidate(IEnumerable<Setting> settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        Settings = [.. settings];

        if (Settings.Select(each => each.Name).Distinct(StringComparer.Ordinal).Count() != Settings.Count)
        {
            throw new ArgumentException("A candidate holds one setting for a name.", nameof(settings));
        }
    }

    /// <summary>The settings, in the order of the dimensions they were drawn for.</summary>
    public IReadOnlyList<Setting> Settings { get; }

    /// <summary>A setting that is a number.</summary>
    /// <param name="name">The dimension.</param>
    /// <returns>The number.</returns>
    /// <exception cref="ArgumentException">No setting has that name.</exception>
    /// <exception cref="InvalidOperationException">The setting is a word.</exception>
    public double Number(string name)
    {
        var setting = Named(name);

        return setting.Choice is null
            ? setting.Number
            : throw new InvalidOperationException($"'{name}' is the word '{setting.Choice}', and not a number: read it with Choice.");
    }

    /// <summary>A setting that is a whole number.</summary>
    /// <param name="name">The dimension.</param>
    /// <returns>The whole number.</returns>
    /// <exception cref="ArgumentException">No setting has that name.</exception>
    /// <exception cref="InvalidOperationException">The setting is a word, or a number that is not whole.</exception>
    public int Whole(string name)
    {
        var number = Number(name);

        return double.IsInteger(number) && Math.Abs(number) <= int.MaxValue
            ? (int)number
            : throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture, $"'{name}' is {number}, which is not a whole number."));
    }

    /// <summary>A setting that is a word.</summary>
    /// <param name="name">The dimension.</param>
    /// <returns>The word.</returns>
    /// <exception cref="ArgumentException">No setting has that name.</exception>
    /// <exception cref="InvalidOperationException">The setting is a number.</exception>
    public string Choice(string name)
    {
        var setting = Named(name);

        return setting.Choice
            ?? throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture, $"'{name}' is the number {setting.Number}, and not a word: read it with Number."));
    }

    /// <inheritdoc />
    public bool Equals(Candidate? other) => other is not null && Settings.SequenceEqual(other.Settings);

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as Candidate);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var setting in Settings)
        {
            hash.Add(setting);
        }

        return hash.ToHashCode();
    }

    private Setting Named(string name) =>
        Settings.FirstOrDefault(each => each.Name == name) is { Name: not null } found
            ? found
            : throw new ArgumentException(
                $"This candidate holds no setting called '{name}': it holds {string.Join(", ", Settings.Select(each => $"'{each.Name}'"))}.", nameof(name));
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// One kind of deal a study trains every candidate on — a split at random, say, or a split in time — with how many deals
/// it is, and how much its mean weighs in the number a trial is judged by.
/// </summary>
/// <remarks>
/// <para>
/// A study's deals are laid out kind after kind, in the order its <see cref="Study.Kinds"/> names them, so deal 0 is the
/// first deal of the first kind; <see cref="DealKindExtensions.KindOf"/> says which kind a deal is, and which of its deals.
/// A trial is judged by the mean of each kind's scores, those means weighed by the kinds' weights.
/// </para>
/// <para>
/// The promise a study makes is made within a kind: every deal of one kind measures on the same test rows, so a candidate
/// is chosen on no row its own kind measures on. Across kinds it is not made. A row one kind measures on may be learned from
/// by another kind — the last weeks a split in time measures on are training rows to a split at random — so a kind's test
/// number is honest only within that kind.
/// </para>
/// </remarks>
public readonly record struct DealKind
{
    /// <summary>Declares a kind of deal.</summary>
    /// <param name="name">What the kind is called: the name a refusal gives, and that tells the kinds apart.</param>
    /// <param name="deals">How many deals are of this kind: at least one.</param>
    /// <param name="weight">How much this kind's mean weighs in a trial's score: a number above nought.</param>
    /// <exception cref="ArgumentNullException">There is no name.</exception>
    /// <exception cref="ArgumentException">The name is empty, or only spaces.</exception>
    /// <exception cref="ArgumentOutOfRangeException">There are no deals, or the weight is nought, below it, or not a number.</exception>
    public DealKind(string name, int deals, double weight)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Name = name;
        Deals = deals >= 1 ? deals : throw new ArgumentOutOfRangeException(nameof(deals), deals, $"The kind '{name}' is of at least one deal.");
        Weight = double.IsFinite(weight) && weight > 0
            ? weight
            : throw new ArgumentOutOfRangeException(nameof(weight), weight, $"The kind '{name}' weighs a number above nought in a trial's score.");
    }

    /// <summary>What the kind is called.</summary>
    public string Name { get; }

    /// <summary>How many deals are of this kind.</summary>
    public int Deals { get; }

    /// <summary>How much this kind's mean weighs in a trial's score.</summary>
    public double Weight { get; }

    /// <summary>The kind as a refusal names it: its name, its deals and its weight.</summary>
    internal string Described => string.Create(CultureInfo.InvariantCulture, $"'{Name}' of {Deals} deals, weighing {Weight}");
}

/// <summary>Which kind a deal of a study is, and which of that kind's deals.</summary>
/// <param name="Kind">The kind the deal is of.</param>
/// <param name="Deal">Which of the kind's deals it is, from nought.</param>
public readonly record struct DealOfAKind(DealKind Kind, int Deal);

/// <summary>Reading a study's kinds of deal: which kind a deal is, and the numbers each kind gives a trial.</summary>
public static class DealKindExtensions
{
    extension(IReadOnlyList<DealKind> kinds)
    {
        /// <summary>Which kind a deal is, the deals being laid out kind after kind, and which of that kind's deals.</summary>
        /// <param name="deal">The deal, from nought, as a pipeline for a deal is handed it.</param>
        /// <returns>The kind, and the deal's place within it from nought.</returns>
        /// <exception cref="ArgumentNullException">There are no kinds.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The deal is below nought, or past the last deal of the last kind.</exception>
        /// <remarks>
        /// A pipeline for a deal reads it to deal its rows as the kind says: <c>kinds.KindOf(deal).Kind.Name == "time" ? … : …</c>.
        /// </remarks>
        public DealOfAKind KindOf(int deal)
        {
            ArgumentNullException.ThrowIfNull(kinds);

            var within = deal;

            foreach (var kind in kinds)
            {
                if (within >= 0 && within < kind.Deals)
                {
                    return new DealOfAKind(kind, within);
                }

                within -= kind.Deals;
            }

            throw new ArgumentOutOfRangeException(nameof(deal), deal, $"The kinds hold {kinds.Sum(kind => kind.Deals)} deals, numbered from nought.");
        }

        /// <summary>The values of every deal, cut kind by kind: one list for each kind, in the order they are dealt; one list for no kinds.</summary>
        /// <param name="values">The values, in deal order; fewer than the kinds' deals for a trial stopped before its last.</param>
        internal IReadOnlyList<IReadOnlyList<T>> Cut<T>(IReadOnlyList<T> values)
        {
            if (kinds.Count == 0)
            {
                return [values];
            }

            var cut = new List<IReadOnlyList<T>>(kinds.Count);
            var from = 0;

            foreach (var kind in kinds)
            {
                cut.Add([.. values.Skip(from).Take(kind.Deals)]);
                from += kind.Deals;
            }

            return cut;
        }

        /// <summary>
        /// The number of each kind that judged a deal, weighed by the kinds' weights: the kind's own number where only one did,
        /// as for no kinds at all, so a study of one kind is judged exactly as one of plain deals.
        /// </summary>
        /// <param name="values">The values, in deal order.</param>
        /// <param name="ofAKind">The number a kind's values give.</param>
        internal double Weighed<T>(IReadOnlyList<T> values, Func<IReadOnlyList<T>, double> ofAKind)
        {
            var cut = kinds.Cut(values);
            var reached = Enumerable.Range(0, cut.Count).Where(at => cut[at].Count > 0).ToArray();

            if (reached.Length <= 1)
            {
                return ofAKind(reached.Length == 1 ? cut[reached[0]] : values);
            }

            return reached.Sum(at => kinds[at].Weight * ofAKind(cut[at])) / reached.Sum(at => kinds[at].Weight);
        }
    }
}

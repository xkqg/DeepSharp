// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>How several members' answers become one, once each has come back into the answer's own units.</summary>
/// <remarks>
/// Numbers are always averaged, and an average is taken of what a person reads: a log is undone before the members meet, since
/// the mean of exponentials is not the exponential of the mean. Classes are either averaged as chances or counted as votes.
/// </remarks>
public enum Blend
{
    /// <summary>The mean of the members' answers: a number is their mean, a chance their mean chance.</summary>
    Average,

    /// <summary>
    /// Each member names its classes — a chance of a half or more, or the likeliest labels where a row holds a number of them —
    /// and the classes most members named are the answer, a tie going to the higher mean chance. Of classes only: votes on answers
    /// that are amounts are refused where the members are put together.
    /// </summary>
    Vote,
}

/// <summary>An ensemble measured by its pipeline's report, beside each of its members measured by the same report on the same rows.</summary>
/// <remarks>
/// The members are measured by what they said; the ensemble by what they said combined, in the answer's own units. The numbers of
/// a member on the test rows are reported and choose nothing: how many members an ensemble holds, and which, are chosen on the
/// validation rows, as everything else is.
/// </remarks>
public sealed class EnsembleMeasures
{
    internal EnsembleMeasures(Measures ensemble, IReadOnlyList<Measures> members, Blend blend)
    {
        Ensemble = ensemble;
        Members = members;
        Blend = blend;
    }

    /// <summary>What the members said, combined, measured.</summary>
    /// <remarks>It can be drawn, and it crosses as text member by member: what each member said is text, what they said together is not.</remarks>
    public Measures Ensemble { get; }

    /// <summary>Each member measured alone, in the order they were given.</summary>
    public IReadOnlyList<Measures> Members { get; }

    /// <summary>How the members' answers were combined.</summary>
    public Blend Blend { get; }
}

/// <summary>Combining several members' answers into one, and what is refused before they are.</summary>
/// <remarks>
/// <see cref="CombinedBackExtensions"/> is the one place members' answers are combined, once each has come back through its own
/// pipeline; this holds how, and the two refusals every place that puts members together asks: too few of them, and votes on
/// answers that are amounts.
/// </remarks>
internal static class Blending
{
    // An answer within a millionth of nought or one is that class: a way back is held to leading back within as much.
    private const double Tolerance = 1e-6;

    extension(Blend blend)
    {
        /// <summary>The answers of several members for the same rows, in the answer's own units, as one answer for each row.</summary>
        /// <param name="members">For each member, a row of answers for each row, the members' rows the same rows.</param>
        /// <param name="ones">How many answers a row holds as one, when they are classes; nought for any number.</param>
        /// <returns>For each row, the combined answers.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The blend is none of the blends.</exception>
        /// <remarks>Asked by <see cref="CombinedBackExtensions"/> alone.</remarks>
        public IReadOnlyList<double[]> Combine(IReadOnlyList<IReadOnlyList<double[]>> members, int ones) =>
            blend switch
            {
                Blend.Average => [.. Enumerable.Range(0, members[0].Count).Select(row => Averaged(members, row))],
                Blend.Vote => [.. Enumerable.Range(0, members[0].Count).Select(row => Voted(members, row, ones))],
                _ => throw Undefined(blend),
            };

        /// <summary>Refuses a value that is none of the blends: a number cast to the set.</summary>
        /// <exception cref="ArgumentOutOfRangeException">The blend is none of the blends.</exception>
        public void ThrowIfUndefined()
        {
            if (!Vocabulary<Blend>.Has(blend))
            {
                throw Undefined(blend);
            }
        }

        /// <summary>
        /// Refuses votes counted of answers that are amounts, where members are put together — before any of them serves, is
        /// calibrated on or is measured — decided by the output of each pipeline the members were trained behind.
        /// </summary>
        /// <param name="pipelines">The pipeline each member was trained behind.</param>
        /// <exception cref="InvalidOperationException">
        /// The blend is a vote, and an output says its answers are amounts, or — where its answers may be classes and the run holds
        /// its rows — a training row's answer, in its own units, is neither nought nor one.
        /// </exception>
        /// <remarks>
        /// An output that says its answers are amounts — free numbers, shares of a whole, a return — decides it alone. One whose
        /// answers may be classes, as the column a target names may, is read where it can be: its training rows' answers, brought back
        /// into their own units. A pipeline read from its file holds no rows, and its output decides alone.
        /// </remarks>
        public void ThrowIfVotedOnAmounts(IReadOnlyList<PreparedData> pipelines)
        {
            if (blend != Blend.Vote)
            {
                return;
            }

            foreach (var prepared in pipelines.Distinct())
            {
                if (prepared.Declaration.Output is not { } output)
                {
                    continue;
                }

                if (!output.AnswersCanBeClasses)
                {
                    throw new InvalidOperationException(
                        $"'{output.Verb}' names its answers {output.Answers.Quoted()} as amounts, and an amount is no class: votes are counted of classes, each nought or one. The members' answers to an amount are averaged.");
                }

                if (prepared.Table.RowCount > 0)
                {
                    ThrowIfAnyIsNoClass(prepared, output);
                }
            }
        }
    }

    extension<T>(IReadOnlyCollection<T> members)
    {
        /// <summary>What is said of too few members to be an ensemble, in the one sentence it is said in; nothing when there are two or more.</summary>
        /// <param name="noun">What the members are, in the plural: networks, candidates, members.</param>
        /// <param name="instead">What to do with fewer: train one with Train, say.</param>
        /// <returns>The sentence, or nothing.</returns>
        public string? FewerThanTwo(string noun, string instead) =>
            members.Count >= 2
                ? null
                : string.Create(CultureInfo.InvariantCulture, $"An ensemble is of at least two {noun}, and {members.Count} {(members.Count == 1 ? "was" : "were")} given: {instead}.");

        /// <summary>Refuses too few members to be an ensemble.</summary>
        /// <param name="noun">What the members are, in the plural.</param>
        /// <param name="instead">What to do with fewer.</param>
        /// <param name="parameter">The parameter the members were handed as.</param>
        /// <exception cref="ArgumentException">There are fewer than two.</exception>
        public void ThrowIfFewerThanTwo(string noun, string instead, string parameter)
        {
            if (members.FewerThanTwo(noun, instead) is { } fewer)
            {
                throw new ArgumentException(fewer, parameter);
            }
        }
    }

    // Every training row's answers, brought back into their own units, nought or one: votes are counted of classes.
    private static void ThrowIfAnyIsNoClass(PreparedData prepared, INamesTheAnswer output)
    {
        var actual = prepared.BackToOriginal(Handover.AnswersOf(prepared, Part.Train).Answers!, Part.Train);
        var readAt = prepared.ReadAtOf(Part.Train);

        for (var row = 0; row < actual.Count; row++)
        {
            for (var column = 0; column < actual[row].Length; column++)
            {
                var value = actual[row][column];

                if (Math.Abs(value) > Tolerance && Math.Abs(value - 1) > Tolerance)
                {
                    throw new InvalidOperationException(string.Create(
                        CultureInfo.InvariantCulture,
                        $"Row {readAt[row] + 1} of '{output.Answers[column]}' is {value}, which is no class: votes are counted of classes, each nought or one. The members' answers to an amount are averaged."));
                }
            }
        }
    }

    // The one refusal of a value that is none of the blends, however it came to be asked for.
    private static ArgumentOutOfRangeException Undefined(Blend blend) =>
        new(nameof(blend), blend, "An ensemble's answers are averaged or voted for.");

    private static double[] Averaged(IReadOnlyList<IReadOnlyList<double[]>> members, int row) =>
        [.. Enumerable.Range(0, members[0][row].Length).Select(column => members.Average(member => member[row][column]))];

    // Each member names its classes: the answers at a half or more, or the likeliest `ones` where a row holds that many, an
    // earlier answer first among equals. The classes named by most members are the answer; a tie goes to the higher mean chance.
    private static double[] Voted(IReadOnlyList<IReadOnlyList<double[]>> members, int row, int ones)
    {
        var width = members[0][row].Length;
        var mean = Enumerable.Range(0, width).Select(column => members.Average(member => member[row][column])).ToArray();
        var votes = new int[width];

        foreach (var member in members)
        {
            foreach (var column in Named(member[row], ones))
            {
                votes[column]++;
            }
        }

        var answer = new double[width];

        if (ones == 0)
        {
            for (var column = 0; column < width; column++)
            {
                answer[column] = (votes[column] * 2 > members.Count || (votes[column] * 2 == members.Count && mean[column] >= 0.5)) ? 1 : 0;
            }

            return answer;
        }

        foreach (var column in Enumerable.Range(0, width).OrderByDescending(column => votes[column]).ThenByDescending(column => mean[column]).Take(ones))
        {
            answer[column] = 1;
        }

        return answer;
    }

    private static IEnumerable<int> Named(double[] said, int ones) =>
        ones == 0
            ? Enumerable.Range(0, said.Length).Where(column => said[column] >= 0.5)
            : Enumerable.Range(0, said.Length).OrderByDescending(column => said[column]).Take(ones);
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

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
    /// and the classes most members named are the answer, a tie going to the higher mean chance. Of classes only.
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

/// <summary>Combining several members' answers into one.</summary>
internal static class Blending
{
    extension(Blend blend)
    {
        /// <summary>The answers of several members for the same rows, in the answer's own units, as one answer for each row.</summary>
        /// <param name="members">For each member, a row of answers for each row, the members' rows the same rows.</param>
        /// <param name="ones">How many answers a row holds as one, when they are classes; nought for any number.</param>
        /// <returns>For each row, the combined answers.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The blend is none of the blends.</exception>
        public IReadOnlyList<double[]> Combine(IReadOnlyList<IReadOnlyList<double[]>> members, int ones) =>
            blend switch
            {
                Blend.Average => [.. Enumerable.Range(0, members[0].Count).Select(row => Averaged(members, row))],
                Blend.Vote => [.. Enumerable.Range(0, members[0].Count).Select(row => Voted(members, row, ones))],
                _ => throw new ArgumentOutOfRangeException(nameof(blend), blend, "An ensemble's answers are averaged or voted for."),
            };
    }

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

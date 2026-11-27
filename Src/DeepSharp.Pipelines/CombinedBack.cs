// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>What one member of an ensemble predicted for some parts, and the pipeline it was trained behind: what brings its answers back.</summary>
/// <param name="Prepared">The run of the pipeline the member was trained behind, which handed it the rows of every part.</param>
/// <param name="Parts">What the member predicted for each part, for the batches its own pipeline handed over, in the units it handed them over in.</param>
internal readonly record struct MemberPredictions(PreparedData Prepared, IReadOnlyList<PartPredictions> Parts)
{
    /// <summary>What the member predicted for one part.</summary>
    /// <param name="part">The part.</param>
    /// <returns>The predictions for it, the first when several are given.</returns>
    internal PartPredictions On(Part part) => Parts.First(each => each.Batch.Part == part);
}

/// <summary>
/// The one place members' answers meet: each member's answers brought back through its own pipeline's way back, put in one order of
/// rows, and combined there.
/// </summary>
/// <remarks>
/// A way back is not a straight line — a log is undone by an exponential — so members meet only once each is in the answer's own
/// units; and a member behind a pipeline of its own hands its rows over in its own order, so each is put in the first member's
/// order of rows before they meet: served rows by where they were handed in, the rows of a part by their keys. Which rows members
/// may meet on is held here too: an ensemble is measured and calibrated on rows every member set aside alike.
/// </remarks>
internal static class CombinedBackExtensions
{
    extension(Blend blend)
    {
        /// <summary>Several members' answers for served rows, each already back in the answer's own units, combined row by row.</summary>
        /// <param name="members">For each member, its answers and what its own pipeline served.</param>
        /// <param name="ones">How many answers a row holds as one, when they are classes; nought for any number.</param>
        /// <returns>The combined answers, in the order the first member's pipeline served the rows, and what it served.</returns>
        /// <exception cref="InvalidOperationException">A row handed in is answered by one member and not by another, whose pipeline left it out.</exception>
        internal ServedAnswers CombinedBack(IReadOnlyList<ServedAnswers> members, int ones)
        {
            var first = members[0].Served.HandedInAt;
            IReadOnlyList<double[]>[] aligned = [.. members.Select((member, at) => Aligned(member, first, at))];

            return new ServedAnswers(blend.Combine(aligned, ones), members[0].Served);
        }

        /// <summary>Several members' predictions for one part's rows, each brought back through its own pipeline, combined row by row.</summary>
        /// <param name="members">For each member, its pipeline and what it predicted for the part.</param>
        /// <param name="part">The part, whose rows every member set aside alike.</param>
        /// <param name="ones">How many answers a row holds as one, when they are classes; nought for any number.</param>
        /// <returns>The combined answers, in the order the first member's pipeline hands the part's rows over.</returns>
        internal IReadOnlyList<double[]> CombinedBack(IReadOnlyList<MemberPredictions> members, Part part, int ones)
        {
            var first = members[0].On(part).Batch.Keys!;
            IReadOnlyList<double[]>[] aligned =
            [
                .. members.Select(member =>
                {
                    var said = member.On(part);

                    return InTheOrderOf(member.Prepared.BackToOriginal(said.Predictions, part), said.Batch.Keys!, first);
                }),
            ];

            return blend.Combine(aligned, ones);
        }
    }

    extension(IReadOnlyList<PreparedData> pipelines)
    {
        /// <summary>Holds every member's pipeline to setting one part's rows aside alike, as the first member's did.</summary>
        /// <param name="part">The part the ensemble is measured or calibrated on.</param>
        /// <exception cref="InvalidOperationException">
        /// A row is in one member's part and not in another's: named as a member that learned from a row another measures on, where
        /// it did, and otherwise as the row the two set aside apart.
        /// </exception>
        /// <remarks>
        /// Each member's own split keeps the rows it measures on away from those it learned from, but not from those another member
        /// learned from: through the blend, a member trained on a row would be measured on it. Rows set aside alike are what makes the
        /// ensemble's number a number on rows no member learned from, and they are compared by their keys, never by where they stand.
        /// </remarks>
        internal void ThrowIfTheyMeasureApart(Part part)
        {
            var first = Counted(pipelines[0].KeysOf(part));

            for (var at = 1; at < pipelines.Count; at++)
            {
                var apart = new Dictionary<RowKey, int>(first);

                foreach (var key in pipelines[at].KeysOf(part))
                {
                    apart[key] = apart.GetValueOrDefault(key) - 1;
                }

                RowKey[] differing = [.. apart.Where(each => each.Value != 0).Select(each => each.Key)];

                if (differing.Length > 0)
                {
                    throw Apart(new MembersApart(pipelines, at, part), differing, apart);
                }
            }
        }
    }

    /// <summary>Two members of an ensemble whose parts disagree, and the part they disagree on.</summary>
    /// <param name="Pipelines">Every member's pipeline.</param>
    /// <param name="Other">The member that disagrees with the first.</param>
    /// <param name="Part">The part.</param>
    private readonly record struct MembersApart(IReadOnlyList<PreparedData> Pipelines, int Other, Part Part);

    // How many rows of each key a part holds.
    private static Dictionary<RowKey, int> Counted(IReadOnlyList<RowKey> keys)
    {
        var counted = new Dictionary<RowKey, int>();

        foreach (var key in keys)
        {
            counted[key] = counted.GetValueOrDefault(key) + 1;
        }

        return counted;
    }

    // The refusal of two members' parts that disagree: a member that learned from a row the other measures on, where there is one,
    // since that is what would leak through the blend; otherwise a row one of them left out, or the first they set aside apart.
    private static InvalidOperationException Apart(MembersApart members, RowKey[] differing, Dictionary<RowKey, int> apart)
    {
        var word = Vocabulary<Part>.WordFor(members.Part, "parts");
        PlacedRow[] placed = [.. differing.Select(key => Placed(members, key, apart[key] > 0))];

        // The other member holds the row in another part than this one, so it learned from it only where this is no training.
        if (placed.Where(each => each.There is { } there && there.IsLearnedFrom).Take(1).ToArray() is [var learned])
        {
            return new InvalidOperationException(
                $"Member {learned.Without + 1} learned from row {learned.Row}, which member {learned.Holder + 1} measures on as '{word}': through the blend, its answer there would be measured on a row it was trained on. "
                + "An ensemble is measured and calibrated on rows every member set aside alike: deal them alike for every member, the same split with the same seeds.");
        }

        // A row one member's pipeline left out says more than one it holds in another part.
        var first = placed.OrderBy(each => each.There is null ? 0 : 1).First();
        var there = first.There is { } part ? $"holds it as '{Vocabulary<Part>.WordFor(part, "parts")}'" : "does not hold it";

        return new InvalidOperationException(
            $"Member {first.Holder + 1} sets row {first.Row} aside as '{word}', and member {first.Without + 1} {there}: "
            + "an ensemble is measured and calibrated on rows every member set aside alike. Deal them alike for every member, the same split with the same seeds, and keep the same rows.");
    }

    // Where a row two members disagree on stands: the member that holds it in the part, the row as that member read it, and what the
    // other member holds it as, if anything.
    private static PlacedRow Placed(MembersApart members, RowKey key, bool firstHoldsIt)
    {
        var holder = firstHoldsIt ? 0 : members.Other;
        var without = firstHoldsIt ? members.Other : 0;
        var held = members.Pipelines[holder];
        var row = held.RowsOf(members.Part).First(each => held.Table.Identities[each].Key == key);
        var other = members.Pipelines[without];
        var there = Enumerable.Range(0, other.Table.RowCount).Where(each => other.Table.Identities[each].Key == key).Select(each => (Part?)other.Parts[each]).FirstOrDefault();

        return new PlacedRow(holder, without, held.Table.Identities[row].ReadAt + 1, there);
    }

    /// <summary>A row two members set aside apart.</summary>
    /// <param name="Holder">The member that holds it in the part, from nought.</param>
    /// <param name="Without">The member that does not, from nought.</param>
    /// <param name="Row">The row, as the holder read it, from one.</param>
    /// <param name="There">The part the other member holds it in; nothing when it holds it nowhere.</param>
    private readonly record struct PlacedRow(int Holder, int Without, int Row, Part? There);

    // One member's answers for served rows in the order the first member's pipeline served them, found by where each was handed in.
    private static IReadOnlyList<double[]> Aligned(ServedAnswers member, IReadOnlyList<int> first, int at)
    {
        var answered = new Dictionary<int, int>();

        for (var row = 0; row < member.Served.HandedInAt.Count; row++)
        {
            answered[member.Served.HandedInAt[row]] = row;
        }

        var firsts = first.ToHashSet();

        if (first.Concat(answered.Keys).Where(handedIn => !(firsts.Contains(handedIn) && answered.ContainsKey(handedIn))).Order().FirstOrDefault(-1) is var left and >= 0)
        {
            var firstAnswered = firsts.Contains(left);

            throw new InvalidOperationException(
                $"Row {left + 1} handed in is answered by member {(firstAnswered ? 1 : at + 1)} and not by member {(firstAnswered ? at + 1 : 1)}, whose pipeline left it out: an ensemble answers a row with every member's answer or not at all. "
                + "Hand in rows every member's pipeline keeps, or give the members pipelines that keep the same rows.");
        }

        return [.. first.Select(handedIn => member.Answers[answered[handedIn]])];
    }

    // One member's answers for a part's rows in the order of the first member's rows, found by their keys: the same record in
    // another order, or twice, is found as the first member holds it.
    private static IReadOnlyList<double[]> InTheOrderOf(IReadOnlyList<double[]> answers, IReadOnlyList<RowKey> keys, IReadOnlyList<RowKey> first)
    {
        var waiting = new Dictionary<RowKey, Queue<int>>();

        for (var row = 0; row < keys.Count; row++)
        {
            if (!waiting.TryGetValue(keys[row], out var rows))
            {
                waiting[keys[row]] = rows = new Queue<int>();
            }

            rows.Enqueue(row);
        }

        return [.. first.Select(key => answers[waiting[key].Dequeue()])];
    }
}

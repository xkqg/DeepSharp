// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>How a model answers some rows: the answers, in their own units, of the rows it is handed.</summary>
/// <param name="rows">The rows, as the pipeline reads them: the columns they were read with.</param>
/// <returns>For each row, a number for each answer, in the order the rows were handed in.</returns>
public delegate IReadOnlyList<double[]> Answering(IRowSource rows);

/// <summary>The rows to explain, and the way the model that is explained answers them.</summary>
/// <param name="Rows">The rows as the pipeline reads them, with the columns they were read with. The answer's own column needs not be among them.</param>
/// <param name="Answering">How the model answers some rows: a trained network's <c>Predict(rows).Answers</c>, say.</param>
public readonly record struct RowsAnswered(IRowSource Rows, Answering Answering);

/// <summary>The draws a question about why is answered from: the seed they are worked out from, and how many are made.</summary>
/// <remarks>
/// Both are said by the caller and neither has a default: an explanation that was drawn from a number nobody chose cannot be drawn
/// again, and one drawn once says nothing of how much it owes to that draw. The seed is the question's own and is shared with nothing
/// the model was trained from.
/// </remarks>
public readonly record struct Redraws
{
    /// <summary>The draws of a question.</summary>
    /// <param name="seed">The number every draw is worked out from.</param>
    /// <param name="repeats">How many are made: for a column's reliance, how many times its values are shuffled; for a row's make-up, how many samples.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one.</exception>
    public Redraws(int seed, int repeats)
    {
        if (repeats < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(repeats), repeats, "A question is drawn at least once.");
        }

        Seed = seed;
        Repeats = repeats;
    }

    /// <summary>The number every draw is worked out from.</summary>
    public int Seed { get; }

    /// <summary>How many draws are made.</summary>
    public int Repeats { get; }
}

/// <summary>The column to move through its range, and at how many places.</summary>
public readonly record struct Moving
{
    /// <summary>A column to move.</summary>
    /// <param name="column">The column, as the rows were read with it.</param>
    /// <param name="points">How many places a number is moved to, from the smallest of the rows to the largest; a category goes to all of its values.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than two places.</exception>
    public Moving(string column, int points)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        if (points < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(points), points, "A column moves through its range at two places at the least.");
        }

        Column = column;
        Points = points;
    }

    /// <summary>The column.</summary>
    public string Column { get; }

    /// <summary>The places a number is moved to.</summary>
    public int Points { get; }
}

/// <summary>How far the answers move with a column, over the draws made.</summary>
/// <param name="Column">The column, as the rows were read with it.</param>
/// <param name="Mean">The mean, over the draws, of how far an answer moves when the column's values are shuffled among the rows, in the answer's own units.</param>
/// <param name="Spread">The sample standard deviation of that over the draws; nought for a single draw.</param>
public sealed record ColumnReliance(string Column, double Mean, double Spread);

/// <summary>Which columns the answers lean on: how far they move when a column's values are shuffled among the rows.</summary>
/// <param name="Answers">The answers measured, and averaged over.</param>
/// <param name="Seed">The number the shuffles were worked out from.</param>
/// <param name="Repeats">How many times each column was shuffled.</param>
/// <param name="Columns">Each column of the rows, the one the answers move most with first.</param>
/// <remarks>
/// The mean absolute change of an answer, in its own units, when the column's values are drawn among the rows again — model
/// reliance, worked out of the answers themselves. It needs no answer to compare with, so it is asked of rows a model has yet to
/// answer; an error-based importance, which scikit-learn's permutation importance is, would need each row's answer and gives no
/// number without it. A column the model ignores moves nothing; one it leans on moves it by as much as it varies and the model uses it.
/// A column that stands with another is drawn on its own, so a pair of columns that say the same thing each move the answers a little.
/// </remarks>
public sealed record Reliance(IReadOnlyList<string> Answers, int Seed, int Repeats, IReadOnlyList<ColumnReliance> Columns);

/// <summary>What the answers do as one column moves through its range, with the rest of each row left as it was.</summary>
/// <param name="Column">The column.</param>
/// <param name="Values">Where the column was moved to, as the rows write it.</param>
/// <param name="Points">The same places as numbers; nothing for a category, which has places and no distances.</param>
/// <param name="Answers">For each place, the mean answer over the rows, a number for each answer, in the answer's own units.</param>
/// <param name="AnswerNames">What each number of an answer stands for.</param>
/// <remarks>
/// Partial dependence: the column is set to each place in every row and the answers averaged. A number moves from the smallest
/// to the largest of the rows handed in, which is the range they hold and not a trimmed one; a category goes to every value the
/// training rows held.
/// </remarks>
public sealed record Dependence(string Column, IReadOnlyList<string> Values, IReadOnlyList<double>? Points, IReadOnlyList<double[]> Answers, IReadOnlyList<string> AnswerNames);

/// <summary>What one column adds to one row's answer.</summary>
/// <param name="Column">The column.</param>
/// <param name="Contributions">A number for each answer: how far the column moves the answer from the background's to the row's, in the answer's own units.</param>
public sealed record ColumnContribution(string Column, IReadOnlyList<double> Contributions);

/// <summary>What one row's answer is made of, column by column.</summary>
/// <param name="Row">Which of the rows handed in, from nought.</param>
/// <param name="Samples">How many samples the contributions are the mean of.</param>
/// <param name="AnswerNames">What each number of an answer stands for.</param>
/// <param name="Answer">The row's answer, a number for each answer.</param>
/// <param name="Background">The mean answer of the rows the samples took the background from.</param>
/// <param name="Columns">Each column of the rows, the one that moves the first answer most first.</param>
/// <remarks>
/// Sampling Shapley values after Štrumbelj and Kononenko: in each sample the columns are put in a random order and the row's own
/// values are put in, one column after another, over those of a row drawn at random; what each column adds is how far the answer
/// moves when it goes in. The contributions add up to the answer less the background, exactly, since each sample's steps
/// run from one to the other.
/// </remarks>
public sealed record RowMakeUp(int Row, int Samples, IReadOnlyList<string> AnswerNames, IReadOnlyList<double> Answer, IReadOnlyList<double> Background, IReadOnlyList<ColumnContribution> Columns);

/// <summary>Why a model answers as it does: perturbing the rows it is handed and replaying them, with nothing fitted and nothing learned.</summary>
public static class WhyExtensions
{
    extension(PreparedData prepared)
    {
        /// <summary>Which columns the answers lean on.</summary>
        /// <param name="handed">The rows, and how the model answers them.</param>
        /// <param name="draws">The seed the shuffles are worked out from and how many times each column is shuffled.</param>
        /// <returns>For each column, how far the answers move when its values are shuffled among the rows.</returns>
        /// <exception cref="ArgumentException">The pipeline names no answer, or the rows hold none.</exception>
        /// <exception cref="InvalidOperationException">The model answered other rows than it was handed.</exception>
        public Reliance RelianceOn(RowsAnswered handed, Redraws draws)
        {
            var asked = Asked.Of(prepared, handed);
            var base0 = asked.Answer(asked.Cells);
            var random = new Random(draws.Seed);
            var columns = new List<ColumnReliance>();

            foreach (var column in asked.Perturbable)
            {
                var at = asked.IndexOf(column);
                var moves = new double[draws.Repeats];

                for (var repeat = 0; repeat < draws.Repeats; repeat++)
                {
                    var order = Shuffled(asked.Cells.Length, random);
                    var shuffled = asked.Answer([.. asked.Cells.Select((row, place) => Put(row, at, asked.Cells[order[place]][at]))]);

                    moves[repeat] = Enumerable.Range(0, base0.Count).Average(row => Enumerable.Range(0, base0[row].Length).Average(answer => Math.Abs(shuffled[row][answer] - base0[row][answer])));
                }

                columns.Add(new ColumnReliance(column, moves.Average(), Spread(moves)));
            }

            return new Reliance(asked.AnswerNames, draws.Seed, draws.Repeats, [.. columns.OrderByDescending(column => column.Mean)]);
        }

        /// <summary>What the answers do as one column moves through its range.</summary>
        /// <param name="handed">The rows, and how the model answers them.</param>
        /// <param name="moving">The column, and at how many places a number is moved to.</param>
        /// <returns>The mean answer at each place.</returns>
        /// <exception cref="ArgumentException">The rows lack the column; it is an answer, a moment or an id; or it holds no number to move through.</exception>
        /// <exception cref="InvalidOperationException">The model answered other rows than it was handed.</exception>
        public Dependence DependenceOn(RowsAnswered handed, Moving moving)
        {
            var asked = Asked.Of(prepared, handed);
            var column = moving.Column;
            var at = asked.IndexOf(column);

            if (at < 0 || !asked.Perturbable.Contains(column, StringComparer.Ordinal))
            {
                throw new ArgumentException(
                    $"'{column}' is not a column the answers can be asked of: the rows hold {string.Join(", ", asked.Perturbable.Select(name => $"'{name}'"))}, and the answer and the id are none of them.",
                    nameof(moving));
            }

            var kind = prepared.Declaration.Steps.OfType<DeclareStep>().SelectMany(step => step.Columns).FirstOrDefault(each => each.Name == column)?.Kind;
            var places = kind switch
            {
                ColumnKind.Category => new Places(Categories(prepared, column) ?? [.. asked.Cells.Select(row => row[at]).Where(cell => cell is not null).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)!], null),
                ColumnKind.Number or ColumnKind.Integer => Grid(asked, moving, kind == ColumnKind.Integer),
                _ => throw new ArgumentException($"'{column}' holds {kind?.ToString().ToLowerInvariant() ?? "something the pipeline does not declare"}, which has no range to move through: a number moves from its smallest to its largest, a category through its values.", nameof(moving)),
            };

            var values = places.Values;
            var every = asked.Answer([.. values.SelectMany(value => asked.Cells.Select(row => Put(row, at, value)))]);
            var n = asked.Cells.Length;
            double[][] means =
            [
                .. values.Select((_, place) => Enumerable.Range(0, asked.AnswerNames.Count).Select(answer => Enumerable.Range(0, n).Average(row => every[(place * n) + row][answer])).ToArray()),
            ];

            return new Dependence(column, values, places.Points, means, asked.AnswerNames);
        }

        /// <summary>What one row's answer is made of.</summary>
        /// <param name="handed">The rows, and how the model answers them.</param>
        /// <param name="row">Which of the rows handed in, from nought.</param>
        /// <param name="draws">The seed the samples are worked out from and how many are drawn.</param>
        /// <returns>What each column adds to the row's answer.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The row is not one of those handed in.</exception>
        /// <exception cref="InvalidOperationException">The model answered other rows than it was handed.</exception>
        public RowMakeUp MakeUpOf(RowsAnswered handed, int row, Redraws draws)
        {
            var asked = Asked.Of(prepared, handed);

            if (row < 0 || row >= asked.Cells.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(row), row, $"There are {asked.Cells.Length} rows, counted from nought.");
            }

            var random = new Random(draws.Seed);
            var columns = asked.Perturbable.Select(asked.IndexOf).ToArray();
            var steps = columns.Length;
            var width = asked.AnswerNames.Count;
            var batch = new List<string?[]>();
            var orders = new int[draws.Repeats][];

            for (var sample = 0; sample < draws.Repeats; sample++)
            {
                var background = asked.Cells[random.Next(asked.Cells.Length)];
                var order = Shuffled(steps, random);
                var current = background;

                orders[sample] = order;
                batch.Add(current);

                foreach (var step in order)
                {
                    current = Put(current, columns[step], asked.Cells[row][columns[step]]);
                    batch.Add(current);
                }
            }

            var said = asked.Answer([.. batch]);
            var contributions = new double[steps][];

            for (var step = 0; step < steps; step++)
            {
                contributions[step] = new double[width];
            }

            var backgrounds = new double[width];
            var answers = new double[width];

            for (var sample = 0; sample < draws.Repeats; sample++)
            {
                var start = sample * (steps + 1);

                for (var answer = 0; answer < width; answer++)
                {
                    backgrounds[answer] += said[start][answer] / draws.Repeats;
                    answers[answer] = said[start + steps][answer];

                    for (var place = 0; place < steps; place++)
                    {
                        contributions[orders[sample][place]][answer] += (said[start + place + 1][answer] - said[start + place][answer]) / draws.Repeats;
                    }
                }
            }

            var parts = asked.Perturbable.Select((name, step) => new ColumnContribution(name, contributions[step])).ToArray();

            return new RowMakeUp(
                row,
                draws.Repeats,
                asked.AnswerNames,
                answers,
                backgrounds,
                [.. parts.OrderByDescending(part => Math.Abs(part.Contributions[0]))]);
        }
    }

    // The rows handed in as text, the columns that can be perturbed, and the model's answers to some of them.
    private sealed class Asked
    {
        private readonly RowsAnswered _handed;
        private readonly string[] _columns;

        private Asked(RowsAnswered handed, string?[][] cells, string[] perturbable, IReadOnlyList<string> answerNames)
        {
            _handed = handed;
            _columns = [.. handed.Rows.ColumnNames];
            Cells = cells;
            Perturbable = perturbable;
            AnswerNames = answerNames;
        }

        public string?[][] Cells { get; }

        public string[] Perturbable { get; }

        public IReadOnlyList<string> AnswerNames { get; }

        public static Asked Of(PreparedData prepared, RowsAnswered handed)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(handed.Rows);
            ArgumentNullException.ThrowIfNull(handed.Answering);

            var output = prepared.Declaration.Output ?? throw new ArgumentException(
                "This pipeline names no answer, so there is no answer to ask why of: declare its output.", nameof(prepared));
            string[] columns = [.. handed.Rows.ColumnNames];
            string?[][] cells = [.. handed.Rows.Rows.Select(row => row.ToArray())];

            if (cells.Length == 0)
            {
                throw new ArgumentException("There are no rows to ask why of: hand in some of the rows the model answers.", nameof(handed));
            }

            var id = prepared.Declaration.Steps.OfType<DeclareStep>().Select(step => step.IdColumn).FirstOrDefault(name => name is not null);
            string[] perturbable = [.. columns.Where(name => !output.Answers.Contains(name, StringComparer.Ordinal) && name != id)];

            return new Asked(handed, cells, perturbable, output.Answers);
        }

        public int IndexOf(string column) => Array.IndexOf(_columns, column);

        // The model's answers to some rows, each as many as it was handed.
        public IReadOnlyList<double[]> Answer(string?[][] rows)
        {
            var said = _handed.Answering(new InMemoryRowSource(_columns, rows.Select(row => (IReadOnlyList<string?>)row)));

            if (said.Count != rows.Length || said.Any(row => row.Length != AnswerNames.Count))
            {
                throw new InvalidOperationException(
                    $"The model answered {said.Count} rows of {rows.Length} it was handed, or with another number of answers than the {AnswerNames.Count} the pipeline names: it is asked of exactly the rows it is given.");
            }

            return said;
        }
    }

    // A row with one cell put over another.
    private static string?[] Put(string?[] row, int at, string? value)
    {
        var copy = (string?[])row.Clone();

        copy[at] = value;

        return copy;
    }

    // The places 0..count-1 in a random order: a Fisher–Yates shuffle from the caller's own stream.
    private static int[] Shuffled(int count, Random random)
    {
        var order = Enumerable.Range(0, count).ToArray();

        for (var at = count - 1; at > 0; at--)
        {
            var other = random.Next(at + 1);

            (order[at], order[other]) = (order[other], order[at]);
        }

        return order;
    }

    private static double Spread(double[] values)
    {
        if (values.Length < 2)
        {
            return 0;
        }

        var mean = values.Average();

        return Math.Sqrt(values.Sum(value => (value - mean) * (value - mean)) / (values.Length - 1));
    }

    // Every value the training rows held of a category column, as the encoders that learned them say; nothing when none did.
    private static string[]? Categories(PreparedData prepared, string column)
    {
        for (var at = 0; at < prepared.Declaration.Steps.Count; at++)
        {
            if (prepared.Declaration.Steps[at] is IEncodesCategories encoder
                && prepared.Fitted.TryGetValue(at, out var fitted)
                && encoder.CategoriesIn(fitted).TryGetValue(column, out var categories))
            {
                return [.. categories];
            }
        }

        return null;
    }

    // The places a column is moved to, as the rows write them, and as numbers when they are numbers.
    private readonly record struct Places(string[] Values, double[]? Points);

    // Where a number is moved to: evenly from the smallest to the largest the rows hold, whole numbers for an integer.
    private static Places Grid(Asked asked, Moving moving, bool whole)
    {
        var at = asked.IndexOf(moving.Column);
        double[] held =
        [
            .. asked.Cells.Select(row => row[at]).Where(cell => !string.IsNullOrWhiteSpace(cell)).Select(cell => double.Parse(cell!, NumberStyles.Float, CultureInfo.InvariantCulture)),
        ];

        if (held.Length == 0)
        {
            throw new ArgumentException($"'{moving.Column}' holds no number in the rows handed in, so there is no range to move through.", nameof(moving));
        }

        var lowest = held.Min();
        var highest = held.Max();
        double[] grid = [.. Enumerable.Range(0, moving.Points).Select(place => lowest + ((highest - lowest) * place / (moving.Points - 1)))];

        if (whole)
        {
            grid = [.. grid.Select(point => Math.Round(point)).Distinct()];
        }

        return new Places([.. grid.Select(point => point.ToString("R", CultureInfo.InvariantCulture))], grid);
    }
}

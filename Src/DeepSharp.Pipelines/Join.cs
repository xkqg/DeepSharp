// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>What becomes of a row of the left file whose key the right file does not hold.</summary>
/// <remarks>Said in every join, because no answer to it is safe to assume: there is no default.</remarks>
public enum Unmatched
{
    /// <summary>The run stops, naming the row and its key.</summary>
    Refuse,

    /// <summary>The row is left out of the rows, and the fit counts how many were.</summary>
    Drop,
}

/// <summary>
/// Read two comma-separated files as one source: each row of the left file, beside the one row of the right file its key
/// names.
/// </summary>
/// <remarks>
/// <para>
/// A join made by hand before the pipeline starts is a step nobody can see in the pipeline's file: which files, on which
/// columns, and what became of a row without a partner. Declared here, the file says all of it, and the rows a pipeline is
/// fitted on are made the same way every time it runs.
/// </para>
/// <para>
/// Each file is read as <c>read.csv</c> reads one: its first line names the columns, its bytes are text in the encoding
/// their byte-order mark names, UTF-8 when it names none, and a relative path is read from the pipeline's folder. The rows
/// are the left file's, in its order, so a row is read at its place among them — its place in the left file, unless rows
/// above it were left out for want of a partner — and a refusal names it there. Each left
/// row takes the one row of the right file whose key is its key; a key the right file holds twice is refused, naming it
/// and its rows, since one left row cannot have two partners. A left key may stand on many rows, and each of them takes
/// the same partner; a right row no left row names is not used.
/// </para>
/// <para>
/// The keys are the columns <see cref="On"/> names, in both files under the same names. A key is the exact text of its
/// cells with the spaces around it taken off — as a row's key reads a cell — compared character by character: no case is
/// folded and no number is read, so <c>f7</c> is not <c>F7</c> and <c>07</c> is not <c>7</c>. A key cell that is empty, or
/// nothing but spaces, is a gap, and a gap matches nothing. The joined rows have the left file's columns, then the right
/// file's but the keys; a column both files have beside the keys is refused by name rather than renamed, since only the
/// person who wrote the files knows which of the two it is. What becomes of a left row without a partner is said, never
/// assumed (<see cref="Unmatched"/>): refused, or left out and counted where the split writes how many rows each part
/// holds, as <c>rows.unmatched</c>.
/// </para>
/// <para>
/// A joined row is known, as every row is, by every cell it holds, so a cell of the right file is as much part of its key
/// as a cell of the left. The join reads the rows the pipeline is fitted on. Rows served to the fitted pipeline later are
/// handed in already joined — the columns of both files in them — since rows handed in take the place of the source, and
/// the join is not made again for them. The courses a notebook offers start from one file, and no course has a place for
/// a join.
/// </para>
/// </remarks>
public sealed record ReadJoinStep : IPipelineStep<ReadJoinStep>, IOpensRows, IReadsFiles, IDescribesColumns
{
    private const string Csv = "csv";

    private static readonly FilePathParameter PathKey = new(
        "path", "Where the comma-separated file will be, when the pipeline runs.", "data.csv");

    private static readonly PartsParameter LeftKey = new(
        "left", "The file whose rows the joined rows are: each of its rows, in its order, beside its partner.", [PartOf("planned.csv")], [CsvFile("planned.csv")])
    {
        Single = true,
    };

    private static readonly PartsParameter RightKey = new(
        "right", "The file each left row takes its partner from: the one row whose key is the left row's key.", [PartOf("arrived.csv")], [CsvFile("arrived.csv")])
    {
        Single = true,
    };

    private static readonly ColumnsParameter OnKey = new(
        "on",
        "The columns a left row and its partner share, named alike in both files: each compared as the exact text of its cells, the spaces around it taken off.",
        ["key"],
        ColumnKinds.Any);

    private static readonly OneOfParameter<Unmatched> UnmatchedKey = new(
        "unmatched",
        "What becomes of a left row whose key the right file does not hold: refuse stops the run, naming the row; drop leaves it out and counts it in the fit.",
        Unmatched.Refuse);

    /// <summary>Declares that the rows come from two comma-separated files, joined on the columns they share.</summary>
    /// <param name="left">Where the file whose rows the joined rows are will be, when the pipeline runs.</param>
    /// <param name="right">Where the file each left row takes its partner from will be.</param>
    /// <param name="on">The columns a left row and its partner share, named alike in both files.</param>
    /// <param name="unmatched">What becomes of a left row whose key the right file does not hold.</param>
    /// <exception cref="ArgumentException">A path is empty, no column is named, or one is named twice.</exception>
    /// <exception cref="ArgumentOutOfRangeException">What becomes of an unmatched row is none of the words: a number cast to them.</exception>
    public ReadJoinStep(string left, string right, IEnumerable<string> on, Unmatched unmatched)
    {
        ArgumentNullException.ThrowIfNull(on);

        Left = PathKey.Require(left);
        Right = PathKey.Require(right);
        On = OnKey.Require([.. on]);
        Unmatched = UnmatchedKey.Require(unmatched);
    }

    /// <summary>Where the file whose rows the joined rows are will be, when the pipeline runs.</summary>
    public string Left { get; }

    /// <summary>Where the file each left row takes its partner from will be, when the pipeline runs.</summary>
    public string Right { get; }

    /// <summary>The columns a left row and its partner share, named alike in both files, in the order they were named.</summary>
    public IReadOnlyList<string> On { get; }

    /// <summary>What becomes of a left row whose key the right file does not hold.</summary>
    public Unmatched Unmatched { get; }

    /// <inheritdoc />
    /// <remarks>The left file, then the right.</remarks>
    public IReadOnlyList<string> Paths => [Left, Right];

    /// <inheritdoc />
    public static string Name => "read.join";

    /// <inheritdoc />
    public static string Purpose =>
        "Reads two comma-separated files as one source: each row of the left file beside the one row of the right file its key names. Rows served later arrive already joined.";

    /// <inheritdoc />
    /// <remarks>New in the eighth version of the pipeline file.</remarks>
    public static int Since => 8;

    /// <inheritdoc />
    public static StepParameters<ReadJoinStep> Parameters { get; } = new StepParameters<ReadJoinStep>()
        .With(LeftKey, step => [PartOf(step.Left)])
        .With(RightKey, step => [PartOf(step.Right)])
        .With(OnKey, step => step.On)
        .With(UnmatchedKey, step => step.Unmatched);

    /// <inheritdoc />
    public string Verb => Name;

    /// <summary>None: the columns it is made on are columns of its files, not of the rows the schema reads.</summary>
    /// <remarks>
    /// A column a step reads is a column of the rows the schema reads, and nothing above the schema reads one; a key named
    /// as one would have every step that follows those columns look for it where there are none yet.
    /// </remarks>
    IReadOnlyList<ColumnRead> IPipelineStep.ColumnsRead => [];

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    /// <remarks>
    /// Each relative path is read from the pipeline's folder; the paths stay as they were written. The files' bytes are read
    /// once each and parsed as <see cref="Open(IReadOnlyList{byte[]}, IReadOnlyList{string})"/> parses them.
    /// </remarks>
    /// <exception cref="FormatException">
    /// A file has no header or a row of the wrong length; a file lacks a column the join is made on; both have another
    /// column of one name; the right file holds a key twice; or a left row has no partner, and such a row is refused.
    /// </exception>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        string[] files = [folder.Resolve(Left), folder.Resolve(Right)];

        return Open([.. files.Select(File.ReadAllBytes)], files);
    }

    /// <inheritdoc />
    /// <exception cref="FormatException">
    /// A file has no header or a row of the wrong length; a file lacks a column the join is made on; both have another
    /// column of one name; the right file holds a key twice; or a left row has no partner, and such a row is refused.
    /// </exception>
    public IRowSource Open(IReadOnlyList<byte[]> bytes, IReadOnlyList<string> files)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentNullException.ThrowIfNull(files);

        if (bytes.Count != 2 || files.Count != 2)
        {
            throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"A join reads two files, the left and the right, and is handed the bytes of {bytes.Count} and the names of {files.Count}."),
                nameof(bytes));
        }

        var left = new NamedRows(CsvRowSource.FromText(bytes[0].AsText(), files[0]), files[0]);
        var right = new NamedRows(CsvRowSource.FromText(bytes[1].AsText(), files[1]), files[1]);
        var header = Header(left.Rows.ColumnNames, left.File, right.Rows.ColumnNames, right.File);

        return Joined(header, left, right.File, Partners(header, right));
    }

    /// <inheritdoc />
    /// <remarks>
    /// The two files' first lines alone, read as the whole files are, so a large file costs that line: the left file's
    /// columns, then the right file's but the keys.
    /// </remarks>
    /// <exception cref="FormatException">A file has no header, lacks a column the join is made on, or both have another column of one name.</exception>
    public IReadOnlyList<string> ColumnNamesIn(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        var left = folder.Resolve(Left);
        var right = folder.Resolve(Right);

        return Header(CsvRowSource.HeaderOf(left), left, CsvRowSource.HeaderOf(right), right).Names;
    }

    /// <inheritdoc />
    public bool Equals(ReadJoinStep? other) =>
        other is not null && Left == other.Left && Right == other.Right && Unmatched == other.Unmatched && On.SequenceEqual(other.On, StringComparer.Ordinal);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Left);
        hash.Add(Right);
        hash.Add(Unmatched);

        foreach (var key in On)
        {
            hash.Add(key);
        }

        return hash.ToHashCode();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A file, the keys or what becomes of an unmatched row is missing or not what it should be.</exception>
    public static ReadJoinStep ReadFrom(JsonElement element) =>
        new(LeftKey.Read(element)[0].Text(PathKey.Key), RightKey.Read(element)[0].Text(PathKey.Key), OnKey.Read(element), UnmatchedKey.Read(element));

    // One file as the step writes it: a comma-separated file at a path.
    private static PartDeclaration PartOf(string path) => new(Csv, [new PartSetting(PathKey.Key, PartValue.Of(path))]);

    // The one kind a file of a join is, read as read.csv reads one; a new part of it starts at the example its side gives, so
    // a part started afresh is the part the step's own example holds.
    private static PartKind CsvFile(string example) => new(
        Csv,
        "A comma-separated file, read as read.csv reads one: its first line names the columns.",
        [new FilePathParameter(PathKey.Key, PathKey.Description, example)]);

    // The joined rows' columns, and where the keys stand in each file and which of the right file's columns come along.
    private JoinedHeader Header(IReadOnlyList<string> left, string leftFile, IReadOnlyList<string> right, string rightFile)
    {
        var leftKeys = KeysIn(left, leftFile);
        var rightKeys = KeysIn(right, rightFile);
        int[] kept = [.. Enumerable.Range(0, right.Count).Where(at => !On.Contains(right[at], StringComparer.Ordinal))];

        // A column both files have is two columns under one name, and which of them a model should see is not a guess the
        // join makes for whoever wrote the files.
        if (kept.Select(at => right[at]).FirstOrDefault(name => left.Contains(name, StringComparer.Ordinal)) is { } both)
        {
            throw new FormatException(
                $"'{both}' is a column of both {leftFile} and {rightFile}, and only the columns the join is made on may be: rename it in one of them.");
        }

        return new JoinedHeader([.. left, .. kept.Select(at => right[at])], leftKeys, rightKeys, kept);
    }

    private int[] KeysIn(IReadOnlyList<string> header, string file)
    {
        var places = new int[On.Count];

        for (var key = 0; key < On.Count; key++)
        {
            places[key] = IndexOf(header, On[key]);

            if (places[key] < 0)
            {
                throw new FormatException(
                    $"'{On[key]}' is a column the join is made on, and {file} has no column of that name. It has: {string.Join(", ", header)}.");
            }
        }

        return places;
    }

    private static int IndexOf(IReadOnlyList<string> names, string name)
    {
        for (var at = 0; at < names.Count; at++)
        {
            if (string.Equals(names[at], name, StringComparison.Ordinal))
            {
                return at;
            }
        }

        return -1;
    }

    // Every key of the right file and its row, refused whole when one stands twice: before any left row is matched, so a
    // row of the left file without a partner never hides a key that has two.
    private Dictionary<string, IReadOnlyList<string?>> Partners(JoinedHeader header, NamedRows right)
    {
        var partners = new Dictionary<string, IReadOnlyList<string?>>(StringComparer.Ordinal);
        var rowsOf = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        string[]? twice = null;
        var row = 0;

        foreach (var cells in right.Rows.Rows)
        {
            row++;

            if (KeyOf(cells, header.RightKeys) is not { } key)
            {
                continue;
            }

            var written = Written(key);

            if (rowsOf.TryGetValue(written, out var rows))
            {
                twice ??= key;
                rows.Add(row);

                continue;
            }

            rowsOf[written] = [row];
            partners[written] = cells;
        }

        if (twice is not null)
        {
            var rows = rowsOf[Written(twice)];

            throw new FormatException(
                $"{Shown(twice)} stands on rows {Listed(rows)} of {right.File}: each left row takes one partner, so a key stands in the right file once.");
        }

        return partners;
    }

    // Each left row, in its order, beside its partner's columns but the keys; a row without one refused or left out and counted.
    private JoinedRows Joined(JoinedHeader header, NamedRows left, string rightFile, Dictionary<string, IReadOnlyList<string?>> partners)
    {
        var joined = new List<IReadOnlyList<string?>>();
        var leftOut = 0;
        var row = 0;

        foreach (var cells in left.Rows.Rows)
        {
            row++;

            var key = KeyOf(cells, header.LeftKeys);

            if (key is not null && partners.TryGetValue(Written(key), out var partner))
            {
                joined.Add([.. cells, .. header.RightKept.Select(at => partner[at])]);

                continue;
            }

            if (Unmatched == Unmatched.Drop)
            {
                leftOut++;

                continue;
            }

            throw new FormatException(key is null
                ? string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {row} of {left.File} has no key: its '{On[Array.FindIndex(header.LeftKeys, at => IsAGap(cells[at]))]}' is a gap, and a gap matches nothing. Fill it in that file, or say unmatched: drop to leave such rows out, counted in the fit.")
                : string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {row} of {left.File} has the key {Shown(key)}, and {rightFile} has no row with it: keys are compared as their text with the spaces around it taken off, character by character. Say unmatched: drop to leave such rows out, counted in the fit."));
        }

        return new JoinedRows(header.Names, joined, Unmatched == Unmatched.Drop ? leftOut : null);
    }

    // The key of a row, as the text of each key cell with the spaces around it taken off; nothing when one of them is a gap.
    private static string[]? KeyOf(IReadOnlyList<string?> cells, int[] keys)
    {
        var key = new string[keys.Length];

        for (var at = 0; at < keys.Length; at++)
        {
            if (IsAGap(cells[keys[at]]))
            {
                return null;
            }

            key[at] = cells[keys[at]]!.Trim();
        }

        return key;
    }

    private static bool IsAGap(string? cell) => string.IsNullOrWhiteSpace(cell);

    // A key of several cells as one text no two keys share: each cell preceded by its length.
    private static string Written(string[] key) =>
        string.Concat(key.Select(cell => string.Create(CultureInfo.InvariantCulture, $"{cell.Length}:{cell}")));

    // A key as a refusal names it: each column the join is made on, with its cell.
    private string Shown(string[] key) => string.Join(", ", key.Select((cell, at) => $"{On[at]} '{cell}'"));

    private static string Listed(List<int> rows) =>
        string.Create(CultureInfo.InvariantCulture, $"{string.Join(", ", rows.Take(rows.Count - 1))} and {rows[^1]}");
}

/// <summary>The columns of a join's rows, and where its keys and the right file's columns that come along stand.</summary>
/// <param name="Names">The left file's columns, then the right file's but the keys.</param>
/// <param name="LeftKeys">Where each key stands in the left file.</param>
/// <param name="RightKeys">Where each key stands in the right file.</param>
/// <param name="RightKept">Where each of the right file's columns that come along stands in it.</param>
internal readonly record struct JoinedHeader(IReadOnlyList<string> Names, int[] LeftKeys, int[] RightKeys, int[] RightKept);

/// <summary>The rows of one file a join reads, and what a refusal names the file as.</summary>
/// <param name="Rows">The file's rows.</param>
/// <param name="File">What a refusal names it as: the path it was read from.</param>
internal readonly record struct NamedRows(CsvRowSource Rows, string File);

/// <summary>
/// The rows a join made: each row of the left file that has a partner, in its order, beside that partner's cells.
/// </summary>
/// <param name="names">The joined rows' columns.</param>
/// <param name="rows">The joined rows.</param>
/// <param name="leftOut">How many left rows had no partner and were left out; nothing for a join that refuses them.</param>
/// <remarks>
/// What a join leaves out is said where a fit says how many rows each part holds, so the source carries the count to the
/// split rather than writing anything of its own.
/// </remarks>
internal sealed class JoinedRows(IReadOnlyList<string> names, IReadOnlyList<IReadOnlyList<string?>> rows, int? leftOut) : IRowSource
{
    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames => names;

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows => rows;

    /// <summary>How many left rows had no partner and were left out; nothing for a join that refuses them.</summary>
    public int? LeftOut => leftOut;
}

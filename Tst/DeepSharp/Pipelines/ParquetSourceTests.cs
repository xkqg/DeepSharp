// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Parquet;
using Parquet.Schema;
using Parquet.Serialization;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A Parquet file types its columns, so it says what each holds as a database does, and the proposal of kinds takes what
/// it says over what the cells look like. Its values reach the pipeline as the text it reads, by the one rule every typed
/// source hands a value over in, so a number is spelled one way whichever of them held it.
/// </summary>
public sealed class ParquetSourceTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-parquet-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public void ThePassengerList_SaysWhatEachOfItsColumnsHolds_AndTheProposalTakesWhatItSays()
    {
        var file = Pdd.Create().ReadParquet(Repository.Fixture("titanic.parquet")).ProposedKinds();
        var csv = Pdd.Create().ReadCsv(Repository.Data("titanic.csv")).ProposedKinds();

        Assert.Equal(csv.Columns.Select(column => column.Name), file.Columns.Select(column => column.Name));
        Assert.All(file.Columns, column => Assert.NotNull(column.Stated));
        Assert.Equal(ColumnKind.Integer, file["survived"].Stated);
        Assert.Equal(ColumnKind.Integer, file["survived"].Kind);
        Assert.Equal(ColumnKind.Boolean, file["adult_male"].Stated);
        Assert.Equal(ColumnKind.Boolean, file["adult_male"].Kind);

        // Words few enough to be groups are proposed as a category, as a database's are.
        Assert.Equal(ColumnKind.Text, file["sex"].Stated);
        Assert.Equal(ColumnKind.Category, file["sex"].Kind);

        // The file holds the ages as the words the comma-separated one spells them in, and says so: what a source states
        // outranks cells that read as numbers.
        Assert.Equal(ColumnKind.Text, file["age"].Stated);
        Assert.Equal(ColumnKind.Text, file["age"].Kind);
        Assert.Equal(ColumnKind.Number, csv["age"].Kind);
        Assert.Null(csv["age"].Stated);
    }

    [Fact]
    public async Task EveryKindAParquetFileHolds_IsStated_HandedOverAsTheOneTextOfATypedValue_AndBindsAsWhatItIs()
    {
        var path = Path.Join(_folder, "kinds.parquet");
        var code = new DataField<string>("code");
        var stock = new DataField<long?>("stock");
        var price = new DataField<double?>("price");
        var flag = new DataField<bool?>("flag");
        var when = new DataField<DateTime?>("when");
        var day = new DataField<DateOnly?>("day");
        var share = new DataField<decimal?>("share");
        var count = new DataField<int>("count");
        var small = new DataField<float?>("small");
        var tiny = new DataField<sbyte?>("tiny");
        var octet = new DataField<byte?>("octet");
        var shortish = new DataField<short?>("shortish");
        var unsignedShort = new DataField<ushort?>("unsignedShort");
        var unsigned = new DataField<uint?>("unsigned");
        var unsignedLong = new DataField<ulong?>("unsignedLong");
        var id = new DataField<Guid?>("id");
        var identity = new Guid("6f9619ff-8b86-d011-b42d-00cf4fc964ff");

        using (var file = File.Create(path))
        {
            await using var writer = await ParquetWriter.CreateAsync(
                new ParquetSchema(code, stock, price, flag, when, day, share, count, small, tiny, octet, shortish, unsignedShort, unsigned, unsignedLong, id),
                file,
                cancellationToken: TestContext.Current.CancellationToken);
            using var group = writer.CreateRowGroup();

            await group.WriteAsync(code, ["007", string.Empty, null!]);
            await group.WriteAsync<long>(stock, new long?[] { 3, null, 5 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<double>(price, new double?[] { 2.0, null, 133.1284878 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<bool>(flag, new bool?[] { true, null, false }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<DateTime>(when, new DateTime?[] { new(2015, 2, 18, 9, 30, 0, DateTimeKind.Utc), null, new(2015, 2, 19, 0, 0, 0, DateTimeKind.Utc) }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<DateOnly>(day, new DateOnly?[] { new(2015, 2, 18), null, new(2016, 1, 1) }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<decimal>(share, new decimal?[] { 0.42m, null, 22m }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<int>(count, new[] { 1, 2, 3 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<float>(small, new float?[] { 0.1f, null, 1.5f }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<sbyte>(tiny, new sbyte?[] { -1, null, 1 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<byte>(octet, new byte?[] { 255, null, 0 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<short>(shortish, new short?[] { -300, null, 300 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<ushort>(unsignedShort, new ushort?[] { 65535, null, 1 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<uint>(unsigned, new uint?[] { 4000000000, null, 1 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<ulong>(unsignedLong, new ulong?[] { 18000000000000000000, null, 1 }, cancellationToken: TestContext.Current.CancellationToken);
            await group.WriteAsync<Guid>(id, new Guid?[] { identity, null, Guid.Empty }, cancellationToken: TestContext.Current.CancellationToken);
        }

        var pipeline = Pdd.Create().ReadParquet(path);
        var proposal = pipeline.ProposedKinds();

        Assert.Equal(
            [ColumnKind.Text, ColumnKind.Integer, ColumnKind.Number, ColumnKind.Boolean, ColumnKind.Timestamp, ColumnKind.Timestamp, ColumnKind.Number,
             ColumnKind.Integer, ColumnKind.Number, ColumnKind.Integer, ColumnKind.Integer, ColumnKind.Integer, ColumnKind.Integer, ColumnKind.Integer,
             ColumnKind.Integer, null],
            proposal.Columns.Select(column => column.Stated));

        // The code written in digits stays words; a price of 2.0 is handed over as 2, as a database's is, and is still the
        // number the file says it is; a date is a day, and a moment is read in universal time.
        var rows = new ReadParquetStep(path).Open(SourceFolder.WorkingDirectory).Rows.ToArray();

        Assert.Equal(
            ["007", "3", "2", "True", "2015-02-18T09:30:00.0000000", "2015-02-18", "0.42", "1", "0.1", "-1", "255", "-300", "65535", "4000000000", "18000000000000000000", "6f9619ff-8b86-d011-b42d-00cf4fc964ff"],
            rows[0]);
        Assert.Equal([string.Empty, null, null, null, null, null, null, "2", null, null, null, null, null, null, null, null], rows[1]);
        Assert.Equal("133.1284878", rows[2][2]);

        var bound = pipeline
            .Declare(schema => schema
                .Text("code", "id")
                .Integer("stock", "count", "tiny", "octet", "shortish", "unsignedShort", "unsigned")
                .Number("price", "share", "small", "unsignedLong")
                .Boolean("flag")
                .Timestamp("when", "day"))
            .Build()
            .Prepare();

        Assert.Equal(new DateTime(2015, 2, 18, 9, 30, 0, DateTimeKind.Utc), ((Column<DateTime>)bound["when"])[0]);
        Assert.Equal(new DateTime(2016, 1, 1, 0, 0, 0, DateTimeKind.Utc), ((Column<DateTime>)bound["day"])[2]);
        Assert.Equal(133.1284878, ((Column<double>)bound["price"])[2]);
        Assert.Null(((Column<long>)bound["stock"])[1]);
        Assert.False(((Column<bool>)bound["flag"])[2]);
        Assert.Equal("007", ((TextColumn)bound["code"])[0]);
    }

    [Fact]
    public async Task EveryRowGroup_IsRead_InTheOrderTheFileHoldsThem()
    {
        var path = Path.Join(_folder, "groups.parquet");
        var number = new DataField<int>("number");

        using (var file = File.Create(path))
        {
            await using var writer = await ParquetWriter.CreateAsync(new ParquetSchema(number), file, cancellationToken: TestContext.Current.CancellationToken);

            foreach (var numbers in new[] { new[] { 1, 2 }, [3], [4, 5, 6] })
            {
                using var group = writer.CreateRowGroup();
                await group.WriteAsync<int>(number, numbers, cancellationToken: TestContext.Current.CancellationToken);
            }
        }

        var rows = new ReadParquetStep(path).Open(SourceFolder.WorkingDirectory).Rows;

        Assert.Equal(["1", "2", "3", "4", "5", "6"], rows.Select(row => row[0]));
    }

    [Fact]
    public async Task AColumnOfListsOrOfGroups_IsRefusedByName_ForACellHoldsOneValue()
    {
        var lists = Path.Join(_folder, "lists.parquet");
        var groups = Path.Join(_folder, "groups.parquet");

        await ParquetSerializer.SerializeAsync([new Tagged { Id = 1, Tags = [7, 8] }], lists, cancellationToken: TestContext.Current.CancellationToken);
        await ParquetSerializer.SerializeAsync([new Addressed { Id = 1, Address = new Address { Street = "Main Street" } }], groups, cancellationToken: TestContext.Current.CancellationToken);

        var list = Assert.Throws<FormatException>(() => Pdd.Create().ReadParquet(lists).ProposedKinds());
        var fields = Assert.Throws<FormatException>(() => Pdd.Create().ReadParquet(groups).ProposedKinds());

        Assert.StartsWith($"{lists}: 'Tags' holds a list rather than one value to a row", list.Message, StringComparison.Ordinal);
        Assert.StartsWith($"{groups}: 'Address' holds a group of fields rather than one value to a row", fields.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AColumnOfRawBytes_IsRefusedByName()
    {
        var path = Path.Join(_folder, "bytes.parquet");
        var blob = new DataField<byte[]>("blob");

        using (var file = File.Create(path))
        {
            await using var writer = await ParquetWriter.CreateAsync(new ParquetSchema(blob), file, cancellationToken: TestContext.Current.CancellationToken);
            using var group = writer.CreateRowGroup();
            await group.WriteAsync(blob, [new byte[] { 1, 2 }]);
        }

        var refused = Assert.Throws<FormatException>(() => Pdd.Create().ReadParquet(path).ProposedKinds());

        Assert.StartsWith($"{path}: 'blob' holds raw bytes", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AParquetFilesColumns_AreNamedFromItsSchema_WithoutReadingItsRows()
    {
        // The passenger list with every byte between its first four and its footer zeroed: its schema is whole, and not one
        // of its rows can be read. Its columns are named all the same, so a take-over of a large file costs its footer.
        var path = Path.Join(_folder, "titanic.parquet");
        var bytes = File.ReadAllBytes(Repository.Fixture("titanic.parquet"));
        var footer = BitConverter.ToInt32(bytes, bytes.Length - 8);

        Array.Clear(bytes, 4, bytes.Length - 8 - footer - 4);
        File.WriteAllBytes(path, bytes);

        var step = new ReadParquetStep(path);
        var csv = new ReadCsvStep(Repository.Data("titanic.csv")).ColumnNamesIn(SourceFolder.WorkingDirectory);

        Assert.Equal(csv, step.ColumnNamesIn(SourceFolder.WorkingDirectory));
        Assert.Equal(15, csv.Count);
        Assert.StartsWith($"{path} cannot be read as a Parquet file", Assert.Throws<FormatException>(() => step.Open(SourceFolder.WorkingDirectory)).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => step.ColumnNamesIn(null!));
    }

    [Fact]
    public void ManyReadersAtOnce_OfTheSameBytes_EachReadTheRowsOneReaderReadsAlone()
    {
        // A pipeline, a notebook and a server can each open the same file at the same moment, on threads of their own: every
        // one of them reads what one reader alone reads, cell for cell.
        var bytes = File.ReadAllBytes(Repository.Fixture("titanic.parquet"));
        var step = new ReadParquetStep("titanic.parquet");
        string?[][] alone = [.. step.Open(bytes, "titanic.parquet").Rows.Select(row => row.ToArray())];
        var differing = new System.Collections.Concurrent.ConcurrentBag<string>();

        Parallel.For(0, 64, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, reader =>
        {
            var rows = step.Open(bytes, "titanic.parquet").Rows.ToArray();

            for (var row = 0; row < alone.Length; row++)
            {
                if (!alone[row].SequenceEqual(rows[row]))
                {
                    differing.Add($"reader {reader}, row {row + 1}: [{string.Join("|", rows[row])}] where alone [{string.Join("|", alone[row])}]");
                }
            }
        });

        Assert.Empty(differing.Take(5));
    }

    [Fact]
    public async Task ManyReadersBesideWritersOfOtherFiles_EachReadTheRowsOneReaderReadsAlone()
    {
        // A process that reads a file on some threads while other Parquet files are written on others: read through buffers
        // rented from the pool the whole process shares, a column of numbers came back with cells marked empty that the file
        // holds and the values below them moved up a row, in half the runs of this test. Every reader reads what one alone does.
        var bytes = File.ReadAllBytes(Repository.Fixture("titanic.parquet"));
        var step = new ReadParquetStep("titanic.parquet");
        string?[][] alone = [.. step.Open(bytes, "titanic.parquet").Rows.Select(row => row.ToArray())];
        var differing = new System.Collections.Concurrent.ConcurrentBag<string>();
        var readers = Enumerable.Range(0, 400).Select(reader => Task.Run(() =>
        {
            var rows = step.Open(bytes, "titanic.parquet").Rows.ToArray();

            for (var row = 0; row < alone.Length; row++)
            {
                if (!alone[row].SequenceEqual(rows[row]))
                {
                    differing.Add($"reader {reader}, row {row + 1}: [{string.Join("|", rows[row])}] where alone [{string.Join("|", alone[row])}]");
                }
            }
        }));
        var writers = Enumerable.Range(0, 400).Select(_ => Task.Run(async () =>
        {
            using var stream = new MemoryStream();

            await ParquetSerializer.SerializeAsync(Enumerable.Range(0, 2000).Select(at => new Tagged { Id = at, Tags = [at, at + 1] }), stream);
            stream.Position = 0;
            await ParquetSerializer.DeserializeAsync<Tagged>(stream);
        }));

        await Task.WhenAll(readers.Concat(writers));

        Assert.Empty(differing.Take(5));
    }

    [Fact]
    public void AFileWhoseRowsCannotBeRead_IsRefusedNamingIt_AsOneWhoseFooterCannotBe()
    {
        // Its footer is whole and its rows are not: the library says so in its own exception, which no door of the
        // pipeline or the notebook knows, so it is refused in the words every unreadable file is.
        var bytes = File.ReadAllBytes(Repository.Fixture("titanic.parquet"));
        var footer = BitConverter.ToInt32(bytes, bytes.Length - 8);

        Array.Clear(bytes, 4, bytes.Length - 8 - footer - 4);

        var refused = Assert.Throws<FormatException>(() => new ReadParquetStep("damaged.parquet").Open(bytes, "damaged.parquet"));

        Assert.StartsWith("damaged.parquet cannot be read as a Parquet file: ", refused.Message, StringComparison.Ordinal);
        Assert.IsType<InvalidDataException>(refused.InnerException);
    }

    [Fact]
    public async Task AColumnThatHoldsNoOneValueToARow_IsRefusedByName_WhenOnlyItsColumnsAreAskedFor()
    {
        var lists = Path.Join(_folder, "lists.parquet");
        var text = Path.Join(_folder, "text.parquet");

        await ParquetSerializer.SerializeAsync([new Tagged { Id = 1, Tags = [7, 8] }], lists, cancellationToken: TestContext.Current.CancellationToken);
        File.Copy(Repository.Data("titanic.csv"), text);

        var list = Assert.Throws<FormatException>(() => new ReadParquetStep(lists).ColumnNamesIn(SourceFolder.WorkingDirectory));
        var notParquet = Assert.Throws<FormatException>(() => new ReadParquetStep(text).ColumnNamesIn(SourceFolder.WorkingDirectory));

        Assert.StartsWith($"{lists}: 'Tags' holds a list rather than one value to a row", list.Message, StringComparison.Ordinal);
        Assert.StartsWith($"{text} cannot be read as a Parquet file", notParquet.Message, StringComparison.Ordinal);
        Assert.Throws<FileNotFoundException>(() => new ReadParquetStep(Path.Join(_folder, "gone.parquet")).ColumnNamesIn(SourceFolder.WorkingDirectory));
    }

    public sealed class Tagged
    {
        public int Id { get; set; }

        public List<int> Tags { get; set; } = [];
    }

    public sealed class Addressed
    {
        public int Id { get; set; }

        public Address Address { get; set; } = new();
    }

    public sealed class Address
    {
        public string Street { get; set; } = string.Empty;
    }
}

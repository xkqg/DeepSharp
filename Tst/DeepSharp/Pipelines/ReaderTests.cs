// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Microsoft.Extensions.DependencyInjection;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The readers of the formats data arrives in beside comma-separated text — Parquet, Excel and JSON — each an extension
/// method in the package that brings what it needs, and each a verb in the pipeline's file that is replayed like the
/// comma-separated one. A row is known by what its file says, so the same cells in any of them are the same rows: the
/// same keys, the same split and the same batches.
/// </summary>
public class ReaderTests
{
    public static TheoryData<string> TitanicInEveryFormat() =>
        new() { "titanic.parquet", "titanic.xlsx", "titanic.xls", "titanic.json" };

    // The reader each file is read by, as a person writes it.
    private static PipelineBuilder Reading(PipelineBuilder pipeline, string path) => Path.GetExtension(path) switch
    {
        ".parquet" => pipeline.ReadParquet(path),
        ".json" => pipeline.ReadJson(path),
        ".csv" => pipeline.ReadCsv(path),
        _ => pipeline.ReadExcel(path),
    };

    // The passenger list as the networks sample declares it, with its words and its true-or-false columns taken as well.
    private static FittingBuilder Declared(PipelineBuilder source) => source
        .Declare(schema => schema
            .Integer("survived", "sibsp", "parch")
            .Category("pclass", "sex", "embarked", "deck")
            .Optional("age", ColumnKind.Number)
            .Number("fare")
            .Boolean("adult_male", "alone"))
        .SplitStratified("survived", train: 0.70, validation: 0.15)
        .FillMissing("age", With.Median)
        .EncodeCategories()
        .Normalise("age", Scale.MidRange)
        .Normalise("fare", Scale.MidRange)
        .Target("survived");

    private static void Same(PreparedData expected, PreparedData actual)
    {
        Assert.Equal(expected.Table.Identities, actual.Table.Identities);
        Assert.Equal(expected.Parts, actual.Parts);

        foreach (var part in new[] { Part.Train, Part.Validation, Part.Test })
        {
            var one = expected.Batch(part);
            var other = actual.Batch(part);

            Assert.NotEmpty(one.Features);
            Assert.Equal(one.FeatureNames, other.FeatureNames);
            Assert.Equal(one.Features, other.Features);
            Assert.Equal(one.AnswerNames, other.AnswerNames);
            Assert.Equal(one.Answers, other.Answers);
            Assert.Equal(one.Keys, other.Keys);
        }
    }

    [Theory]
    [MemberData(nameof(TitanicInEveryFormat))]
    public void TheTitanicPassengerList_InEachFormat_GivesTheKeysTheSplitAndTheBatchesOfItsCsv(string file)
    {
        var csv = Declared(Pdd.Create().ReadCsv(Repository.Data("titanic.csv"))).Build().Run();
        var other = Declared(Reading(Pdd.Create(), Repository.Fixture(file))).Build().Run();

        Assert.Equal(891, other.Table.RowCount);
        Same(csv, other);
    }

    [Theory]
    [MemberData(nameof(TitanicInEveryFormat))]
    public void APipelineFileWrittenWithEachReader_ReadsBackThroughTheCatalog_AndRunsAndServesAsItWasWritten(string file)
    {
        var folder = Directory.CreateTempSubdirectory("deepsharp-readers-").FullName;

        try
        {
            // Written as a pipeline moved together with its data is: the file named from the pipeline's own folder.
            File.Copy(Repository.Fixture(file), Path.Join(folder, file));
            var document = Path.Join(folder, "titanic.pipeline.json");
            var declaration = Declared(Reading(Pdd.Create(), file)).Declaration;
            var run = new Pipeline(declaration, null, SourceFolder.OfDocument(document)).Run();

            File.WriteAllText(document, run.ToJson());

            var text = File.ReadAllText(document);
            var read = PipelineDeclaration.FromJson(text, Shipped.Catalog());

            Assert.Equal(declaration, read);
            Same(run, new Pipeline(read, null, SourceFolder.OfDocument(document)).Run());

            // The fit read back serves the file's rows exactly as the run prepared them.
            var loaded = PreparedData.FromJson(text, Shipped.Catalog());
            var served = loaded.Served(((IOpensRows)loaded.Declaration.Steps[0]).Open(SourceFolder.OfDocument(document)));
            var prepared = new[] { Part.Train, Part.Validation, Part.Test }
                .Select(part => run.Batch(part))
                .SelectMany(batch => batch.Keys!.Zip(batch.Features))
                .ToLookup(row => row.First, row => row.Second);

            Assert.Equal(run.Table.Identities.Select(identity => identity.Key), served.Keys);
            Assert.All(served.Keys!.Zip(served.Features), row => Assert.Equal(prepared[row.First].First(), row.Second));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Theory]
    [InlineData("titanic.parquet", "read.parquet", "DeepSharp.Pipelines.Parquet")]
    [InlineData("titanic.xlsx", "read.excel", "DeepSharp.Pipelines.Excel")]
    [InlineData("titanic.json", "read.json", "DeepSharp.Pipelines.Json")]
    public void AFileReadByAReader_IsRefusedWhereItsPackageIsNotRegistered_NamingThePackage(string file, string verb, string package)
    {
        var json = Declared(Reading(Pdd.Create(), file)).Declaration.ToJson();

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains($"'{verb}' is a step from {package}, which is not registered here", refused.Message, StringComparison.Ordinal);
        Assert.Equal(package, StepCatalog.PackageThatBrings(verb));
    }

    [Theory]
    [InlineData(".parquet", "cannot be read as a Parquet file")]
    [InlineData(".xlsx", "cannot be read as an Excel workbook")]
    [InlineData(".json", "is not JSON")]
    public void EachReader_RefusesAFileItCannotRead_NamingTheFileAndWhy(string reader, string why)
    {
        // The comma-separated passenger list, handed to each of the other readers by mistake.
        var folder = Directory.CreateTempSubdirectory("deepsharp-readers-").FullName;
        var misread = Path.Join(folder, "titanic" + reader);

        try
        {
            File.Copy(Repository.Data("titanic.csv"), misread);

            var refused = Assert.Throws<FormatException>(() => Declared(Reading(Pdd.Create(), misread)).Build().Run());

            Assert.StartsWith(misread + " " + why, refused.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Theory]
    [InlineData(".parquet")]
    [InlineData(".xlsx")]
    [InlineData(".json")]
    public void EachReader_RefusesAFileThatIsNotThere_AsTheCommaSeparatedOneDoes(string reader)
    {
        var path = Path.Join(Repository.Root, "nothing" + reader);

        Assert.Throws<FileNotFoundException>(() => Declared(Pdd.Create().ReadCsv(Path.ChangeExtension(path, ".csv"))).Build().Run());
        Assert.Throws<FileNotFoundException>(() => Declared(Reading(Pdd.Create(), path)).Build().Run());
    }

    public static TheoryData<string> EveryFileAReaderReads() =>
        new() { "titanic.csv", "titanic.parquet", "titanic.xlsx", "titanic.xls", "titanic.json" };

    // The passenger list in one format, where the tests keep it: the comma-separated file among the published data.
    private static string PassengerList(string file) =>
        Path.GetExtension(file) == ".csv" ? Repository.Data(file) : Repository.Fixture(file);

    [Theory]
    [MemberData(nameof(EveryFileAReaderReads))]
    public void EveryReader_OpensFromItsBytesTheRowsItOpensFromItsPath(string file)
    {
        // Whoever read a file's bytes to fingerprint them has the rows parsed from those very bytes, and they are the rows
        // the file opens as: the same columns, the same cells, and what a Parquet file states each column holds.
        var path = PassengerList(file);
        var read = Assert.IsAssignableFrom<IReadsAFile>(Reading(Pdd.Create(), path).Declaration.Steps[0]);

        var fromPath = ((IOpensRows)read).Open(SourceFolder.WorkingDirectory);
        var fromBytes = read.Open(File.ReadAllBytes(path), path);

        Assert.Equal(path, read.Path);
        Assert.Equal(fromPath.ColumnNames, fromBytes.ColumnNames);
        Assert.Equal(891, fromBytes.Rows.Count());
        Assert.Equal(fromPath.Rows, fromBytes.Rows);
        Assert.Equal(Path.GetExtension(file) == ".parquet", fromBytes is IStatesKinds);
        Assert.Equal((fromPath as IStatesKinds)?.StatedKinds, (fromBytes as IStatesKinds)?.StatedKinds);
    }

    [Fact]
    public void EveryStepThisLibraryShipsThatNamesItsColumns_ReadsAFile()
    {
        // A source that can name its columns before anything runs is a file, and every such step this library ships opens
        // its rows from the file's bytes as well: the one thing a notebook asks of its first block.
        var naming = Shipped.StepAssemblies
            .SelectMany(assembly => assembly.GetTypes())
            .Where(type => type is { IsClass: true, IsAbstract: false } && typeof(INamesItsColumns).IsAssignableFrom(type))
            .ToArray();

        Assert.Equal(
            [typeof(ReadCsvStep), typeof(ReadExcelStep), typeof(ReadJsonStep), typeof(ReadParquetStep)],
            naming.OrderBy(type => type.Name, StringComparer.Ordinal));
        Assert.All(naming, type => Assert.True(typeof(IReadsAFile).IsAssignableFrom(type), $"{type.Name} names its columns and reads no file."));
        Assert.All(naming, type => Assert.True(typeof(IOpensRows).IsAssignableFrom(type), $"{type.Name} reads a file and opens no rows."));
    }

    [Theory]
    [InlineData(".csv", "has no header row")]
    [InlineData(".parquet", "cannot be read as a Parquet file")]
    [InlineData(".xlsx", "cannot be read as an Excel workbook")]
    [InlineData(".json", "is not JSON")]
    public void BytesAReaderCannotRead_AreRefusedNamingTheFileTheyCameFrom(string reader, string why)
    {
        // The comma-separated passenger list's bytes, handed to each of the other readers by mistake; nothing at all, to the
        // comma-separated reader, which reads any text.
        var read = (IReadsAFile)Reading(Pdd.Create(), "titanic" + reader).Declaration.Steps[0];
        var named = "named" + reader;
        byte[] bytes = reader == ".csv" ? [] : File.ReadAllBytes(Repository.Data("titanic.csv"));

        var refused = Assert.Throws<FormatException>(() => read.Open(bytes, named));

        Assert.StartsWith(named + " " + why, refused.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => read.Open(null!, named));
        Assert.Throws<ArgumentException>(() => read.Open(bytes, " "));
    }

    [Theory]
    [MemberData(nameof(TitanicInEveryFormat))]
    public void ASchemaTakenOverAtEachReader_SaysWhichOfTheFilesColumnsItNeverShowed_AsAtTheCommaSeparatedFile(string file)
    {
        var saved = new PipelinePreset(new DeclareStep(
            [new ColumnDeclaration("survived", ColumnKind.Integer, false), new ColumnDeclaration("fare", ColumnKind.Number, false)]));

        Pdd.Create().ReadCsv(Repository.Data("titanic.csv")).Declare(saved, out var atTheCsv);
        Reading(Pdd.Create(), Repository.Fixture(file)).Declare(saved, out var atTheFile);

        Assert.Equal(13, atTheFile.NewColumns!.Count);
        Assert.Equal(atTheCsv.NewColumns, atTheFile.NewColumns);
        Assert.Equal(atTheCsv.Changes, atTheFile.Changes);
    }

    [Fact]
    public void RowsDeclaredToBeHandedIn_AndHandedNone_HaveNoColumnsToSayAnythingOf()
    {
        var saved = new PipelinePreset(new DeclareStep([new ColumnDeclaration("survived", ColumnKind.Integer, false)]));

        Pdd.Create().Add(new ReadRowsStep("rows handed in later")).Declare(saved, out var declared);

        Assert.Null(declared.NewColumns);
    }

    [Fact]
    public void AHostThatOffersEachReadersSteps_ReadsAFileThatNamesThem()
    {
        var services = new ServiceCollection();

        services.AddDeepSharpPipelines();
        services.AddSingleton<IStepContribution, ParquetSteps>();
        services.AddSingleton<IStepContribution, ExcelSteps>();
        services.AddSingleton<IStepContribution, JsonSteps>();

        using var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<StepCatalog>();

        foreach (var file in new[] { "titanic.parquet", "titanic.xlsx", "titanic.json" })
        {
            var declaration = Declared(Reading(Pdd.Create(), file)).Declaration;

            Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), catalog));
        }

        Assert.Throws<ArgumentNullException>(() => ParquetSourceExtensions.WithParquet(null!));
        Assert.Throws<ArgumentNullException>(() => ExcelSourceExtensions.WithExcel(null!));
        Assert.Throws<ArgumentNullException>(() => JsonSourceExtensions.WithJson(null!));
        Assert.Throws<ArgumentNullException>(() => ParquetSourceExtensions.ReadParquet(null!, "a.parquet"));
        Assert.Throws<ArgumentNullException>(() => ExcelSourceExtensions.ReadExcel(null!, "a.xlsx"));
        Assert.Throws<ArgumentNullException>(() => ExcelSourceExtensions.ReadExcel(null!, "a.xlsx", "sheet"));
        Assert.Throws<ArgumentNullException>(() => JsonSourceExtensions.ReadJson(null!, "a.json"));
        Assert.Throws<ArgumentException>(() => Pdd.Create().ReadParquet(" "));
        Assert.Throws<ArgumentException>(() => Pdd.Create().ReadExcel(" "));
        Assert.Throws<ArgumentException>(() => Pdd.Create().ReadJson(" "));
        Assert.Throws<ArgumentNullException>(() => new ReadParquetStep("a.parquet").Open(null!));
        Assert.Throws<ArgumentNullException>(() => new ReadExcelStep("a.xlsx").Open(null!));
        Assert.Throws<ArgumentNullException>(() => new ReadJsonStep("a.json").Open(null!));
    }

    [Theory]
    [InlineData(typeof(ReadParquetStep), "Parquet")]
    [InlineData(typeof(ReadExcelStep), "ExcelDataReader")]
    [InlineData(typeof(ReadJsonStep), null)]
    public void EachReadersPackage_ReferencesThePipelineAndTheOneLibraryItBrings_AndNothingElse(Type step, string? library)
    {
        // What a reader's package carries is what a project that references it carries, so it is a closed list: the
        // pipeline, and the library that reads the format — none at all for JSON, which .NET reads itself.
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var referenced = step.Assembly.GetReferencedAssemblies().Select(reference => reference.Name!).ToArray();
        var outside = referenced
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(library is null ? ["DeepSharp.Pipelines"] : ["DeepSharp.Pipelines", library])
            .ToArray();

        Assert.True(outside.Length == 0, $"{step.Assembly.GetName().Name} references {string.Join(", ", outside)}.");
        Assert.Contains("DeepSharp.Pipelines", referenced);
        Assert.Equal(library is not null, referenced.Contains(library));
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A workbook is how data arrives from people rather than from systems. Its cells are read as the sheet types them — a
/// number as the number it holds, never the text a culture shows it as; a date as a moment — its first row names the
/// columns, and the first sheet is read unless the pipeline names another. The workbooks here were written by Excel.
/// </summary>
public class ExcelSourceTests
{
    public static TheoryData<string> EveryFormat() => new() { "people.xlsx", "people.xls", "people.xlsb" };

    [Theory]
    [MemberData(nameof(EveryFormat))]
    public void EveryCellIsHandedOverAsTheSheetTypesIt(string file)
    {
        var rows = new ReadExcelStep(Repository.Fixture(file)).Open(SourceFolder.WorkingDirectory);

        Assert.Equal(["name", "age", "born", "adult", "note"], rows.ColumnNames);

        // A number with a fraction as the number it holds; a date the sheet types as one, as the moment it is; true or
        // false; a number the sheet holds as text, as the text; an empty cell empty, as the same sheet saved as
        // comma-separated text is; and a formula's error as Excel spells it.
        Assert.Equal(
            [
                ["Zoë Café", "22.5", "2015-02-18T00:00:00.0000000", "True", "22.0"],
                ["Bob", string.Empty, "#DIV/0!", "False", string.Empty],
            ],
            rows.Rows);
    }

    [Theory]
    [MemberData(nameof(EveryFormat))]
    public void ASheetsCellsBind_AsWhatTheSheetTypesThem(string file)
    {
        var people = Pdd.Create()
            .ReadExcel(Repository.Fixture(file))
            .Declare(schema => schema
                .Text("name", "note")
                .Number("age")
                .Boolean("adult")
                .Column(new ColumnDeclaration("born", ColumnKind.Timestamp, Optional: false) { Missing = "#DIV/0!" }))
            .Build()
            .Prepare();

        Assert.Equal(new DateTime(2015, 2, 18, 0, 0, 0, DateTimeKind.Utc), ((Column<DateTime>)people["born"])[0]);
        Assert.Null(((Column<DateTime>)people["born"])[1]);
        Assert.Equal(22.5, ((Column<double>)people["age"])[0]);
        Assert.Null(((Column<double>)people["age"])[1]);
        Assert.True(((Column<bool>)people["adult"])[0]);
        Assert.Equal("22.0", ((TextColumn)people["note"])[0]);
    }

    [Theory]
    [MemberData(nameof(EveryFormat))]
    public void TheFirstSheetIsRead_UnlessTheReaderNamesAnother(string file)
    {
        var path = Repository.Fixture(file);

        Assert.Equal(["name", "age", "born", "adult", "note"], Pdd.Create().ReadExcel(path).ProposedKinds().Columns.Select(column => column.Name));

        var pets = Pdd.Create().ReadExcel(path, "pets").ProposedKinds();

        Assert.Equal(["kind", "legs"], pets.Columns.Select(column => column.Name));
        Assert.Equal(ColumnKind.Integer, pets["legs"].Kind);
        Assert.Null(pets["legs"].Stated);
    }

    [Theory]
    [MemberData(nameof(EveryFormat))]
    public void ASheetTheWorkbookDoesNotHave_IsRefused_NamingTheSheetsItHas(string file)
    {
        var path = Repository.Fixture(file);

        var refused = Assert.Throws<FormatException>(() => Pdd.Create().ReadExcel(path, "fish").ProposedKinds());

        Assert.Equal($"{path} has no sheet called 'fish'. Its sheets are: people, pets.", refused.Message);
    }

    [Fact]
    public void ASheetWithNothingOnIt_HasNoColumnsToName_AndIsRefused()
    {
        var folder = Directory.CreateTempSubdirectory("deepsharp-excel-").FullName;

        try
        {
            // A workbook of one empty sheet, as the smallest one Excel writes: its sheet holds no row at all.
            var path = Path.Join(folder, "empty.xlsx");

            using (var zip = System.IO.Compression.ZipFile.Open(path, System.IO.Compression.ZipArchiveMode.Create))
            {
                Entry(zip, "[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>""");
                Entry(zip, "_rels/.rels", """<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>""");
                Entry(zip, "xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="blank" sheetId="1" r:id="rId1"/></sheets></workbook>""");
                Entry(zip, "xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>""");
                Entry(zip, "xl/worksheets/sheet1.xml", """<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData/></worksheet>""");
            }

            var refused = Assert.Throws<FormatException>(() => Pdd.Create().ReadExcel(path).ProposedKinds());

            Assert.Equal($"The sheet 'blank' of {path} has no header row, so its columns have no names.", refused.Message);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        static void Entry(System.IO.Compression.ZipArchive zip, string name, string text)
        {
            using var writer = new StreamWriter(zip.CreateEntry(name).Open());
            writer.Write(text);
        }
    }

    [Theory]
    [MemberData(nameof(EveryFormat))]
    public void AWorkbookCutShort_IsRefused_NamingTheFileAndWhy(string file)
    {
        var folder = Directory.CreateTempSubdirectory("deepsharp-excel-").FullName;

        try
        {
            // Half a workbook, as a copy stopped part of the way through leaves one.
            var path = Path.Join(folder, file);
            var whole = File.ReadAllBytes(Repository.Fixture(file));

            File.WriteAllBytes(path, whole[..(whole.Length / 2)]);

            var refused = Assert.Throws<FormatException>(() => Pdd.Create().ReadExcel(path).ProposedKinds());

            Assert.StartsWith($"{path} cannot be read as an Excel workbook: ", refused.Message, StringComparison.Ordinal);
            Assert.NotNull(refused.InnerException);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void ReadingAWorkbook_LeavesEveryEncodingTheProgramHadAsItWas()
    {
        // The code pages the reader asks for are offered to the whole program, and offering them changes none it had.
        int[] had = [65001, 1200, 1201, 12000, 12001, 20127, 28591];
        var before = had.Select(System.Text.Encoding.GetEncoding).ToArray();

        Pdd.Create().ReadExcel(Repository.Fixture("people.xlsx")).ProposedKinds();

        Assert.Equal(before, had.Select(System.Text.Encoding.GetEncoding), ReferenceEqualityComparer.Instance);
        Assert.Equal(1252, System.Text.Encoding.GetEncoding(1252).CodePage);
    }

    [Fact]
    public void ASheetNamedInAPipelineFile_IsReadBackWithIt_AndOneLeftOutReadsTheFirst()
    {
        var catalog = StepCatalog.BuiltIn().WithExcel();
        var named = Pdd.Create().ReadExcel("people.xlsx", "pets").Declaration;
        var first = Pdd.Create().ReadExcel("people.xlsx").Declaration;

        Assert.Contains("\"sheet\": \"pets\"", named.ToJson(), StringComparison.Ordinal);
        Assert.DoesNotContain("sheet", first.ToJson(), StringComparison.Ordinal);
        Assert.Equal(named, PipelineDeclaration.FromJson(named.ToJson(), catalog));
        Assert.Equal(first, PipelineDeclaration.FromJson(first.ToJson(), catalog));
        Assert.Equal("pets", ((ReadExcelStep)named.Steps[0]).Sheet);
        Assert.Null(((ReadExcelStep)first.Steps[0]).Sheet);

        // A sheet of nothing but spaces names none, in the chain and in a file alike.
        Assert.Null(new ReadExcelStep("people.xlsx", " ").Sheet);
        Assert.Equal(first, PipelineDeclaration.FromJson("""{"declaration":[{"step":"read.excel","path":"people.xlsx","sheet":""}]}""", catalog));

        // A new block starts with the first sheet, and the reference says the sheet may be left out.
        Assert.Equal("""{"step":"read.excel","path":"data.xlsx"}""", catalog.Describe("read.excel").Template);
        Assert.Contains("| `sheet` | words; may be left out | left out |", catalog.VerbReference(), StringComparison.Ordinal);
    }
}

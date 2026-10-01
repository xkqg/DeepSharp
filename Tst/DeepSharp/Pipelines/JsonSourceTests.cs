// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// JSON is what an API hands back: an array of records, one object a row. It is text, like a comma-separated file, so a
/// value is read as the file writes it — a number in its own spelling — and what is not one value to a cell is refused by
/// name, with its record, rather than flattened by guesswork.
/// </summary>
public sealed class JsonSourceTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-json-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string Written(string json)
    {
        var path = Path.Join(_folder, $"{Guid.NewGuid():N}.json");

        File.WriteAllText(path, json);

        return path;
    }

    [Fact]
    public void EveryValueIsReadAsTheFileWritesIt_AndAKeyARecordLeavesOut_IsAGap()
    {
        var path = Written("""
            [
              { "name": "Zoë", "age": 22.0, "big": 1e3, "adult": true, "deck": null, "note": "" },
              { "age": 7, "adult": false, "town": "Cherbourg" }
            ]
            """);

        var rows = new ReadJsonStep(path).Open(SourceFolder.WorkingDirectory);

        // The columns in the order their keys first appear; a number as it is spelled, true and false as they are.
        Assert.Equal(["name", "age", "big", "adult", "deck", "note", "town"], rows.ColumnNames);
        Assert.Equal(
            [
                ["Zoë", "22.0", "1e3", "true", null, string.Empty, null],
                [null, "7", null, "false", null, null, "Cherbourg"],
            ],
            rows.Rows);

        var bound = Pdd.Create()
            .ReadJson(path)
            .Declare(schema => schema.Text("name", "town").Number("age", "big").Boolean("adult").Optional("deck", ColumnKind.Text))
            .Build()
            .Prepare();

        Assert.Equal(1000, ((Column<double>)bound["big"])[0]);
        Assert.Null(((Column<double>)bound["big"])[1]);
        Assert.False(((Column<bool>)bound["adult"])[1]);
        Assert.Null(((TextColumn)bound["name"])[1]);
    }

    [Fact]
    public void AFileWithAByteOrderMark_IsReadAsOneWithout()
    {
        var path = Path.Join(_folder, "marked.json");

        File.WriteAllText(path, """[{"a":1}]""", new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

        Assert.Equal(["1"], new ReadJsonStep(path).Open(SourceFolder.WorkingDirectory).Rows.Single());
    }

    [Theory]
    [InlineData("""[{"a":1},{"a":{"b":2}}]""", "Record 2 of {0} holds an object under 'a'")]
    [InlineData("""[{"a":[1,2]}]""", "Record 1 of {0} holds a list under 'a'")]
    public void AValueThatIsNotOneValue_IsRefusedByName_WithItsRecord(string json, string refusal)
    {
        var path = Written(json);

        var refused = Assert.Throws<FormatException>(() => Pdd.Create().ReadJson(path).ProposedKinds());

        Assert.StartsWith(string.Format(System.Globalization.CultureInfo.InvariantCulture, refusal, path), refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"a":1}""", "{0} holds an object where a table's records stand: an array of objects, one a row.")]
    [InlineData("""[{"a":1}, 2]""", "Record 2 of {0} is a number, where each record is an object of its cells.")]
    [InlineData("""[{"a":1}, "b"]""", "Record 2 of {0} is words, where each record is an object of its cells.")]
    [InlineData("""[[1]]""", "Record 1 of {0} is a list, where each record is an object of its cells.")]
    [InlineData("""[true]""", "Record 1 of {0} is true or false, where each record is an object of its cells.")]
    [InlineData("""[null]""", "Record 1 of {0} is nothing, where each record is an object of its cells.")]
    [InlineData("""[]""", "{0} holds no records, so its columns have no names.")]
    [InlineData("""[{"a":1,"a":2}]""", "Record 1 of {0} names 'a' twice.")]
    public void WhatIsNotAnArrayOfRecords_IsRefused_SayingWhatItIs(string json, string refusal)
    {
        var path = Written(json);

        var refused = Assert.Throws<FormatException>(() => Pdd.Create().ReadJson(path).ProposedKinds());

        Assert.Equal(string.Format(System.Globalization.CultureInfo.InvariantCulture, refusal, path), refused.Message);
    }

    [Fact]
    public void TextThatIsNotJson_IsRefused_SayingWhere()
    {
        var path = Written("""[{"a":1},""");

        var refused = Assert.Throws<FormatException>(() => Pdd.Create().ReadJson(path).ProposedKinds());

        Assert.StartsWith($"{path} is not JSON: ", refused.Message, StringComparison.Ordinal);
        Assert.IsType<System.Text.Json.JsonException>(refused.InnerException, exactMatch: false);
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A part of a model, named with the settings that name takes: a layer, the optimizer that moves it, the loss it is
/// judged by. The pipeline knows none of them — the package that brings the step says which names there are and what
/// each takes — and holds every one of them to that: a name it was not given, a setting the name does not take, and a
/// setting the name takes and the file leaves out are each refused where they are written, so a declaration never runs
/// on a number nobody wrote down.
/// </summary>
public class PartsParameterTests
{
    private static readonly WholeNumberParameter Units = new("units", "How many numbers the layer answers with.", 16) { AtLeast = 1 };

    private static readonly NumberParameter Rate = new("rate", "The share of numbers left out while training.", 0.2) { Above = 0 };

    private static readonly TrueOrFalseParameter Best = new("best", "Whether the run ends holding its best numbers.", true);

    private static readonly PartKind Dense = new("dense", "A layer that answers with as many numbers as it is given.", [Units]);

    private static readonly PartKind Relu = new("relu", "A layer that leaves what is below nought at nought.", []);

    private static readonly PartKind Dropout = new("dropout", "A layer that leaves numbers out while it trains.", [Rate]);

    private static PartsParameter Layers() => new(
        "layers", "The layers, from the one the rows reach first.", [Dense.Declared(), Relu.Declared()], [Dense, Relu, Dropout]);

    private static PartsParameter Stopping() => new(
        "stopping", "When the run stops before its last pass.", [new PartKind("patience", "Waits.", [Best]).Declared()], [new PartKind("patience", "Waits.", [Best])])
    {
        Single = true,
    };

    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement;

    private static string Written(PartsParameter parameter, IReadOnlyList<PartDeclaration> value)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            parameter.Write(writer, value);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    [Fact]
    public void APartIsReadAsTheNameItGivesAndTheSettingsThatNameTakes()
    {
        var read = Layers().Read(Json("""{"layers": [{"kind": "dense", "units": 24}, {"kind": "relu"}]}"""));

        Assert.Equal(["dense", "relu"], read.Select(part => part.Kind));
        Assert.Equal(24, read[0].Whole("units"));
        Assert.Empty(read[1].Settings);
    }

    [Fact]
    public void ASettingIsReadAsTheKindOfValueItsOwnParameterHolds()
    {
        var read = Layers().Read(Json("""{"layers": [{"kind": "dropout", "rate": 0.25}]}"""));

        Assert.Equal(0.25, read[0].Number("rate"));
        Assert.Equal(0.25, read[0].Settings[0].Value.Number);
    }

    [Fact]
    public void ANameTheStepNeverGave_IsRefused_NamingTheOnesItDid()
    {
        var refused = Assert.Throws<FormatException>(() => Layers().Read(Json("""{"layers": [{"kind": "transformer"}]}""")));

        Assert.Contains("transformer", refused.Message, StringComparison.Ordinal);
        Assert.Contains("dense", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASettingTheNameDoesNotTake_IsRefused()
    {
        var refused = Assert.Throws<FormatException>(() => Layers().Read(Json("""{"layers": [{"kind": "relu", "units": 3}]}""")));

        Assert.Contains("units", refused.Message, StringComparison.Ordinal);
        Assert.Contains("relu", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASettingTheNameTakes_AndTheFileLeavesOut_IsRefused()
    {
        // Nothing is filled in behind the writer's back: a run replayed from the file is the run that was written down.
        var refused = Assert.Throws<FormatException>(() => Layers().Read(Json("""{"layers": [{"kind": "dense"}]}""")));

        Assert.Contains("units", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AListThatIsNoList_AndAPartThatIsNoPart_AreRefused()
    {
        Assert.Throws<FormatException>(() => Layers().Read(Json("""{"layers": "dense"}""")));
        Assert.Throws<FormatException>(() => Layers().Read(Json("""{"layers": [3]}""")));
        Assert.Throws<FormatException>(() => Layers().Read(Json("""{"layers": [{"units": 3}]}""")));
        Assert.Throws<FormatException>(() => Layers().Read(Json("{}")));
    }

    [Fact]
    public void WhatIsWritten_IsWhatIsReadBack()
    {
        var parameter = Layers();
        var read = parameter.Read(Json("""{"layers": [{"kind": "dense", "units": 24}, {"kind": "relu"}, {"kind": "dropout", "rate": 0.5}]}"""));

        Assert.Equal("""{"layers":[{"kind":"dense","units":24},{"kind":"relu"},{"kind":"dropout","rate":0.5}]}""", Written(parameter, read));
        Assert.Equal(read, parameter.Read(Json(Written(parameter, read))));
    }

    [Fact]
    public void OnePart_IsWrittenAsItself_RatherThanAsAListOfOne()
    {
        var parameter = Stopping();
        var read = parameter.Read(Json("""{"stopping": {"kind": "patience", "best": false}}"""));

        Assert.False(read[0].YesOrNo("best"));
        Assert.Equal("""{"stopping":{"kind":"patience","best":false}}""", Written(parameter, read));
        Assert.Throws<FormatException>(() => parameter.Read(Json("""{"stopping": [{"kind": "patience", "best": false}]}""")));
    }

    [Fact]
    public void TwoPartsThatSayTheSameThing_AreTheSamePart()
    {
        var one = Layers().Read(Json("""{"layers": [{"kind": "dense", "units": 24}]}"""));
        var same = Layers().Read(Json("""{"layers": [{"kind": "dense", "units": 24}]}"""));
        var other = Layers().Read(Json("""{"layers": [{"kind": "dense", "units": 25}]}"""));

        Assert.Equal(one[0], same[0]);
        Assert.Equal(one[0].GetHashCode(), same[0].GetHashCode());
        Assert.NotEqual(one[0], other[0]);
        Assert.NotEqual(one[0], new PartDeclaration("relu", []));
    }

    [Fact]
    public void APartAskedForASettingItDoesNotHold_SaysSo()
    {
        var read = Layers().Read(Json("""{"layers": [{"kind": "relu"}]}"""));

        Assert.Throws<KeyNotFoundException>(() => read[0].Whole("units"));
        Assert.Throws<KeyNotFoundException>(() => read[0].Number("rate"));
        Assert.Throws<KeyNotFoundException>(() => read[0].YesOrNo("best"));
    }

    [Fact]
    public void TheNamesAndTheirSettings_AreWhatTheFormAndTheWordsAreBuiltFrom()
    {
        var parameter = Layers();

        Assert.Equal(["dense", "relu", "dropout"], parameter.Kinds.Select(kind => kind.Name));
        Assert.Equal(["units"], parameter.Kinds[0].Settings.Select(setting => setting.Key));
        Assert.Equal("A layer that answers with as many numbers as it is given.", parameter.Kinds[0].Purpose);
        Assert.Equal(["dense", "relu"], parameter.Example.Select(part => part.Kind));
    }

    [Fact]
    public void AKindWithNoName_OrNoSettingsAtAll_IsRefusedWhereItIsWritten()
    {
        Assert.Throws<ArgumentException>(() => new PartKind(" ", "Nothing.", []));
        Assert.Throws<ArgumentException>(() => new PartKind("dense", " ", []));
        Assert.Throws<ArgumentNullException>(() => new PartKind("dense", "A layer.", null!));
        Assert.Throws<ArgumentException>(() => new PartsParameter("layers", "The layers.", [], []));
    }
    [Fact]
    public void ASettingOfEveryKindThereIs_IsWrittenAsItsOwnKindWritesIt_AndReadBackAsItself()
    {
        // A part's settings are written and read through the parameters its name takes, whichever kinds those are, so a
        // step that declares parts may describe them with any kind the library has.
        StepParameter[] kinds =
        [
            new TextParameter("text", "Words.", "words"),
            new FilePathParameter("path", "A file.", "data.csv"),
            new ColumnParameter("column", "A column.", "fare", ColumnKinds.Numbers),
            new NewColumnParameter("made", "A column it makes.", "made"),
            new NumberParameter("number", "A number.", 1.5),
            new WholeNumberParameter("whole", "A whole number.", 3),
            new TrueOrFalseParameter("yesOrNo", "Whether it is so.", true),
            new OneOfParameter<Scale>("scale", "A scale.", Scale.MidRange),
            new FillStrategyParameter("fill", "A way of filling.", With.Mean, ["mean"]) { What = "filling a gap" },
        ];

        foreach (var setting in kinds)
        {
            var kind = new PartKind("everything", "A part that takes one setting of this kind.", [setting]);
            var parameter = new PartsParameter("parts", "The parts.", [kind.Declared()], [kind]);
            var written = Written(parameter, [kind.Declared()]);

            Assert.Contains($"\"kind\":\"everything\"", written, StringComparison.Ordinal);
            Assert.Equal(kind.Declared(), parameter.Read(Json(written))[0]);
        }

        // A kind that holds several values is refused where the name is given its settings: a part's setting is one
        // value, so that a part is written, read back and compared as itself.
        StepParameter[] several =
        [
            new ColumnsParameter("columns", "Columns.", ["fare"], ColumnKinds.Numbers),
            new SeveralOfParameter<Part>("parts", "Parts.", [Part.Train]),
            new SplitSharesParameter(),
            new ColumnDeclarationsParameter("declared", "Declared columns.", [new ColumnDeclaration("fare", ColumnKind.Number, false)]),
            new PartsParameter("inside", "Parts of a part.", [new PartDeclaration("relu", [])], [new PartKind("relu", "A layer.", [])]),
        ];

        Assert.All(several, setting =>
        {
            var refused = Assert.Throws<ArgumentException>(() => new PartKind("everything", "A part.", [setting]));

            Assert.Contains("holds one value", refused.Message, StringComparison.Ordinal);
        });

        // And a setting a file may leave out: a part is written with everything its name takes, so there is no such thing.
        StepParameter[] absent =
        [
            new ShareParameter("share", "A share."),
            new NewColumnParameter("made", "A column it makes.", "made", optional: true),
        ];

        Assert.All(absent, setting =>
        {
            var refused = Assert.Throws<ArgumentException>(() => new PartKind("everything", "A part.", [setting]));

            Assert.Contains("always written", refused.Message, StringComparison.Ordinal);
        });
    }

    [Fact]
    public void APartNamedAtTheCallSite_IsHeldToTheNamesAndSettingsItsStepGave()
    {
        var parameter = Layers();

        Assert.Equal(2, parameter.Require([Dense.Declared(), Relu.Declared()]).Count);
        Assert.Throws<ArgumentException>(() => parameter.Require([]));
        Assert.Throws<ArgumentException>(() => parameter.Require([new PartDeclaration("transformer", [])]));
        Assert.Throws<ArgumentException>(() => parameter.Require([new PartDeclaration("relu", [new PartSetting("units", PartValue.Of(3))])]));
        Assert.Throws<ArgumentException>(() => parameter.Require([new PartDeclaration("dense", [])]));
        Assert.Throws<ArgumentException>(() => Stopping().Require([new PartKind("patience", "Waits.", [Best]).Declared(), new PartKind("patience", "Waits.", [Best]).Declared()]));
        Assert.Throws<ArgumentNullException>(() => parameter.Require(null!));
    }

    [Fact]
    public void AValueNoSettingCouldHold_IsRefusedWhereItIsMade()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => PartValue.Of(double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => PartValue.Of(double.PositiveInfinity));
        Assert.Throws<ArgumentNullException>(() => PartValue.Of((string)null!));
        Assert.Equal(PartValues.YesOrNo, PartValue.Of(true).Holds);
        Assert.Equal(PartValues.Text, PartValue.Of("words").Holds);
        Assert.Equal(PartValues.Number, PartValue.Of(1.5).Holds);
    }

    [Fact]
    public void TwoSettingsUnderOneKey_AndAPartOfNoKindAtAll_AreRefusedWhereTheyAreWritten()
    {
        var units = new WholeNumberParameter("units", "Units.", 1);

        Assert.Throws<ArgumentException>(() => new PartKind("dense", "A layer.", [units, units]));
        Assert.Throws<ArgumentException>(() => new PartsParameter("layers", "The layers.", [Dense.Declared()], [Dense, Dense]));
        Assert.Throws<ArgumentNullException>(() => new PartsParameter("layers", "The layers.", [Dense.Declared()], null!));
        Assert.Null(Dense.Setting("elsewhere"));
        Assert.False(new PartDeclaration("relu", []).Holds("units"));
    }

    [Fact]
    public void APartHoldingWordsAndAPartHoldingNothing_AreReadAndComparedAsTheyStand()
    {
        var named = new PartKind("named", "A part with a word and a column it makes.", [
            new TextParameter("word", "Words.", "words"),
            new NewColumnParameter("made", "A column it makes.", "made"),
        ]);
        var parameter = new PartsParameter("parts", "The parts.", [named.Declared()], [named]);
        var read = parameter.Read(Json("""{"parts": [{"kind": "named", "word": "other", "made": "here"}]}"""));

        Assert.Equal("other", read[0].Text("word"));
        Assert.Equal("here", read[0].Text("made"));

        Assert.True(read[0].Holds("word"));
        Assert.False(read[0].Holds("elsewhere"));
        Assert.Throws<KeyNotFoundException>(() => read[0].Text("elsewhere"));

        // A part no settings were ever given compares and counts as one that was given none.
        Assert.Equal(new PartDeclaration("relu", []), default(PartDeclaration) with { Kind = "relu" });
        Assert.Equal(new PartDeclaration("relu", []).GetHashCode(), (default(PartDeclaration) with { Kind = "relu" }).GetHashCode());
        Assert.False((default(PartDeclaration) with { Kind = "relu" }).Holds("units"));
        Assert.Throws<KeyNotFoundException>(() => (default(PartDeclaration) with { Kind = "relu" }).Number("units"));
    }

}

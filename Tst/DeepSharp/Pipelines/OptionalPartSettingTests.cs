// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A name a part may give can gain a setting after files were written that never heard of it. Such a setting says what
/// leaving it out means, and then a part written without it is the part written with it at that: read back as that, written
/// again without it, equal to it, and drawn by the schema, the reference and the form as something a file may leave out. A
/// setting that says no such thing is still required of every part, as it always was.
/// </summary>
public class OptionalPartSettingTests
{
    private static readonly WholeNumberParameter Size = new("size", "How many bulbs the lamp holds.", 4) { AtLeast = 1 };

    private static readonly TrueOrFalseParameter Shiny = new("shiny", "Whether the lamp shines.", true) { LeftOut = false };

    private static readonly NumberParameter Glow = new("glow", "How much it glows.", 0.9) { AtLeast = 0, LeftOut = 0 };

    private static readonly PartKind Lamp = new("lamp", "A lamp.", [Size, Shiny, Glow]);

    private static PartsParameter Lamps() => new("lamps", "The lamps.", [Lamp.Declared()], [Lamp]);

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
    public void ASettingTheFileLeavesOut_IsReadAsWhatLeavingItOutMeans()
    {
        var read = Lamps().Read(Json("""{"lamps": [{"kind": "lamp", "size": 6}]}"""));

        Assert.Equal(["size", "shiny", "glow"], read[0].Settings.Select(setting => setting.Key));
        Assert.Equal(6, read[0].Whole("size"));
        Assert.False(read[0].YesOrNo("shiny"));
        Assert.Equal(0, read[0].Number("glow"));
    }

    [Fact]
    public void ASettingSaidAtWhatLeavingItOutMeans_IsThePartOneThatLeavesItOutIs()
    {
        var left = Lamps().Read(Json("""{"lamps": [{"kind": "lamp", "size": 6}]}"""));
        var said = Lamps().Read(Json("""{"lamps": [{"kind": "lamp", "size": 6, "shiny": false, "glow": 0}]}"""));

        Assert.Equal(left[0], said[0]);
        Assert.Equal(left[0].GetHashCode(), said[0].GetHashCode());
    }

    [Fact]
    public void ASettingAtWhatLeavingItOutMeans_IsNotWritten_AndOneAwayFromIt_IsWrittenAfterTheOnesBeforeIt()
    {
        var plain = Lamps().Read(Json("""{"lamps": [{"kind": "lamp", "size": 6, "shiny": false, "glow": 0}]}"""));
        var lit = Lamps().Read(Json("""{"lamps": [{"kind": "lamp", "size": 6, "shiny": true, "glow": 0.25}]}"""));

        Assert.Equal("""{"lamps":[{"kind":"lamp","size":6}]}""", Written(Lamps(), plain));
        Assert.Equal("""{"lamps":[{"kind":"lamp","size":6,"shiny":true,"glow":0.25}]}""", Written(Lamps(), lit));
        Assert.Equal(lit[0], Lamps().Read(Json(Written(Lamps(), lit)))[0]);
    }

    [Fact]
    public void ASettingThatSaysNothingOfLeavingItOut_IsStillRefusedWhenTheFileLeavesItOut()
    {
        var refused = Assert.Throws<FormatException>(() => Lamps().Read(Json("""{"lamps": [{"kind": "lamp", "glow": 1}]}""")));

        Assert.Contains("leaves out 'size'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APartBuiltInCodeWithoutAnOptionalSetting_IsHeldToTheNamesAndTheSettingsRequired_AndComesBackWithItAtItsDefault()
    {
        var parameter = Lamps();
        var bare = new PartDeclaration("lamp", [new PartSetting("size", PartValue.Of(6))]);

        var held = parameter.Require([bare]);

        Assert.Equal(["size", "shiny", "glow"], held[0].Settings.Select(setting => setting.Key));
        Assert.Equal(parameter.Read(Json("""{"lamps": [{"kind": "lamp", "size": 6}]}"""))[0], held[0]);
        Assert.Throws<ArgumentException>(() => parameter.Require([new PartDeclaration("lamp", [new PartSetting("glow", PartValue.Of(1))])]));
        Assert.Throws<ArgumentException>(() => parameter.Require([new PartDeclaration("lamp", [new PartSetting("size", PartValue.Of(6)), new PartSetting("hue", PartValue.Of(1))])]));
    }

    [Fact]
    public void ANewPartStartsAtWhatLeavingASettingOutMeans_NotAtTheExampleItIsShownWith()
    {
        var started = Lamp.Declared();

        Assert.Equal(4, started.Whole("size"));
        Assert.False(started.YesOrNo("shiny"));
        Assert.Equal(0, started.Number("glow"));
        Assert.Equal(started, Lamps().Read(Json("""{"lamps": [{"kind": "lamp", "size": 4}]}"""))[0]);
    }

    [Fact]
    public void ThePartsInTheCatalog_AreDrawnByTheSchemaAndTheReference_WithTheSettingsAFileMayLeaveOutLeftOutOfWhatIsRequired()
    {
        var catalog = StepCatalog.BuiltIn();

        catalog.Register<LampLearnerStep>();

        var schema = JsonNode.Parse(catalog.JsonSchema())!;
        var lamp = Find(schema, node => node["properties"]?["glow"] is not null)!;

        Assert.Equal(["kind", "size"], lamp["required"]!.AsArray().Select(name => (string)name!));
        Assert.NotNull(lamp["properties"]!["shiny"]);
        Assert.Contains("`lamp`, which takes `size` and may leave out `shiny` and `glow`", catalog.VerbReference(), StringComparison.Ordinal);
    }

    [Fact]
    public void AStepHoldingAPartThatSaysNothingNew_IsWrittenAsItWas_AndReadBackAsItself()
    {
        var catalog = StepCatalog.BuiltIn();

        catalog.Register<LampLearnerStep>();

        var step = new LampLearnerStep([Lamp.Declared()]);
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            ((IPipelineStep)step).WriteTo(writer);
        }

        var text = Encoding.UTF8.GetString(buffer.WrittenSpan);

        Assert.DoesNotContain("shiny", text, StringComparison.Ordinal);
        Assert.DoesNotContain("glow", text, StringComparison.Ordinal);
        Assert.Equal(step, catalog.ReadStep(text));
    }

    // The first object in the document, depth first, that a test describes.
    private static JsonNode? Find(JsonNode? node, Func<JsonNode, bool> test)
    {
        if (node is JsonObject owner)
        {
            if (test(owner))
            {
                return owner;
            }

            return owner.Select(each => Find(each.Value, test)).FirstOrDefault(found => found is not null);
        }

        return node is JsonArray list ? list.Select(each => Find(each, test)).FirstOrDefault(found => found is not null) : null;
    }

    /// <summary>A learner a package of its own could bring, whose parts may leave settings out.</summary>
    public sealed record LampLearnerStep : IPipelineStep<LampLearnerStep>, INamesTheLearner, IDescribesColumns
    {
        private static readonly PartsParameter LampsKey = new("lamps", "The lamps.", [Lamp.Declared()], [Lamp]);

        public LampLearnerStep(IReadOnlyList<PartDeclaration> lamps) => Lamps = LampsKey.Require(lamps);

        public IReadOnlyList<PartDeclaration> Lamps { get; }

        public static string Name => "learn.lamps";

        public static string Purpose => "Names the lamps this pipeline is declared for.";

        public static StepParameters<LampLearnerStep> Parameters { get; } = new StepParameters<LampLearnerStep>()
            .With(LampsKey, step => step.Lamps);

        public string Verb => Name;

        public Needs Needs => Needs.Numbers;

        public static LampLearnerStep ReadFrom(JsonElement element) => new(LampsKey.Read(element));

        public ColumnState After(ColumnState before) => before;

        public bool Equals(LampLearnerStep? other) => other is not null && Lamps.SequenceEqual(other.Lamps);

        public override int GetHashCode() => Lamps.Aggregate(0, (hash, lamp) => HashCode.Combine(hash, lamp));
    }
}

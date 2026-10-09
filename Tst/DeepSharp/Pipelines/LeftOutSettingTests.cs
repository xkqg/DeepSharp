// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A number, or a part, that a step gained after files were written without it: leaving it out means what the step did
/// before it had it, that meaning is never written, and so a step that says nothing of it writes itself as it did before
/// and keeps its key — as a whole number a file may leave out already does.
/// </summary>
public class LeftOutSettingTests
{
    private static readonly PartKind Constant = new("constant", "The rate the optimizer starts at, every epoch.", []);

    private static readonly PartKind Halving = new(
        "halving", "The rate halved every so many epochs.", [new WholeNumberParameter("every", "How many epochs between two halvings.", 10) { AtLeast = 1 }]);

    [Fact]
    public void ANumberAFileMayLeaveOut_ReadsAsWhatLeavingItOutMeans_AndThatIsNeverWritten()
    {
        // The most norm a run's gradients are clipped to: none unless said, which is never written, and never below none.
        var clip = new NumberParameter("clip", "The most norm.", 0) { AtLeast = 0, LeftOut = 0 };

        Assert.Equal(0, clip.LeftOut);
        Assert.Empty(clip.RequiredKeys);
        Assert.Equal(0, clip.Read(Step("""{"step":"x"}""")));
        Assert.Equal(2.5, clip.Read(Step("""{"step":"x","clip":2.5}""")));
        Assert.Equal("{}", Written(clip, 0));
        Assert.Equal("""{"clip":2.5}""", Written(clip, 2.5));
        Assert.Contains("0 or more", Assert.Throws<ArgumentOutOfRangeException>(() => clip.Require(-1)).Message, StringComparison.Ordinal);
        Assert.Throws<FormatException>(() => clip.Read(Step("""{"step":"x","clip":"far"}""")));
    }

    [Fact]
    public void ANumberEveryFileWrites_IsRequired_AndWrittenWhateverItHolds()
    {
        var rate = new NumberParameter("rate", "How far.", 0.1) { Above = 0 };

        Assert.Null(rate.LeftOut);
        Assert.Equal(["rate"], rate.RequiredKeys);
        Assert.Equal("""{"rate":0}""", Written(rate, 0));
        Assert.Throws<FormatException>(() => rate.Read(Step("""{"step":"x"}""")));
    }

    [Fact]
    public void APartAFileMayLeaveOut_ReadsAsThePartLeavingItOutMeans_AndThatPartIsNeverWritten()
    {
        var schedule = Schedule();

        Assert.Equal([Constant.Declared()], schedule.LeftOut!);
        Assert.Empty(schedule.RequiredKeys);
        Assert.Equal([Constant.Declared()], schedule.Read(Step("""{"step":"x"}""")));
        Assert.Equal("{}", Written(schedule, [Constant.Declared()]));
        Assert.Equal("""{"schedule":{"kind":"halving","every":3}}""", Written(schedule, [Halving.Declared() with { Settings = [new PartSetting("every", PartValue.Of(3))] }]));
        Assert.Equal("halving", schedule.Read(Step("""{"step":"x","schedule":{"kind":"halving","every":3}}"""))[0].Kind);
        Assert.Equal([Constant.Declared()], schedule.Read(Step("""{"step":"x","schedule":{"kind":"constant"}}""")));
    }

    [Fact]
    public void APartEveryFileWrites_IsRequired_AndAPartMayBeLeftOutOnlyAsPartsItKnows()
    {
        var parts = new PartsParameter("schedule", "How the rate changes.", [Constant.Declared()], [Constant, Halving]) { Single = true };

        Assert.Null(parts.LeftOut);
        Assert.Null((new PartsParameter("schedule", "How the rate changes.", [Constant.Declared()], [Constant]) { LeftOut = null }).LeftOut);
        Assert.Equal(["schedule"], parts.RequiredKeys);
        Assert.Equal("""{"schedule":{"kind":"constant"}}""", Written(parts, [Constant.Declared()]));
        Assert.Throws<ArgumentException>(() => new PartsParameter("schedule", "How the rate changes.", [Constant.Declared()], [Constant, Halving])
        {
            Single = true,
            LeftOut = [new PartDeclaration("cyclic", [])],
        });
    }

    private static PartsParameter Schedule() =>
        new("schedule", "How the rate changes.", [Constant.Declared()], [Constant, Halving]) { Single = true, LeftOut = [Constant.Declared()] };

    private static JsonElement Step(string json)
    {
        using var document = JsonDocument.Parse(json);

        return document.RootElement.Clone();
    }

    private static string Written<T>(StepParameter<T> parameter, T value)
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
}

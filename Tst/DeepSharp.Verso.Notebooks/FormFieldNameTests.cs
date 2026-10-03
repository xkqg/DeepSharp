// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// Every field the form draws is one the form reads back, and the name is the parameter's own.
/// </summary>
/// <remarks>
/// A field's name is built from the parameter it belongs to and what the field is about — one column of a set, the kind a
/// schema gives a column, whether that column may be absent, how its moments are written, which of its values stands for a
/// gap. Each is written in one place and read in one place, and the two are found here by reflection and run against each
/// other: a part of a parameter that gains a way of being written and no way of being read says so the day it is written.
/// The parameter is the receiver rather than its key, so the two names a field is built from cannot be handed over the
/// wrong way round.
/// </remarks>
public class FormFieldNameTests
{
    // Every way of writing a field, paired with the way of reading it back: a method on a parameter that hands back the
    // field's name, and the one named after it that says whether a field is that and what it is about.
    public static TheoryData<string> EveryPart()
    {
        var parts = new TheoryData<string>();

        foreach (var (writer, _) in Pairs())
        {
            parts.Add(writer.Name);
        }

        return parts;
    }

    [Theory]
    [MemberData(nameof(EveryPart))]
    public void EveryWayOfWritingAField_IsReadBackAsWhatItWasAbout(string part)
    {
        var (writer, reader) = Pairs().Single(each => each.Writer.Name == part);
        var parameter = NotebookVerbs.Catalog().Describe("declare").Parameters[0];
        var about = Written(writer, parameter, "a/b");
        var read = new object?[] { parameter, about, null };

        Assert.StartsWith(parameter.Key, about, StringComparison.Ordinal);
        Assert.True((bool)reader.Invoke(null, read)!, $"'{about}' is written by {writer.Name} and read back by nothing.");
        Assert.Equal("a/b", read[2]);
    }

    [Theory]
    [MemberData(nameof(EveryPart))]
    public void AFieldOfAnotherParameter_IsReadBackByNoPart(string part)
    {
        var (writer, reader) = Pairs().Single(each => each.Writer.Name == part);
        var parameters = NotebookVerbs.Catalog().Describe("declare").Parameters;
        var mine = parameters[0];
        var another = NotebookVerbs.Catalog().Describe("normalise").Parameters[0];

        Assert.NotEqual(mine.Key, another.Key);
        Assert.False((bool)reader.Invoke(null, [another, Written(writer, mine, "a/b"), null])!);
    }

    public static TheoryData<string, string> EveryTwoParts()
    {
        var pairs = new TheoryData<string, string>();

        foreach (var (writer, _) in Pairs())
        {
            foreach (var (other, reader) in Pairs().Where(each => each.Writer.Name != writer.Name && each.Reader.Name != "IsMember"))
            {
                pairs.Add(writer.Name, other.Name);
            }
        }

        return pairs;
    }

    [Theory]
    [MemberData(nameof(EveryTwoParts))]
    public void NoPartOfAParameter_ReadsAnotherPartsField(string part, string other)
    {
        // The parts are kept apart by the word in the middle of the name, so one is never read as another. The exception is
        // the switch for one column, which has no word of its own — it is the bare name, so the form asks it only of a
        // parameter whose fields are switches, and it is left out of this.
        var (writer, _) = Pairs().Single(each => each.Writer.Name == part);
        var (_, reader) = Pairs().Single(each => each.Writer.Name == other);
        var parameter = NotebookVerbs.Catalog().Describe("declare").Parameters[0];

        Assert.False((bool)reader.Invoke(null, [parameter, Written(writer, parameter, "a/b"), null])!);
    }

    [Fact]
    public void EveryPartThereIs_IsExercised()
    {
        // The theories are only worth anything if they find every part: one added or taken away says so here first.
        Assert.Equal(
            ["Absent", "Format", "Kind", "Member", "Missing"],
            Pairs().Select(each => each.Writer.Name).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void APlaceAndOneOfItsSettings_AreReadBackAsTheNumberAndTheName()
    {
        var parameter = NotebookVerbs.Catalog().Describe("declare").Parameters[0];

        Assert.True(parameter.IsPlace(parameter.Place(2), out var place) && place == 2);
        Assert.True(parameter.IsSetting(parameter.Setting(3, "period"), out var at, out var setting) && at == 3 && setting == "period");
        Assert.False(parameter.IsPlace($"{parameter.Key}/first", out _));
        Assert.False(parameter.IsPlace($"{parameter.Key}/-1", out _));
        Assert.False(parameter.IsPlace("period", out _));
        Assert.Equal($"{parameter.Key}/value", parameter.Number());
    }

    // The field's name, as that way of writing it writes it for this parameter.
    private static string Written(MethodInfo writer, StepParameter parameter, string about) =>
        (string)writer.Invoke(null, [parameter, about])!;

    // Found rather than listed: each way of writing a field that takes what the field is about as words, and the reader
    // named after it.
    private static IEnumerable<(MethodInfo Writer, MethodInfo Reader)> Pairs()
    {
        var members = typeof(NotebookVerbs).Assembly.GetType("DeepSharp.Verso.Notebooks.FormVocabulary")!
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);

        foreach (var writer in members.Where(each =>
            each.ReturnType == typeof(string)
            && each.GetParameters() is [{ ParameterType: var first }, { ParameterType: var second }]
            && first == typeof(StepParameter) && second == typeof(string)))
        {
            if (members.FirstOrDefault(each => each.Name == "Is" + writer.Name) is { } reader)
            {
                yield return (writer, reader);
            }
        }
    }
}

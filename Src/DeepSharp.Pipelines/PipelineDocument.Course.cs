// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace DeepSharp.Pipelines;

// The course's part of the file: a course written and read through the same door as a pipeline and a preset.
internal sealed partial class PipelineDocument
{
    private const string CourseKey = "course";

    private static readonly string[] CourseKeys = [VersionKey, CourseKey];

    /// <summary>Writes a course: the version, then each step as its verb and what has been said of it.</summary>
    /// <param name="course">The course.</param>
    /// <returns>The file, indented, with a line feed between lines on every system.</returns>
    public static string Write(PipelineCourse course)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, PipelineDeclaration.Version);
            writer.WriteStartArray(CourseKey);

            foreach (var step in course.Steps)
            {
                writer.WriteStartObject();
                writer.WriteString(StepCatalog.StepKey, step.Verb);

                using var said = JsonDocument.Parse(step.Said);

                foreach (var property in said.RootElement.EnumerateObject())
                {
                    property.WriteTo(writer);
                }

                writer.WriteEndObject();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan).ReplaceLineEndings("\n");
    }

    /// <summary>Reads a course, each step through the catalog.</summary>
    /// <param name="json">The file.</param>
    /// <param name="catalog">The verbs its steps may name.</param>
    /// <returns>The course.</returns>
    /// <exception cref="PipelineFileException">Anything in the file is wrong; every fault is named.</exception>
    public static PipelineCourse ReadCourse(string json, StepCatalog catalog) => Whole(json).CourseFile(catalog);

    private PipelineCourse CourseFile(StepCatalog catalog)
    {
        using var document = JsonDocument.Parse(_text.Bytes);
        var root = document.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
        {
            _text.Fault(_text.Root, "A course file is one JSON object, holding its steps under 'course'.");

            throw _text.Refused();
        }

        // Named without exception, as a preset's is: no course was written before this version, and what it says of a verb
        // is read as that version meant it.
        var version = root.TryGetProperty(VersionKey, out _) ? VersionOf(root) : NoVersion("A course file");

        FaultKeysNotHeld(root, CourseKeys, "A course file");

        var steps = CourseStepsOf(root, version, catalog);

        _text.ThrowIfFaulty();

        return PipelineCourse.Of(steps);
    }

    /// <summary>The steps a course file lists, each read for what has been said of it; the ones that cannot be are faulted.</summary>
    private List<CourseStep> CourseStepsOf(JsonElement root, int version, StepCatalog catalog)
    {
        List<CourseStep> steps = [];

        if (!root.TryGetProperty(CourseKey, out var written) || written.ValueKind != JsonValueKind.Array)
        {
            _text.Fault(
                written.ValueKind == JsonValueKind.Undefined ? _text.Root : _text.Of(CourseKey),
                "A course file holds its steps as a list, under 'course'.");

            return steps;
        }

        if (written.GetArrayLength() == 0)
        {
            _text.Fault(_text.Of(CourseKey), "A course holds at least one step.");

            return steps;
        }

        var at = 0;

        foreach (var element in written.EnumerateArray())
        {
            if (CourseStepOf(element, at, version, catalog) is { } step)
            {
                steps.Add(step);
            }

            at++;
        }

        return steps;
    }

    /// <summary>One step of a course file: its verb and what is said of it, held to what the verb itself would read.</summary>
    private CourseStep? CourseStepOf(JsonElement element, int at, int version, StepCatalog catalog)
    {
        var place = _text.ElementsUnder(CourseKey)[at];
        var number = at + 1;

        if (element.ValueKind != JsonValueKind.Object)
        {
            _text.Fault(place, $"Step {number}: A step of a course is an object with a 'step' in it.");

            return null;
        }

        if (!element.TryGetProperty(StepCatalog.StepKey, out var named) || named.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(named.GetString()))
        {
            _text.Fault(place, $"Step {number}: A step of a course names its verb, as text under '{StepCatalog.StepKey}'.");

            return null;
        }

        var said = new JsonObject();

        foreach (var property in element.EnumerateObject().Where(property => property.Name != StepCatalog.StepKey))
        {
            said[property.Name] = JsonNode.Parse(property.Value.GetRawText());
        }

        var step = CourseStep.Of(named.GetString()!, said);

        // Judged as far as the verb can judge it now, by the rule the declaration refuses by: a step still waiting for something
        // is held to the keys it names, and a step with everything said to the verb's own words.
        var refusals = step.Judged(catalog, version).Refusals;

        foreach (var refusal in refusals)
        {
            _text.Fault(place, $"Step {number}: {refusal}");
        }

        return refusals.Count > 0 ? null : step;
    }
}

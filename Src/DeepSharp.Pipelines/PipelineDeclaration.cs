// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The ordered steps of a pipeline, as they were declared.
/// </summary>
/// <remarks>
/// This is the half of a saved pipeline that a person writes. The other half — what each step learned while
/// being fitted — is written by the fit and kept apart from it, which is what allows the same declaration to
/// be re-fitted on fresh data and two runs to be compared by their declarations alone.
/// </remarks>
public sealed class PipelineDeclaration : IEquatable<PipelineDeclaration>
{
    private readonly IPipelineStep[] _steps;

    /// <summary>A declaration of exactly these steps, in this order.</summary>
    /// <param name="steps">The steps, in the order they were written.</param>
    public PipelineDeclaration(IEnumerable<IPipelineStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        _steps = [.. steps];
    }

    /// <summary>The steps, in the order they were written.</summary>
    public IReadOnlyList<IPipelineStep> Steps => _steps;

    /// <summary>Writes the declaration as JSON: the machine's copy, which diffs and travels.</summary>
    /// <returns>The declaration as one JSON document.</returns>
    public string ToJson()
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteStartArray("declaration");

            foreach (var step in _steps)
            {
                step.WriteTo(writer);
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    /// <summary>Reads a declaration back, using the verbs this library ships with.</summary>
    /// <param name="json">The document a declaration was written as.</param>
    /// <returns>The declaration the document describes.</returns>
    /// <exception cref="FormatException">The document is not a declaration, or a step is incomplete.</exception>
    /// <exception cref="NotSupportedException">The document names a step nothing has registered.</exception>
    public static PipelineDeclaration FromJson(string json) => FromJson(json, StepCatalog.BuiltIn());

    /// <summary>Reads a declaration back, using the verbs a given catalog knows.</summary>
    /// <param name="json">The document a declaration was written as.</param>
    /// <param name="catalog">The verbs that may appear in it.</param>
    /// <returns>The declaration the document describes.</returns>
    /// <exception cref="FormatException">The document is not a declaration, or a step is incomplete.</exception>
    /// <exception cref="NotSupportedException">The document names a step the catalog does not know.</exception>
    public static PipelineDeclaration FromJson(string json, StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        using var document = JsonDocument.Parse(json);

        if (!document.RootElement.TryGetProperty("declaration", out var steps)
            || steps.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException("A pipeline file holds a 'declaration' with the steps in it.");
        }

        var read = new List<IPipelineStep>(steps.GetArrayLength());

        foreach (var step in steps.EnumerateArray())
        {
            read.Add(catalog.Read(step));
        }

        return new PipelineDeclaration(read);
    }

    /// <summary>The declaration as it is spoken: <c>read.csv -&gt; split.byTime -&gt; fill.missing</c>.</summary>
    /// <returns>The verbs in order, which is what a person checks first.</returns>
    public override string ToString() => string.Join(" -> ", _steps.Select(step => step.Verb));

    /// <summary>Whether two declarations say the same thing, step for step.</summary>
    /// <param name="other">The declaration to compare with.</param>
    /// <returns><see langword="true"/> when both declare the same steps in the same order.</returns>
    public bool Equals(PipelineDeclaration? other) =>
        other is not null && _steps.SequenceEqual(other._steps);

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as PipelineDeclaration);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var step in _steps)
        {
            hash.Add(step);
        }

        return hash.ToHashCode();
    }
}

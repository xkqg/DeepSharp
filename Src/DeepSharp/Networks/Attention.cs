// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How attention looks: over how many heads, whether a step may look only at the steps up to its own, and how many of its weights are left out as it trains.</summary>
/// <param name="Heads">How many heads the width of a step is split into, each looking for itself; it divides the width.</param>
public readonly record struct Attending(int Heads)
{
    /// <summary>Whether a step may look at no step after its own; it may look at any, unless said.</summary>
    public bool Causal { get; init; }

    /// <summary>The share of the weights left out while a layer trains, from nothing to below one; none, unless said.</summary>
    public double Dropout { get; init; }
}

/// <summary>What attention looks at: the queries that ask, the keys they are held against, and the values they take their answer from.</summary>
/// <param name="Query">A batch of series by steps by the width of a step: what asks.</param>
/// <param name="Key">A batch of series by steps by the width of a step: what each query is held against.</param>
/// <param name="Value">A batch of series by as many steps as the keys by the width of a step: what the answer is made of.</param>
public readonly record struct QueryKeyValue(Tensor Query, Tensor Key, Tensor Value);

/// <summary>
/// A layer that makes each step of a series out of the steps it is held against, in the share each one counts: scaled dot-product attention,
/// as PyTorch's <c>scaled_dot_product_attention</c> works it out.
/// </summary>
/// <remarks>
/// <para>
/// A query is held against every key, the products scaled by one over the root of a head's width and shared out among the keys by a softmax;
/// the answer is the values taken in those shares. A key the mask leaves out counts for nothing — a one in the mask is a step that takes part, a
/// nought one that is padding — and so does a key after a step's own where the layer is causal. A step with no key left to look at is made
/// nothing, as PyTorch makes it, and never a number that is not finite.
/// </para>
/// <para>
/// It is made of the products the seam already has, one series and one head at a time, so the gradient of a loss through it is the one the
/// tape works out and no engine learned anything for it.
/// </para>
/// </remarks>
public abstract class Attentive : Layer
{
    private const string HeadsKey = "heads";
    private const string CausalKey = "causal";
    private const string DropoutKey = "dropout";

    // A mask leaves a key out by adding this much below nothing to its score: far enough to share out nothing, near enough to stay finite.
    private const double Far = 1e9;

    private protected Attentive(Attending how)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(how.Heads, 1);
        _ = Dropout.RequireRate(how.Dropout);

        Attending = how;
    }

    /// <summary>How the layer looks: its heads, whether it is causal, and the share of its weights it leaves out as it trains.</summary>
    public Attending Attending { get; }

    /// <summary>The build a layer was written with, read back.</summary>
    /// <param name="settings">The object the layer was written as.</param>
    /// <param name="rebuilding">The reading of the file.</param>
    private protected static Attending Built(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new Attending(rebuilding.Whole(settings, HeadsKey))
        {
            Causal = rebuilding.Holds(settings, CausalKey),
            Dropout = rebuilding.Number(settings, DropoutKey, 0),
        };
    }

    /// <summary>Writes how the layer looks: its heads, and the two that say yes only where they do.</summary>
    /// <param name="writer">The writer.</param>
    private protected void WriteBuild(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(HeadsKey, Attending.Heads);

        if (Attending.Causal)
        {
            writer.WriteBoolean(CausalKey, true);
        }

        if (Attending.Dropout > 0)
        {
            writer.WriteNumber(DropoutKey, Attending.Dropout);
        }
    }

    /// <summary>What the layer makes of the queries, the keys and the values.</summary>
    /// <param name="pass">The backend the arithmetic runs on, and whether this pass trains.</param>
    /// <param name="looked">What it looks at.</param>
    /// <param name="mask">Which keys take part, a one, and which are padding, a nought: a series by key; nothing for keys that all take part.</param>
    /// <returns>A batch of series by the steps of the queries by the width of a step.</returns>
    /// <exception cref="ArgumentException">The queries, the keys or the values are not batches of series of one width the heads divide, or the mask is of another shape.</exception>
    private protected Tensor Attend(Pass pass, QueryKeyValue looked, Tensor? mask)
    {
        Require(looked, mask);

        var backend = pass.Backend;
        var (query, key, value) = looked;
        var count = query.Shape[0];
        var steps = query.Shape[1];
        var embed = query.Shape[2];
        var keys = key.Shape[1];
        var heads = Attending.Heads;
        var each = embed / heads;
        var queries = backend.Reshape(query, new Shape(count, steps * embed));
        var keyed = backend.Reshape(key, new Shape(count, keys * embed));
        var valued = backend.Reshape(value, new Shape(count, keys * embed));
        var chosen = Enumerable.Range(0, heads).Select(head => Selection.Of(embed, each, (row, column) => row == (head * each) + column)).ToArray();
        var put = chosen.Select(backend.Transpose).ToArray();
        var scale = backend.Scalar(1 / Math.Sqrt(each));
        var draws = pass.Mode == PassMode.Training && Attending.Dropout > 0 ? pass.Draws($"attention:{Path}") : null;
        Tensor? gathered = null;

        for (var series = 0; series < count; series++)
        {
            var pick = Pick(count, series);
            var asked = backend.Reshape(backend.MatMul(pick, queries), new Shape(steps, embed));
            var held = backend.Reshape(backend.MatMul(pick, keyed), new Shape(keys, embed));
            var taken = backend.Reshape(backend.MatMul(pick, valued), new Shape(keys, embed));
            var allowed = Allowed(backend, mask is null ? null : backend.Reshape(backend.MatMul(pick, mask), new Shape(1, keys)), steps, keys);
            var penalty = allowed is null ? null : backend.Scale(backend.Subtract(allowed, backend.Fill(allowed.Shape, 1f)), backend.Scalar(Far));
            var alive = allowed is null ? null : backend.MatMul(backend.Positive(backend.MatMul(allowed, backend.Fill(new Shape(keys, 1), 1f))), backend.Fill(new Shape(1, keys), 1f));
            Tensor? joined = null;

            for (var head = 0; head < heads; head++)
            {
                var scores = backend.Scale(backend.MatMul(backend.MatMul(asked, chosen[head]), backend.Transpose(backend.MatMul(held, chosen[head]))), scale);

                if (penalty is not null)
                {
                    scores = backend.Add(scores, penalty);
                }

                var weights = backend.Exp(backend.LogSoftmax(scores));

                if (alive is not null)
                {
                    weights = backend.Multiply(weights, alive);
                }

                if (draws is not null)
                {
                    weights = backend.Multiply(weights, Left(draws, steps, keys));
                }

                var placed = backend.MatMul(backend.MatMul(weights, backend.MatMul(taken, chosen[head])), put[head]);

                joined = joined is null ? placed : backend.Add(joined, placed);
            }

            var spread = backend.MatMul(Pick(count, series, column: true), backend.Reshape(joined!, new Shape(1, steps * embed)));

            gathered = gathered is null ? spread : backend.Add(gathered, spread);
        }

        return backend.Reshape(gathered!, new Shape(count, steps, embed));
    }

    // The row of a matrix of so many rows that holds the one at a place — or the column of a matrix of so many columns: a one there, noughts elsewhere.
    private static Tensor Pick(int count, int at, bool column = false) =>
        Tensor.From(column ? new Shape(count, 1) : new Shape(1, count), [.. Enumerable.Range(0, count).Select(place => place == at ? 1f : 0f)]);

    // Which keys a step may look at: a one where it may, as the mask of the series says and, for a causal layer, as the step's place does.
    private Tensor? Allowed(ITensorBackend backend, Tensor? row, int steps, int keys)
    {
        Tensor? allowed = row is null ? null : backend.MatMul(backend.Fill(new Shape(steps, 1), 1f), row);

        if (Attending.Causal)
        {
            var before = Selection.Of(steps, keys, (step, key) => key <= step);

            allowed = allowed is null ? before : backend.Multiply(allowed, before);
        }

        return allowed;
    }

    // The weights left out as a layer trains, and the ones kept scaled by one over the share kept: drawn in the order of the keys of each step.
    private Tensor Left(Draws draws, int steps, int keys)
    {
        var kept = 1 - Attending.Dropout;
        var scale = (float)(1 / kept);

        return Tensor.From(new Shape(steps, keys), [.. Enumerable.Range(0, steps * keys).Select(_ => draws.NextDouble() < kept ? scale : 0f)]);
    }

    private void Require(QueryKeyValue looked, Tensor? mask)
    {
        var (query, key, value) = looked;

        if (query.Shape.Rank != 3 || key.Shape.Rank != 3 || value.Shape.Rank != 3 || query.Shape[1] < 1 || key.Shape[1] < 1)
        {
            throw new ArgumentException(
                $"Attention reads batches of series, series by step by channel, of one step at least, and was handed queries of {query.Shape}, keys of {key.Shape} and values of {value.Shape}.",
                nameof(looked));
        }

        if (key.Shape != value.Shape || key.Shape[0] != query.Shape[0] || key.Shape[2] != query.Shape[2])
        {
            throw new ArgumentException(
                $"Attention holds the keys of as many series and of as wide a step as the queries, and values of the very shape of the keys: queries of {query.Shape}, keys of {key.Shape}, values of {value.Shape}.",
                nameof(looked));
        }

        if (query.Shape[2] % Attending.Heads != 0)
        {
            throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"Attention of {Attending.Heads} heads splits a step of {query.Shape[2]} channels, and that is not a whole number of channels for each head."),
                nameof(looked));
        }

        if (mask is not null && mask.Shape != new Shape(key.Shape[0], key.Shape[1]))
        {
            throw new ArgumentException(
                $"The mask of keys of {key.Shape} is a one or a nought for each key of each series, series by step, and was handed a {mask.Shape} one.",
                nameof(mask));
        }
    }
}

/// <summary>
/// Scaled dot-product attention with no number of its own: it asks of queries, keys and values handed to it apart, or of a series that is all three.
/// </summary>
/// <remarks>
/// <para>
/// Given one step it takes the series as queries, keys and values alike; given a second, it takes a mask of the steps that take part. Given
/// three — <c>.Takes("queries", "keys", "values")</c> — it takes them as PyTorch's <c>scaled_dot_product_attention</c> does, in the order
/// query, key, value, and a fourth is the mask of the keys.
/// </para>
/// </remarks>
public sealed class Attention : Attentive, ISaved<Attention>
{
    /// <summary>Attention looking as it is told to.</summary>
    /// <param name="how">The heads, whether the layer is causal, and the share of its weights it leaves out as it trains.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one head, or the dropout is not a share below one.</exception>
    public Attention(Attending how)
        : base(how)
    {
    }

    /// <inheritdoc />
    public static string Name => "attention";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static Attention Rebuild(JsonElement settings, Rebuilding rebuilding) => new(Built(settings, rebuilding));

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer) => WriteBuild(writer);

    /// <summary>Runs the layer over queries, keys and values.</summary>
    /// <param name="looked">What it looks at.</param>
    /// <param name="mask">Which keys take part, a one, and which are padding, a nought: a series by key; nothing for keys that all take part.</param>
    /// <param name="pass">The backend the arithmetic runs on, and whether this pass trains.</param>
    /// <returns>A batch of series by the steps of the queries by the width of a step.</returns>
    /// <exception cref="ArgumentException">What it looks at is not batches of series of one width the heads divide, or the mask is of another shape.</exception>
    public Tensor Forward(QueryKeyValue looked, Tensor? mask, Pass pass)
    {
        ArgumentNullException.ThrowIfNull(pass);

        return Attend(pass, looked, mask);
    }

    /// <summary>Runs the layer over a series, as queries, keys and values alike, given the mask of the steps that take part.</summary>
    /// <param name="input">The series: a batch of series by steps by channels.</param>
    /// <param name="mask">Which steps take part, a one, and which are padding, a nought: a series by step.</param>
    /// <param name="pass">The backend the arithmetic runs on, and whether this pass trains.</param>
    /// <returns>A batch of series by steps by channels.</returns>
    public Tensor Forward(Tensor input, Tensor mask, Pass pass)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(mask);
        ArgumentNullException.ThrowIfNull(pass);

        return Attend(pass, new QueryKeyValue(input, input, input), mask);
    }

    /// <inheritdoc />
    internal override bool Accepts(int steps) => steps is >= 1 and <= 4;

    /// <inheritdoc />
    internal override Tensor Taking(IReadOnlyList<Tensor> inputs, Pass pass) => inputs.Count switch
    {
        1 => Forward(inputs[0], pass),
        2 => Forward(inputs[0], inputs[1], pass),
        3 => Forward(new QueryKeyValue(inputs[0], inputs[1], inputs[2]), null, pass),
        _ => Forward(new QueryKeyValue(inputs[0], inputs[1], inputs[2]), inputs[3], pass),
    };

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => Attend(pass, new QueryKeyValue(input, input, input), null);
}

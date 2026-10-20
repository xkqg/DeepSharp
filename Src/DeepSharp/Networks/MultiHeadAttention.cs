// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Multi-head attention over a series, as PyTorch's <c>nn.MultiheadAttention</c> works it out for a batch of series that is the queries, the
/// keys and the values alike: the series projected into all three, attention over the heads, and the heads projected back.
/// </summary>
/// <remarks>
/// <para>
/// A layer holds, named as PyTorch names them, <c>in_proj_weight</c> and <c>in_proj_bias</c> — the three projections, query, key and value, one
/// block after another along the outputs — and the linear layer <c>out_proj</c>, with its own <c>weight</c> and <c>bias</c>. The weights are kept
/// inputs by outputs, as a linear layer's are, so a reader turns the ones PyTorch writes outputs by inputs. They start as PyTorch starts them:
/// the projection evenly within the Glorot bound, its bias and the output's bias at nothing.
/// </para>
/// <para>
/// Given a mask as its second take — <c>.Takes("steps", "mask")</c> — a batch of series of unequal lengths is read each as it would be read
/// alone: a padding step is a key nobody looks at. It looks at every step of its series, or at no step after its own where it is causal.
/// </para>
/// </remarks>
public sealed class MultiHeadAttention : Attentive, ISaved<MultiHeadAttention>
{
    private const string EmbedKey = "embed";

    /// <summary>Multi-head attention over steps of the given width, starting as PyTorch's does.</summary>
    /// <param name="embed">How many channels a step holds, which the heads divide.</param>
    /// <param name="how">The heads, whether the layer is causal, and the share of its weights it leaves out as it trains.</param>
    /// <param name="draws">The draws its start is taken from: the projection first, then the output's weights.</param>
    /// <exception cref="ArgumentOutOfRangeException">A step holds fewer than one channel, there are fewer than one head, or the dropout is not a share below one.</exception>
    /// <exception cref="ArgumentException">The heads do not divide the channels.</exception>
    public MultiHeadAttention(int embed, Attending how, Draws draws)
        : base(how)
    {
        ArgumentNullException.ThrowIfNull(draws);
        ArgumentOutOfRangeException.ThrowIfLessThan(embed, 1);

        if (embed % how.Heads != 0)
        {
            throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"Attention of {how.Heads} heads splits a step of {embed} channels, and that is not a whole number of channels for each head."),
                nameof(how));
        }

        Embed = embed;

        var fans = new Fans(embed, 3 * embed);

        InProjectionWeight = AddParameter("in_proj_weight", new GlorotUniform().Draw(new Shape(embed, 3 * embed), fans, draws));
        InProjectionBias = AddParameter("in_proj_bias", Tensor.Zeros(new Shape(3 * embed)));
        OutProjection = AddLayer("out_proj", new Dense(embed, embed, new Beginning(draws) { Bias = new Zeros() }));
    }

    /// <summary>How many channels a step holds.</summary>
    public int Embed { get; }

    /// <summary>The projection of a step into its query, its key and its value: channels by three times the channels.</summary>
    public Parameter InProjectionWeight { get; }

    /// <summary>The bias of that projection: three times the channels.</summary>
    public Parameter InProjectionBias { get; }

    /// <summary>The linear layer the heads, laid side by side, are projected back by.</summary>
    public Dense OutProjection { get; }

    /// <inheritdoc />
    public static string Name => "multiHeadAttention";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static MultiHeadAttention Rebuild(JsonElement settings, Rebuilding rebuilding) =>
        new(rebuilding.Whole(settings, EmbedKey), Built(settings, rebuilding), rebuilding.Draws);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(EmbedKey, Embed);
        WriteBuild(writer);
    }

    /// <summary>Runs the layer over a batch of series of unequal lengths.</summary>
    /// <param name="input">The series: a batch of series by steps by channels.</param>
    /// <param name="mask">Which steps take part, a one, and which are padding, a nought: a batch by steps.</param>
    /// <param name="pass">The backend the arithmetic runs on, and whether this pass trains.</param>
    /// <returns>A batch of series by steps by channels.</returns>
    /// <exception cref="ArgumentException">The input is not a batch of series of as many channels as the layer reads, or the mask is of another shape.</exception>
    public Tensor Forward(Tensor input, Tensor mask, Pass pass)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(mask);
        ArgumentNullException.ThrowIfNull(pass);

        return Run(input, mask, pass);
    }

    /// <inheritdoc />
    internal override bool Accepts(int steps) => steps is 1 or 2;

    /// <inheritdoc />
    internal override Tensor Taking(IReadOnlyList<Tensor> inputs, Pass pass) => Run(inputs[0], inputs.Count == 2 ? inputs[1] : null, pass);

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => Run(input, null, pass);

    private Tensor Run(Tensor input, Tensor? mask, Pass pass)
    {
        if (input.Shape.Rank != 3 || input.Shape[2] != Embed || input.Shape[1] < 1)
        {
            throw new ArgumentException(
                $"Attention of {Embed} channels reads a batch of series, series by step by channel, of one step at least, and was handed a {input.Shape} one.",
                nameof(input));
        }

        var backend = pass.Backend;
        var rows = backend.Reshape(input, new Shape(input.Shape.Count / Embed, Embed));
        var projected = backend.AddRow(backend.MatMul(rows, InProjectionWeight.Value), InProjectionBias.Value);
        var looked = new QueryKeyValue(Part(backend, projected, input.Shape, 0), Part(backend, projected, input.Shape, 1), Part(backend, projected, input.Shape, 2));

        return OutProjection.Forward(Attend(pass, looked, mask), pass);
    }

    // The query, the key or the value of every step: a block of the projection's columns, laid out as the series.
    private Tensor Part(ITensorBackend backend, Tensor projected, Shape series, int part) =>
        backend.Reshape(backend.MatMul(projected, Selection.Of(3 * Embed, Embed, (row, column) => row == (part * Embed) + column)), series);
}

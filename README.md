# DeepSharp — deep learning in C#

DeepSharp is a neural-network library for C#, .NET 10 and .NET 8. You build a network the way you prefer:
stack layers and fit them, or write the forward pass yourself. Both reach the same model, train with the
same loop, and save the same checkpoint.

It has no native dependency. Tensors and their arithmetic run on .NET's own SIMD primitives, so a model
built here travels inside an ordinary application — a console tool, a service, a desktop app — without a
hundred megabytes of platform-specific binaries alongside it and without a Python installation anywhere.

And it can draw. A training run produces a loss curve, a confusion matrix and a learning-rate schedule as
pictures, from the metrics the loop already has, with no separate viewer to start and no browser needed.

```csharp
using DeepSharp.Tensors;

var backend = new CpuBackend();

var a = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);
var b = Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]);

var sum = backend.Add(a, b);   // Tensor 2x2 — 11, 22, 33, 44
```

## Where this is

**This repository is at its beginning.** What is here works, is tested and is the foundation everything else
stands on — shapes that refuse a mismatch where it is written, immutable tensors, and the backend seam that
lets the arithmetic move elsewhere later without a model noticing. What is described above as the shape of
the library is the direction, not a claim about today. The [changelog](CHANGELOG.md) says what each release
actually added, and nothing is released until it is true.

Built so far:

| | |
|---|---|
| `Shape` | The axes of a tensor, as a value. Refuses a negative axis and a count that would overflow. |
| `Tensor` | A shape and the values that fill it, row-major and immutable. |
| `ITensorBackend` | Where the arithmetic happens — the seam that keeps a model independent of it. |
| `CpuBackend` | The one that ships: .NET's vector registers, no native library. |

Next, in order: automatic differentiation, the layer types, the optimizers, and the training loop that
drives them.

## Why not TorchSharp or TensorFlow.NET

Both are bindings: they hand you the Python library's API, in C# syntax, with the native engine underneath.
That is the right answer when you want everything PyTorch can do and you can carry its binaries. DeepSharp
answers a different question — a model that ships with your application, reads like C# rather than like
transliterated Python, and shows you what it is doing without a second tool.

For the heavy work, that is not a rivalry. The backend seam is there so the same model can hand its
arithmetic to a native engine when the work outgrows a CPU.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first — it carries the rules this repository is actually held to:
a failing test before the code, zero warnings, and a coverage gate that fails rather than reports.

## Licence

MIT. See [LICENSE](LICENSE).

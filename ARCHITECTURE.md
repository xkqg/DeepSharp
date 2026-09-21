# Architecture

This file records the decisions that shape DeepSharp, and why each one was taken. A decision that is not
written down here is not a decision, it is a habit.

## The layout

```
Src/DeepSharp/            the library
  Tensors/                Shape, Tensor, ITensorBackend, CpuBackend
Tst/DeepSharp/            the tests, mirroring the library's folders
```

## Decisions

### A tensor knows nothing about arithmetic

`Tensor` is a shape and the values that fill it. Everything done to one is done by an `ITensorBackend`.

The alternative — methods on the tensor itself, `a.Add(b)` — reads better for one line and then decides the
architecture: the tensor has to know where its arithmetic happens, so it has to hold a backend, so either
every tensor carries one or there is a shared one somewhere. The second is a global that any code in the
process can reach and replace, which is exactly what this fleet does not build.

So the backend is handed to what needs it, and the model stays independent of where its work runs. That seam
exists from the first line rather than being retrofitted, because retrofitting it means touching every
operation that was written without it.

### A tensor never changes

Handing the same tensor to two layers is safe, and a caller who reuses a scratch buffer cannot rewrite a
tensor they already handed over — `Tensor.From` copies. An in-place variant will come when a measurement
shows the copying costs something that matters; until then, the correctness is worth more than the
allocation.

### A shape is a value with no meaningless default

`default(Shape)` cannot be prevented — a struct can always be brought into existence without a constructor.
So it means something sensible: no axes, which is a single value, which is what a loss is. `Count` is read
rather than stored for that reason alone.

The axis count is accumulated as a `long` and refused above `int.MaxValue`. An `int` product wraps around
silently, and a wrapped count buys a buffer far too small for what is about to be written into it — which
surfaces as memory corruption thousands of operations later, nowhere near the shape that caused it.

### The CPU backend uses .NET's own primitives

`System.Numerics.Tensors` gives vectorised element-wise arithmetic with no native dependency. That is the
promise this library makes: a model that travels inside an ordinary application. A native engine can be
added behind the same seam later, for work that outgrows a CPU; it will never become required.

### Charts come from the training loop, not from the caller

When the training loop exists, the metrics it already keeps are what the charts are drawn from — the caller
never assembles arrays to plot. The drawing itself lives in a separate, optional package, so a trainer on a
headless machine does not carry a renderer.

## What is deliberately absent

- **A global backend, context or session.** Nothing reaches for a shared instance.
- **A graph that is not a model.** The declarative front door will produce the same object the imperative
  one does. Two representations of one network means two engines to keep in step, and they diverge on the
  first unusual model.
- **A native dependency in the core package.** It stays optional, behind the seam, forever.

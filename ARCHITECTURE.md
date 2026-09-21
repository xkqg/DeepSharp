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

### The engine is a choice, and this library is what sits above it

An engine gives you fast arithmetic. It does not give you a way to describe a network in C#, pour real data
into it, drive a training loop, keep checkpoints or show what happened. That is what DeepSharp is, and it is
why an engine sits behind `ITensorBackend` rather than being the thing you program against.

`CpuBackend` is the one that needs no installing: `System.Numerics.Tensors`, the processor's own vector
instructions, nothing native. A TorchSharp backend belongs beside it as an equal — libtorch is a library
like any other, and rewriting what it already does well would be the most expensive way to learn nothing.

What differs between them is what a project has to ship, not what a model has to say. The light one travels
inside an application; libtorch is 76 MB per platform. Both are legitimate, the choice is the caller's, and
it is one line.

### Charts come from the training loop, not from the caller

When the training loop exists, the metrics it already keeps are what the charts are drawn from — the caller
never assembles arrays to plot. The drawing itself lives in a separate, optional package, so a trainer on a
headless machine does not carry a renderer.

## What is borrowed on purpose

Anything available from outside is something nobody here has to write or maintain, and that is the default
answer for every well-solved problem: DataFrames, file formats, compression, the test runner, the vector
maths. A hand-rolled version of any of those costs forever and buys nothing.

libtorch is on that list too. What this repository builds is the part nobody else provides: the C# shape of
a model, the path data takes into it, the loop that trains it, and the picture at the end.

The one rule about a dependency is where it lands. Anything heavy gets its own package, so a project that
does not want it never carries it, and the core stays light enough to travel inside an application. A model
is written against the seam and cannot tell which engine is underneath.

## What is not borrowed

The fleet's GPU training host has optimisations it earned against its own workload. None of them come here.
They were measured on one machine, one family of models and one kind of data, and a speed-up that cannot be
re-measured in this repository is a guess with a good reputation.

A published binding is the opposite case. TorchSharp is a dependency: versioned, readable, replaceable, and
tested by people who are not us. If a backend is ever written against it, that is borrowing a library, not
borrowing a result.

## What is deliberately absent

- **A global backend, context or session.** Nothing reaches for a shared instance.
- **A graph that is not a model.** The declarative front door will produce the same object the imperative
  one does. Two representations of one network means two engines to keep in step, and they diverge on the
  first unusual model.
- **A required engine.** No backend is mandatory. The core carries the light one; anything heavier is its
  own package, and a model cannot tell which is underneath it.

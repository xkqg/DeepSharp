# Changelog

What changed in each release, and what it means for you. The heading of a section is the version it shipped
as. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0]

The first release. It is the foundation the rest of the library is built on: the numbers, their size, and
where the arithmetic happens.

### Added

- **`Shape` — how big a tensor is.** `new Shape(2, 3)` is two rows of three, and it prints as `2x3`. Two
  shapes with the same sizes count as the same shape, so you can compare them and use them as keys. A size
  that cannot exist is refused the moment you write it: a negative length, or sizes so large that the total
  could not be counted. A shape with no sizes at all is a single number — which is what a loss is.

- **`Tensor` — the numbers themselves.** A shape, and the values filling it. A tensor never changes once it
  exists, so handing the same one to two parts of your network is safe. `Tensor.From` takes a copy of what
  you give it, so reusing your own array afterwards cannot alter a tensor you have already passed on.

- **`CpuBackend` — the arithmetic.** Adds and multiplies tensors, using your processor's vector
  instructions through .NET's own maths library. There is nothing native behind it and nothing to install.
  Adding two tensors of different sizes is refused, with a message naming both.

- **`ITensorBackend` — the plug it all goes through.** Every calculation a network does goes through this,
  which is why a model written today can run its heavy work on something else later without being rewritten.
  You create a backend and hand it to what needs it; nothing goes looking for a shared one of its own accord.

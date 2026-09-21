# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The heading of a section is
the version number it shipped as; a section without a number has not shipped.

## Unreleased

### Added

- **A shape that refuses a mismatch where it is written.** `Shape` holds the length of each axis as a value:
  two shapes with the same axes are equal, can be compared and hashed, and print as `2x3`. A negative axis is
  refused at the point it is written, and so is a set of axes that would describe more values than can be
  counted — the count is accumulated as a `long` because an `int` product wraps around in silence and buys a
  buffer far too small for what is about to be written into it. A shape with no axes is a single value, which
  is what a loss is, and that holds for a shape that was never given a constructor to run.

- **A tensor that never changes.** `Tensor` is a shape and the values that fill it, laid out row-major.
  `Tensor.From` copies what it is given, so a caller who reuses a scratch buffer cannot rewrite a tensor they
  have already handed to a layer. Handing the same tensor to two layers is therefore safe.

- **The seam the arithmetic runs behind.** `ITensorBackend` is where everything done to a tensor actually
  happens, and it is handed to whatever needs it rather than reached for — there is no shared instance
  anywhere. A model written against the seam does not know, and never has to learn, where its work runs.

- **The backend that ships: `CpuBackend`.** Element-wise addition and multiplication on .NET's own vector
  registers, through `System.Numerics.Tensors`, with no native library behind it and nothing to install. Two
  tensors of different shapes are refused with a message that names both. Lengths that do not divide evenly
  into a vector register are covered by tests, because the tail of a vectorised loop is where an off-by-one
  hides quietly.

# Contributing to DeepSharp

These are the rules this repository is held to. They are short, and none of them is optional.

## A failing test comes first

Every addition, change, fix and refactor starts with a test that fails. Write it, run it, watch it fail for
the reason you expect, and only then write the code that makes it pass. A test written afterwards proves
that the code does what it does; a test written first proves that the code does what was asked.

A bug starts with a test that reproduces it. If you cannot reproduce it in a test, you do not yet know what
the bug is.

## The build ends at zero warnings

Warnings are errors here — the build says so, so there is no log to read and nothing to forget. New code
introduces none. This includes documentation: every public member of the library carries an XML comment,
because the public API is the product.

Tests are exempt from the documentation rule. A test's name is its documentation, and a `<summary>` that
repeats the name is noise.

## Coverage is a gate, not a report

90% of lines and 90% of branches, **per class** and over the library as a whole, checked in CI. Below either
number the build fails and the failing classes are named. A gate that reports and continues is a report.

Per class is the part that does the work. A total has a big denominator: a hundred well-covered classes
carry an untested one across the line, and the gap stays invisible until somebody edits it. Code that moves
into a smaller class brings its untested branches with it, and they only become visible at the granularity
they landed in. The compiler's own types — a lambda's closure, an iterator's state machine — are counted
with the class they were generated for, because that is the class somebody wrote.

If a branch is hard to reach, that is usually the code telling you it should not exist. Delete it before you
reach for a test that pretends to cover it.

## Names say what a thing is

- No `Helper`, `Util`, `Utility` or `Manager` classes. A shared function on a type is an extension method on
  that type; a shared abstraction across types is a generic base or interface. A class whose name ends in
  `Helper` is a drawer nobody owns.
- Several values out of a method is a `readonly record struct` with named members, never a tuple. A tuple
  loses its names in IntelliSense, in stack traces and in the documentation.
- Nothing reaches for a shared mutable instance of its own accord. What a type needs is handed to it.

## The version is handed out, never invented

The owner names the number when the work starts. It is written once, as the single `<Version>` in
`Directory.Build.props`, which every packable project takes, and that same number heads the changelog and
stands in the README at once — a test holds the three together, and another refuses a second `<Version>`
anywhere. Nothing is packed or published before it is named.

## Every change ends with the documents

The same commit that changes behaviour updates what describes it: the README when the shape of the library
changes, `ARCHITECTURE.md` when a decision changes, the changelog always, and the XML comments on whatever
you touched. A change whose documents lag is not finished.

Product text — README, changelog, XML comments, the wiki — is written for people, in plain language, and
carries measurements rather than dates.

## Commits

The subject line says what changed, in plain English, in the present tense: *the shape refuses an axis that
would overflow*, not *fix*. The reasoning and the numbers go in the body. One commit per finished piece of
work, and it is pushed only when the whole thing is green locally.

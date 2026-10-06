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
- An extension method is written as a member of an `extension(T value) { … }` block, never with a `this`
  parameter. The members inside a block carry no `static` and no `this`, so the call site reads as a sentence and the
  thing the method is about is the receiver rather than the first of its arguments. The enclosing class is still
  `static`, which the language requires. A `public static` method whose first parameter is the thing the method is
  about belongs in such a block; one whose whole body is `new`, handing back a fresh instance of the type it names, is
  a factory and stays where it is.
- Several values out of a method is a `readonly record struct` with named members, never a tuple. A tuple
  loses its names in IntelliSense, in stack traces and in the documentation.
- A method or a constructor takes four parameters at most, public or not, and the receiver of an extension method is
  one of them. A fifth says something is missing. Ask first which object owns what the method does, and which of the
  parameters is really a collaborator with behaviour of its own; only values that are set once and do nothing belong
  together in a `readonly record struct`, named for what they are, never an options bag named after the method.
  Three things are not counted: what the compiler writes on its own (the method behind a lambda or a local function, an
  iterator, the members a record is given, such as its `Deconstruct`), what the runtime implements (a delegate's
  `Invoke`, `BeginInvoke` and `EndInvoke`), and the primary constructor of a positional record, which lists the record's
  members rather than a call's arguments. `ParameterListTests` reads every package and holds it to this. The only members
  that take more are forms an earlier release published: each is kept so that code written against it still compiles,
  marked obsolete in favour of its shorter form, and named in that test, so no other can join them.
- Nothing reaches for a shared mutable instance of its own accord. What a type needs is handed to it.

## The version is handed out, never invented

The owner names the number when the work starts. It is written once, as the single `<Version>` in
`Directory.Build.props`, which every packable project takes, and that same number heads the changelog and
stands in the README at once — a test holds the three together, and another refuses a second `<Version>`
anywhere. Nothing is packed or published before it is named.

A published version is also what the next one is held to. When a package is packed, its public surface is compared with
the last release's — `PackageValidationBaselineVersion` in `Directory.Build.props`, moved to the version just published
in the commit that begins the next one — so a change nobody meant to make breaks the build instead of somebody's upgrade.
A package that is new in a release is added to `PublishedBefore` when the one after it begins, and a test holds that list
to the packable projects. A break that is meant is written into a suppression file beside the project, and said out loud
in the changelog's upgrading notes.

## What the steps write, and is committed

`pipeline.schema.json`, which editors validate a pipeline file against, and `VERBS.md`, the reference of every verb, are
written by the steps themselves and committed, and `ProjectionTests` fails when either no longer says what the steps do.
A change to what a verb takes or writes writes them again: run the core's suite for that class with
`DEEPSHARP_REGEN_GOLDEN=1`, read what changed, and commit it with the change.

```
DEEPSHARP_REGEN_GOLDEN=1 dotnet run --project Tst/DeepSharp/DeepSharp.Tests.csproj -c Release -f net10.0 -- -class DeepSharp.Tests.Pipelines.ProjectionTests
```

They are never edited by hand, since a hand-edited reference is the drift they exist to prevent.

## Every change ends with the documents

The same commit that changes behaviour updates what describes it: the README when the shape of the library
changes, `ARCHITECTURE.md` when a decision changes, the changelog always, and the XML comments on whatever
you touched. A change whose documents lag is not finished.

Product text — README, changelog, XML comments, the wiki — is written for people, in plain language, and
carries measurements rather than dates.

The code in the README and on the wiki is code a person copies, so tests compile it against the packages as built and
fail on a warning: the README's example runs, every C# block of the wiki compiles, and a block the wiki marks as a
notebook's cell runs in Verso's own engine. The wiki is a repository of its own; clone it beside this one, as
`DeepSharp.wiki`, where those tests and the build machine read it.

## Nothing here asks Binance

`DeepSharp.Pipelines.Binance` fetches from Binance, and the address a build runs on shares one request budget with
whatever else runs there, so a test, a check or a program in a document that a suite runs never sends it a request. The
suites that run documents send every address but this machine's to a proxy nothing listens at, before any test is found;
the check of the package lands from a stand-in venue on this machine; and a test holds the documents that are run to
never naming the landing — a page that lands is compiled and never run. The venue the tests ask is generated, and no
payload of Binance's goes into the repository, since the data comes under terms of its own. Seeing Binance's real answer
for a change is one request made by hand, to `/api/v3/time` or for a page of at most ten candles, and never a loop.

## Commits

The subject line says what changed, in plain English, in the present tense: *the shape refuses an axis that
would overflow*, not *fix*. The reasoning and the numbers go in the body. One commit per finished piece of
work, and it is pushed only when the whole thing is green locally.

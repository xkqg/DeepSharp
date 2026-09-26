# Security

## Reporting a vulnerability

Report it privately, not in a public issue: open a
[security advisory](https://github.com/xkqg/DeepSharp/security/advisories/new) on this repository. That
channel is visible only to the maintainer until a fix is released, so the report does not become an
exploit notice while there is nothing to upgrade to.

You will get an answer within a week. If the report is confirmed you will be told when a fix is planned and
credited in the release notes unless you would rather not be.

## What is in scope

DeepSharp runs the model you give it, inside your own process. That makes three things worth reporting, and a
fourth for `deepsharp-serve`, the one part of DeepSharp that listens on a port:

- **A model file or checkpoint that takes over the process when it is loaded.** Reading a file must never be
  able to run code. If a crafted file leads to execution, arbitrary file access, or a crash that a caller
  cannot catch, that is a vulnerability.
- **A tensor operation that reads or writes outside its own buffer.** Shapes are checked where they are
  handed over precisely so that a wrong shape is a refusal and never a stray write.
- **A dependency of this package with a known advisory.** Report it even if you are not sure it is reachable
  from here; deciding that is our job.
- **A way past `deepsharp-serve`'s boundary.** The notebook it serves runs code as the person who started it, so
  the server listens on that computer alone, answers only a request carrying the token it printed, makes a change only
  for its own page, answers only under a name of that computer, and serves nothing from the folder it runs in. A
  browser carries the server's cookie for a page from any port of that computer, which is why the page a change comes
  from is checked as well as the token. It writes into that folder only a new notebook a page asks for, under the bare
  name of a `.verso` file, and never over a file. A request that reaches a notebook from another computer, from a page
  that does not hold the token, or under another name, a change that comes from any page but the server's own — one on
  the same computer included — a request that reads a file from that folder, or one that makes the server write
  anywhere else or over a file, is a vulnerability.

## What is not

- A model that gives bad answers, trains poorly, or overfits. That is a correctness issue — open an issue.
- Running out of memory on a tensor you asked for. A shape that does not fit is refused where it is written;
  one that fits but is larger than your machine is your own arithmetic.
- Anything that requires an attacker to already be running code in your process.

## Supported versions

The most recent released version is the one that gets fixes. This project has no long-term support branches;
if you are pinned to an older version, say so in the report and you will be told what the upgrade involves.

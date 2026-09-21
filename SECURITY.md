# Security

## Reporting a vulnerability

Report it privately, not in a public issue: open a
[security advisory](https://github.com/xkqg/DeepSharp/security/advisories/new) on this repository. That
channel is visible only to the maintainer until a fix is released, so the report does not become an
exploit notice while there is nothing to upgrade to.

You will get an answer within a week. If the report is confirmed you will be told when a fix is planned and
credited in the release notes unless you would rather not be.

## What is in scope

DeepSharp runs the model you give it, inside your own process. That makes three things worth reporting:

- **A model file or checkpoint that takes over the process when it is loaded.** Reading a file must never be
  able to run code. If a crafted file leads to execution, arbitrary file access, or a crash that a caller
  cannot catch, that is a vulnerability.
- **A tensor operation that reads or writes outside its own buffer.** Shapes are checked where they are
  handed over precisely so that a wrong shape is a refusal and never a stray write.
- **A dependency of this package with a known advisory.** Report it even if you are not sure it is reachable
  from here; deciding that is our job.

## What is not

- A model that gives bad answers, trains poorly, or overfits. That is a correctness issue — open an issue.
- Running out of memory on a tensor you asked for. A shape that does not fit is refused where it is written;
  one that fits but is larger than your machine is your own arithmetic.
- Anything that requires an attacker to already be running code in your process.

## Supported versions

The most recent released version is the one that gets fixes. This project has no long-term support branches;
if you are pinned to an older version, say so in the report and you will be told what the upgrade involves.

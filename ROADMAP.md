# Roadmap

What DeepSharp has built and what it will build, in the order the work is done: what is done first, and below it what
is still to come. The one rule behind the order is that nothing stands above a row it needs; among the rows that are free
to move, the one a reader reaches for most often, and pays least for, comes first. There are no dates — a row moves when it
is true, and the [changelog](CHANGELOG.md) is where it says so.

| | What it is | In one line |
|---|---|---|
| **Done** | `Shape`, `Tensor`, `ITensorBackend`, `CpuBackend` | Everything else is written in terms of these. |
| **Done** | [Pipeline](https://github.com/xkqg/DeepSharp/wiki/Pipeline) | The artefact PDD is named after: declared once, fitted on train alone, replayed unchanged. |
| **Done** | [Notebook](https://github.com/xkqg/DeepSharp/wiki/Notebook) | The same pipeline as a Verso notebook, one block per step. |
| **Done** | `DeepSharp.Verso.Api` and `DeepSharp.Verso.Serve` | The notebook hosted in an application of your own, and DeepSharp's own server. |
| **Done** | Automatic differentiation | The gradient a network learns by. |
| **Done** | The layers | Each a forward pass and its gradient, matched against PyTorch on the same rows. |
| **Done** | Losses and optimizers | The parts that turn a gradient into a change. |
| **Done** | The training loop | Batches, epochs, validation, early stopping, checkpoints and history — and a network trained behind a pipeline. |
| **Done** | The two vocabularies | Keras's words or PyTorch's, lowered onto the same model, bit for bit. |
| **Done** | The charts | `DeepSharp.Charts`: what a run keeps, drawn as SVG with MatPlotLibNet. |
| **Done** | A TorchSharp backend | The same model, its arithmetic on libtorch, on the processor or a graphics card. |
| **Done** | Loading what already exists | Reading a model trained elsewhere, from PyTorch, Keras and ONNX files. |
| **Done** | Settling a gap before the features are worked out | `settle.gaps`: a value no row decided, written where the features are worked out. |
| **Done** | A trainer from ML.NET | Boosted trees and a forest, measured beside a network on the same rows. |
| **Done** | A course to fill in | Every step of a pipeline in its order, waiting for what only a person knows. |
| **Done** | [Live sources](https://github.com/xkqg/DeepSharp/wiki/Live-sources) | A source that answers differently each time, fetched once and landed as a file. |
| **Done** | Optimizing | Searching for a network, answers that are a distribution, and what the pipeline owes a network. |
| **Done** | Everything a window can be | Every convolution and pooling setting that used to be refused, built. |
| **Done** | Everything the parts can be told | Every layer and optimizer setting that used to be refused, built. |
| **Done** | What the flock team asked for a third time | Bins, a periodic embedding, how alike the splits are, a better search, averaged weights, a per-key history. |
| **Done** | The vocabulary a model is written in | The activations, normalisations, losses and measures other libraries write. |
| **Done** | A model watched after it ships | The rows a served model meets, measured window by window against the training rows its pipeline keeps. |
| **Later** | Data that is not a table | Images, sound and text read in as rows, varied while training only, and text cut into tokens. |
| **Later** | Several networks, one answer | Networks trained apart — a study's best, or one design under other seeds — answering together. |
| **Later** | The answer and what it is worth | Splits that cut more than once, an interval around the answer, and why a model answers as it does. |
| **Later** | Training further from what was read | A rate of its own for a group of layers, and the low-rank pair beside a frozen layer. |
| **Later** | A network that is not a straight line | Branches, so a residual block, a U-shaped network or two inputs can be written, read and saved. |
| **Later** | A row that remembers | The layers that carry something from one step of a series to the next. |
| **Later** | Attention, and what it attends over | Attention, its many-headed form, the order of a sequence, and one encoder block. |
| **Later** | Writing what others can read | Writing a model in the graph format every runtime reads. |
| **Later** | A table of its own | A model that reads a table a column at a time, measured against ML.NET's trees. |
| **Later** | The engine's own arithmetic | Half-width numbers, a scaled loss and larger batches in the same memory, behind the seam. |
| **Later** | More than one graphics card | One run spread over several cards, by an engine behind the seam that already does it. |
| **Later** | Smaller numbers to serve with | A trained model kept in whole numbers of eight bits, so it answers in less memory and time. |

Every row has a section of its own on the [roadmap in the wiki](https://github.com/xkqg/DeepSharp/wiki/Roadmap): what it
takes, why it stands where it does, and what is left out on purpose. This table is the wiki's, and a test holds the two to
one another.

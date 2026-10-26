# Roadmap

What DeepSharp has built and what it will build, in the order the work is done: what is done first, and below it what
is still to come; at the moment nothing is queued, and a row stands here when it is asked for. The one rule behind the order is that nothing stands above a row it needs; among the rows that are free
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
| **Done** | Training further from what was read | A rate of its own for a group of layers, and the low-rank pair beside a held kernel. |
| **Done** | Data that is not a table | Images, sound and text read in as rows, varied while training only, and text cut into tokens. |
| **Done** | Several networks, one answer | Networks trained apart — a study's best, or one design under other seeds — answering together. |
| **Done** | The answer and what it is worth | Splits that cut more than once, an interval around the answer, and why a model answers as it does. |
| **Done** | What the flock team asked for a fourth time | The light engine's products and Adam's step faster by the very same bits, a rate that follows the training loss, a history over its last few places, and a study that prepares a pipeline for every candidate. |
| **Done** | A network that is not a straight line | Branches: a residual block, a block that reads everything made before it, two inputs and a U, written, read and saved. |
| **Done** | A row that remembers | An LSTM and a GRU, in one direction or both, to the last state or at every step, over series of unequal lengths. |
| **Done** | Attention, and what it attends over | Attention, its many-headed form, the positions of the steps, and one encoder block as PyTorch makes it. |
| **Done** | Writing what others can read | `DeepSharp.Export.Onnx`: a trained network written as an ONNX graph and answered through by ONNX Runtime. |
| **Done** | A table of its own | A model that reads a table a column at a time was built and measured against ML.NET's trees, and not kept. |
| **Done** | The engine's own arithmetic | Numbers kept in half the width with a full-width copy, a scaled loss, a sum of gradients kept wide, and layers worked out again to fit a larger batch. |
| **Done** | More than one graphics card | One run spread over two cards was built and measured against one card taking two batches to a step, and not kept. |
| **Done** | Smaller numbers to serve with | A trained model in whole numbers of eight bits was built and measured against the answers it gave before, and not kept. |
| **Done** | What the flock team asked for a fifth time | A split or a shuffle may rank by a declared id, a study may stop a trial between deals, and a trainer from ML.NET may be fitted and asked to predict without a pipeline. |

Every row has a section of its own on the [roadmap in the wiki](https://github.com/xkqg/DeepSharp/wiki/Roadmap): what it
takes, why it stands where it does, and what is left out on purpose. This table is the wiki's, and a test holds the two to
one another.

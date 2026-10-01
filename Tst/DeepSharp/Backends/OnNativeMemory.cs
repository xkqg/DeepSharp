// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tests.Backends.Contract;
using DeepSharp.Tests.Backends.Parts;

// What every engine is held to, on an engine of the tests' own that differs from the light one in the two ways libtorch
// does: its values live in memory allocated outside .NET, and it adds its totals up in single precision.
namespace DeepSharp.Tests.Backends.NativeMemory;

public sealed class TensorBackendTests() : TensorBackendContract(new NativeMemoryBackend());

public sealed class RefusalTests() : RefusalContract(new NativeMemoryBackend());

public sealed class AgreementTests() : AgreementContract(new NativeMemoryBackend());

public sealed class TitanicStepTests() : TitanicStepContract(new NativeMemoryBackend());

public sealed class RecordingBackendTests() : RecordingBackendContract(new NativeMemoryBackend());

public sealed class CompositionTests() : CompositionContract(new NativeMemoryBackend());

public sealed class DenseTests() : DenseContract(new NativeMemoryBackend());

public sealed class DropoutTests() : DropoutContract(new NativeMemoryBackend());

public sealed class ConvolutionTests() : ConvolutionContract(new NativeMemoryBackend());

public sealed class NormalisationTests() : NormalisationContract(new NativeMemoryBackend());

public sealed class LossTests() : LossContract(new NativeMemoryBackend());

public sealed class LayerTests() : LayerContract(new NativeMemoryBackend());

public sealed class OptimizerTests() : OptimizerContract(new NativeMemoryBackend());

public sealed class SequentialTests() : SequentialContract(new NativeMemoryBackend());

public sealed class LoopTests() : LoopContract(new NativeMemoryBackend());

public sealed class NetworkDocumentTests() : NetworkDocumentContract(new NativeMemoryBackend());

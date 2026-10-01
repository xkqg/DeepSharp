// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Contract;

// What every engine is held to, on the light engine: the one that ships in the core, which every other is measured against.
namespace DeepSharp.Tests.Backends.Cpu;

public sealed class TensorBackendTests() : TensorBackendContract(new CpuBackend());

public sealed class RefusalTests() : RefusalContract(new CpuBackend());

public sealed class AgreementTests() : AgreementContract(new CpuBackend());

public sealed class TitanicStepTests() : TitanicStepContract(new CpuBackend());

public sealed class RecordingBackendTests() : RecordingBackendContract(new CpuBackend());

public sealed class CompositionTests() : CompositionContract(new CpuBackend());

public sealed class DenseTests() : DenseContract(new CpuBackend());

public sealed class DropoutTests() : DropoutContract(new CpuBackend());

public sealed class ConvolutionTests() : ConvolutionContract(new CpuBackend());

public sealed class NormalisationTests() : NormalisationContract(new CpuBackend());

public sealed class LossTests() : LossContract(new CpuBackend());

public sealed class LayerTests() : LayerContract(new CpuBackend());

public sealed class OptimizerTests() : OptimizerContract(new CpuBackend());

public sealed class SequentialTests() : SequentialContract(new CpuBackend());

public sealed class LoopTests() : LoopContract(new CpuBackend());

public sealed class NetworkDocumentTests() : NetworkDocumentContract(new CpuBackend());

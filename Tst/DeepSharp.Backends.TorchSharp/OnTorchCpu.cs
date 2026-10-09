// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Backends.TorchSharp;
using DeepSharp.Tests.Backends.Contract;

// What every engine is held to, on libtorch on this machine's processor: the engine an application brings libtorch for.
namespace DeepSharp.Tests.Backends.Torch;

public sealed class TensorBackendTests() : TensorBackendContract(TorchBackend.OnCpu());

public sealed class RefusalTests() : RefusalContract(TorchBackend.OnCpu());

public sealed class AgreementTests() : AgreementContract(TorchBackend.OnCpu());

public sealed class TitanicStepTests() : TitanicStepContract(TorchBackend.OnCpu());

public sealed class RecordingBackendTests() : RecordingBackendContract(TorchBackend.OnCpu());

public sealed class CompositionTests() : CompositionContract(TorchBackend.OnCpu());

public sealed class DenseTests() : DenseContract(TorchBackend.OnCpu());

public sealed class DropoutTests() : DropoutContract(TorchBackend.OnCpu());

public sealed class ConvolutionTests() : ConvolutionContract(TorchBackend.OnCpu());

public sealed class SpatialConvolutionTests() : SpatialConvolutionContract(TorchBackend.OnCpu());

public sealed class PoolingTests() : PoolingContract(TorchBackend.OnCpu());

public sealed class SpatialDropoutTests() : SpatialDropoutContract(TorchBackend.OnCpu());

public sealed class NormalisationTests() : NormalisationContract(TorchBackend.OnCpu());

public sealed class LossTests() : LossContract(TorchBackend.OnCpu());

public sealed class LayerTests() : LayerContract(TorchBackend.OnCpu());

public sealed class OptimizerTests() : OptimizerContract(TorchBackend.OnCpu());

public sealed class SequentialTests() : SequentialContract(TorchBackend.OnCpu());

public sealed class LoopTests() : LoopContract(TorchBackend.OnCpu());

public sealed class NetworkDocumentTests() : NetworkDocumentContract(TorchBackend.OnCpu());

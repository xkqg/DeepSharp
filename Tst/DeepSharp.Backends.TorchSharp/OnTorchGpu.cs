// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tests.Backends.Contract;

// What every engine is held to, on libtorch on the first graphics card: run where the suite is built with -p:Libtorch=cuda on
// a machine with an NVIDIA card, and skipped everywhere else.
namespace DeepSharp.Tests.Backends.TorchGpu;

public sealed class TensorBackendTests() : TensorBackendContract(Card.First());

public sealed class RefusalTests() : RefusalContract(Card.First());

public sealed class AgreementTests() : AgreementContract(Card.First());

public sealed class TitanicStepTests() : TitanicStepContract(Card.First());

public sealed class RecordingBackendTests() : RecordingBackendContract(Card.First());

public sealed class CompositionTests() : CompositionContract(Card.First());

public sealed class DenseTests() : DenseContract(Card.First());

public sealed class DropoutTests() : DropoutContract(Card.First());

public sealed class ConvolutionTests() : ConvolutionContract(Card.First());

public sealed class SpatialConvolutionTests() : SpatialConvolutionContract(Card.First());

public sealed class PoolingTests() : PoolingContract(Card.First());

public sealed class SpatialDropoutTests() : SpatialDropoutContract(Card.First());

public sealed class NormalisationTests() : NormalisationContract(Card.First());

public sealed class LossTests() : LossContract(Card.First());

public sealed class LayerTests() : LayerContract(Card.First());

public sealed class OptimizerTests() : OptimizerContract(Card.First());

public sealed class SequentialTests() : SequentialContract(Card.First());

public sealed class LoopTests() : LoopContract(Card.First());

public sealed class NetworkDocumentTests() : NetworkDocumentContract(Card.First());

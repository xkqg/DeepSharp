// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tests.Backends.Contract;
using DeepSharp.Tests.Backends.Parts;

// An engine of somebody else's, whose arithmetic is the light engine's on a storage of its own: held to what the seam
// refuses, in the light engine's words.
namespace DeepSharp.Tests.Backends.CopyCounting;

public sealed class RefusalTests() : RefusalContract(new CopyCountingBackend());

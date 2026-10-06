// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using Xunit.Sdk;
using Xunit.v3;

[assembly: TestPipelineStartup(typeof(DeepSharp.Tests.NoVenue))]

namespace DeepSharp.Tests;

/// <summary>
/// No test reaches the network. The addresses of an exchange are the only ones a program a suite runs could fetch from, and
/// the program would be a guest of somebody else's budget with nothing to say it was: so before any test is found or run,
/// every address but the machine's own is sent to a proxy nothing listens at, and a fetch that was not stood in front of by a
/// stand-in finds nothing, rather than the exchange.
/// </summary>
/// <remarks>
/// Set for the whole assembly, early, because discovery runs code too. A stand-in that listens on this machine is still
/// answered: the proxy bypasses it. This one file is compiled into the suites that run documents, as the four-parameter
/// rule's is into every suite.
/// </remarks>
public sealed class NoVenue : ITestPipelineStartup
{
    /// <inheritdoc />
    public ValueTask StartAsync(IMessageSink diagnosticMessageSink)
    {
        HttpClient.DefaultProxy = new WebProxy("http://127.0.0.1:9", BypassOnLocal: true);

        return default;
    }

    /// <inheritdoc />
    public ValueTask StopAsync() => default;
}

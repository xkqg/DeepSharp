// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Serve;

/// <summary><c>deepsharp-serve</c>: DeepSharp's own server, the notebook in your browser.</summary>
public static class Program
{
    /// <summary>Runs the server with the command line it was started with, until Ctrl+C stops it.</summary>
    /// <param name="args">The command line's words.</param>
    /// <returns>0 once it stopped, or said its usage; 1 when it could not listen; 2 when its command line was refused.</returns>
    public static Task<int> Main(string[] args) => NotebookServer.RunAsync(args, Console.Out, Console.Error, CancellationToken.None);
}

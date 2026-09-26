// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Verso.Serve;

/// <summary>What <c>deepsharp-serve</c> is told on its command line.</summary>
/// <param name="Path">The notebook to serve, or the folder of notebooks, as a full path.</param>
/// <param name="Port">The port to listen on, on this computer alone; 0 lets the system pick one.</param>
/// <param name="OpenBrowser">Whether a browser is opened on the address once the server listens.</param>
/// <param name="Help">Whether only the usage was asked for.</param>
public readonly record struct ServeOptions(string Path, int Port, bool OpenBrowser, bool Help)
{
    /// <summary>How to use it.</summary>
    public const string Usage = """
        Usage: deepsharp-serve [notebook | folder] [--port N] [--no-browser]

          notebook | folder   the notebook to serve, or a folder of notebooks; the folder it runs in when neither is named
          --port N            the port to listen on, on this computer alone; one the system picks when none is named
          --no-browser        say the address and open no browser, for a computer you reach from another
          --help              say this
        """;

    /// <summary>
    /// How long a notebook no page has shown stays open, when nothing runs in it and nothing in it is unsaved: longer
    /// than a page takes to load again. A minute, unless an application that builds the server says otherwise.
    /// </summary>
    public TimeSpan Grace { get; init; } = TimeSpan.FromMinutes(1);

    /// <summary>Reads a command line.</summary>
    /// <param name="args">The command line's words.</param>
    /// <param name="workingDirectory">The folder the tool runs in, which a path is read from.</param>
    /// <returns>What it said.</returns>
    /// <exception cref="ArgumentException">It said something the tool cannot use; the message says what.</exception>
    public static ServeOptions Parse(IReadOnlyList<string> args, string workingDirectory)
    {
        ArgumentNullException.ThrowIfNull(args);

        string? path = null;
        var port = 0;
        var browser = true;
        var help = false;

        for (var at = 0; at < args.Count; at++)
        {
            switch (args[at])
            {
                case "--help":
                    help = true;
                    break;
                case "--no-browser":
                    browser = false;
                    break;
                case "--port":
                    port = ++at < args.Count ? PortOf(args[at]) : throw new ArgumentException("--port needs a number after it.");
                    break;
                case var flag when flag.StartsWith("--", StringComparison.Ordinal):
                    throw new ArgumentException($"'{flag}' is not an option deepsharp-serve knows.");
                case var named:
                    path = path is null ? Existing(named, workingDirectory) : throw new ArgumentException($"'{named}' is a second notebook or folder; deepsharp-serve serves one.");
                    break;
            }
        }

        return new ServeOptions(path ?? workingDirectory, port, browser, help);
    }

    private static int PortOf(string word) =>
        int.TryParse(word, NumberStyles.None, CultureInfo.InvariantCulture, out var port) && port <= 65535
            ? port
            : throw new ArgumentException($"'{word}' is no port: a port is a number from 0 to 65535.");

    private static string Existing(string named, string workingDirectory)
    {
        var path = System.IO.Path.GetFullPath(named, workingDirectory);

        return File.Exists(path) || Directory.Exists(path) ? path : throw new ArgumentException($"'{named}' is neither a notebook nor a folder.");
    }
}

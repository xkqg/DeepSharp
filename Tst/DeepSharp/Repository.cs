// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests;

/// <summary>
/// The repository these tests run inside: its root, and the published data the samples read.
/// </summary>
/// <remarks>
/// Found by walking up from the running test to the folder that holds the solution, so the tests read the
/// same files whether they run from an IDE, from the command line or on a build machine. It was written
/// out eighteen times, once per test class, which is eighteen chances for one of them to look for a
/// different file.
/// </remarks>
internal static class Repository
{
    /// <summary>The folder that holds the solution.</summary>
    public static string Root { get; } = FindRoot();

    /// <summary>A file among the published data the samples read.</summary>
    /// <param name="file">The file's name, such as <c>titanic.csv</c>.</param>
    /// <returns>Its full path.</returns>
    public static string Data(string file) => Path.Join(Root, "Samples", "data", file);

    private static string FindRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "DeepSharp.slnx")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName
               ?? throw new InvalidOperationException("These tests are not running inside the repository they test.");
    }
}

// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using System.Text.Json;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// What Verso's own installer lays out for the notebook package, beside what this suite's build resolves for it. The list
/// is the one <c>tools/verso/check.sh</c> holds a real install to — the package just made, installed by Verso's installer
/// on each runtime a Verso host runs on — so what the package claims and what Verso installs are held to one list.
/// </summary>
internal static class VersoInstall
{
    // The runtime this suite's build is for, named as Verso names the folder it installs a package into for that runtime.
    private static readonly string Runtime = new DirectoryInfo(Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory)).Name;

    /// <summary>The list of what Verso's installer lays out for the package on this suite's runtime.</summary>
    public static string List => Path.Join(Repository.Root, "tools", "verso", $"installed.{Runtime}.txt");

    /// <summary>The files Verso's installer lays out for the package on this suite's runtime, by name, in ordinal order.</summary>
    public static IReadOnlyList<string> Listed => [.. File.ReadLines(List).Where(line => line.Length > 0).Order(StringComparer.Ordinal)];

    /// <summary>
    /// The assemblies this suite's build resolves for the package — its own and every one it depends on, by file name, in
    /// ordinal order — read from the dependencies the build wrote for this suite, starting at the package's project.
    /// </summary>
    public static IReadOnlyList<string> Resolved
    {
        get
        {
            var suite = typeof(VersoInstall).Assembly.GetName().Name;
            using var dependencies = JsonDocument.Parse(File.ReadAllText(Path.Join(AppContext.BaseDirectory, $"{suite}.deps.json")));
            var libraries = dependencies.RootElement.GetProperty("targets").EnumerateObject().First().Value.EnumerateObject()
                .ToDictionary(library => library.Name[..library.Name.IndexOf('/', StringComparison.Ordinal)], library => library.Value);
            var files = new SortedSet<string>(StringComparer.Ordinal);
            var reached = new HashSet<string>();
            var next = new Stack<string>(["DeepSharp.Verso.Notebooks"]);

            // A dependency the build left to the runtime has no library of its own here, and nothing to walk into.
            while (next.TryPop(out var name))
            {
                if (!reached.Add(name) || !libraries.TryGetValue(name, out var library))
                {
                    continue;
                }

                if (library.TryGetProperty("runtime", out var runtime))
                {
                    files.UnionWith(runtime.EnumerateObject().Select(file => Path.GetFileName(file.Name)));
                }

                if (library.TryGetProperty("dependencies", out var named))
                {
                    foreach (var dependency in named.EnumerateObject())
                    {
                        next.Push(dependency.Name);
                    }
                }
            }

            return [.. files];
        }
    }

    /// <summary>Whether the runtime this suite runs on carries a library itself.</summary>
    /// <param name="file">The library's file name.</param>
    /// <returns>Whether the runtime's own folder holds it.</returns>
    public static bool CarriedByTheRuntime(string file) => File.Exists(Path.Join(RuntimeEnvironment.GetRuntimeDirectory(), file));

    /// <summary>
    /// Lays the package out in a folder as Verso's installer lays it out: every file the list names, as this suite's build
    /// has it, and nothing else — a file an earlier layout left there is taken away first. A library the build leaves to the
    /// runtime is not laid out, as the runtime carries it.
    /// </summary>
    /// <remarks>
    /// This lays the files out; it does not install them. Loaded in this process, whose own context holds every one of
    /// them, a file missing from the list would be found all the same: only <c>tools/verso/check.sh</c>, which has Verso's
    /// installer install the package in a host holding nothing of DeepSharp, can see that.
    /// </remarks>
    /// <param name="folder">The folder, made when it is not there.</param>
    /// <returns>The notebook's own assembly in it.</returns>
    public static string LayOut(string folder)
    {
        var listed = Listed;

        Directory.CreateDirectory(folder);

        foreach (var stale in Directory.GetFiles(folder).Where(file => !listed.Contains(Path.GetFileName(file))))
        {
            File.Delete(stale);
        }

        foreach (var file in listed.Where(file => File.Exists(Path.Join(AppContext.BaseDirectory, file))))
        {
            File.Copy(Path.Join(AppContext.BaseDirectory, file), Path.Join(folder, file), overwrite: true);
        }

        return Path.Join(folder, "DeepSharp.Verso.Notebooks.dll");
    }
}

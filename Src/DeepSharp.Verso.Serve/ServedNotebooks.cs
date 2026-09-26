// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Extensions;

namespace DeepSharp.Verso.Serve;

/// <summary>
/// The notebooks a server serves: the one it was started beside, or the files of the folder it was started in that
/// Verso's engine reads — asked of the engine itself, so a format it learns is served without a word here.
/// </summary>
/// <param name="options">What the command line said.</param>
internal sealed class ServedNotebooks(ServeOptions options) : IAsyncDisposable
{
    // Verso's engine with its own parts, made when it is first asked which files it reads.
    private readonly Lazy<Task<ExtensionHost>> _engine = new(async () =>
    {
        var engine = new ExtensionHost();

        await engine.LoadBuiltInExtensionsAsync();

        return engine;
    });

    private string Folder => Directory.Exists(options.Path) ? options.Path : System.IO.Path.GetDirectoryName(options.Path)!;

    /// <summary>The names of the notebooks served, in order: file names, never paths.</summary>
    /// <returns>The names.</returns>
    public async Task<IReadOnlyList<string>> NamesAsync()
    {
        if (File.Exists(options.Path))
        {
            return [System.IO.Path.GetFileName(options.Path)];
        }

        var serializers = (await _engine.Value).GetSerializers();

        return
        [
            .. Directory.EnumerateFiles(Folder)
                .Where(file => serializers.Any(serializer => serializer.CanImport(file)))
                .Select(file => System.IO.Path.GetFileName(file))
                .Order(StringComparer.Ordinal),
        ];
    }

    /// <summary>The file a name is, when it is the name of a notebook served; nothing for any other name.</summary>
    /// <param name="name">The name a page asked for.</param>
    /// <returns>The file, or nothing.</returns>
    public async Task<string?> FileOfAsync(string name) =>
        (await NamesAsync()).Contains(name, StringComparer.Ordinal) ? System.IO.Path.Join(Folder, name) : null;

    /// <summary>Closes the engine, when it was ever asked.</summary>
    /// <returns>When it is closed.</returns>
    public async ValueTask DisposeAsync()
    {
        if (_engine.IsValueCreated)
        {
            await (await _engine.Value).DisposeAsync();
        }
    }
}

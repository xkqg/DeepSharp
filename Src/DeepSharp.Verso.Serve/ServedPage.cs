// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Net.Http.Headers;

namespace DeepSharp.Verso.Serve;

/// <summary>
/// The page the server serves: its own markup, with what it draws diagrams and formulas with carried inside it — Mermaid,
/// and KaTeX with its stylesheet and fonts — held as text the page runs only once a diagram or a formula first shows, so
/// the page fetches nothing from anywhere; and a tag that names it, so a browser that has it already is only told it is
/// unchanged.
/// </summary>
/// <remarks>
/// Each library is held as a JSON string in a script the browser does not run, written with every character HTML reads as
/// markup escaped, so nothing a library holds can end the script it stands in. KaTeX's stylesheet names its fonts as files
/// beside it; the page writes each into it instead, as the woff2 face every browser the page runs in reads.
/// </remarks>
internal sealed partial class ServedPage
{
    // Where the carried libraries go in the page's markup.
    private const string Marker = "<!-- carried -->";

    private readonly byte[] _bytes;
    private readonly EntityTagHeaderValue _tag;

    private ServedPage(byte[] bytes)
    {
        _bytes = bytes;
        _tag = new EntityTagHeaderValue($"\"{Convert.ToHexString(SHA256.HashData(bytes))}\"");
    }

    /// <summary>The page, put together from what the tool carries.</summary>
    /// <returns>The page.</returns>
    public static ServedPage Carrying()
    {
        var carried = string.Concat(
            Held("carried-mermaid", Text("carried/mermaid/mermaid.min.js")),
            Held("carried-katex", Text("carried/katex/katex.min.js")),
            Held("carried-katex-css", Face().Replace(Text("carried/katex/katex.min.css"), face => Written(face.Groups["font"].Value))));

        return new ServedPage(Encoding.UTF8.GetBytes(Text("page.html").Replace(Marker, carried, StringComparison.Ordinal)));
    }

    /// <summary>
    /// The page as a browser is answered it: named by its tag, and asked for again each time it is shown, so a browser that
    /// has it already is told only that it is unchanged.
    /// </summary>
    /// <param name="context">The request.</param>
    /// <returns>The answer.</returns>
    public IResult Answer(HttpContext context)
    {
        context.Response.Headers.CacheControl = "no-cache";

        return Results.Bytes(_bytes, "text/html; charset=utf-8", entityTag: _tag);
    }

    // A text held in the page, and never run by it until the page asks for it.
    private static string Held(string id, string text) => $"<script type=\"application/json\" id=\"{id}\">{JsonSerializer.Serialize(text)}</script>\n";

    // A font face KaTeX names as a file beside its stylesheet, written into the stylesheet as the woff2 face.
    private static string Written(string font) => $"url(data:font/woff2;base64,{Convert.ToBase64String(Bytes($"carried/katex/fonts/{font}.woff2"))}) format(\"woff2\")";

    private static string Text(string name) => Encoding.UTF8.GetString(Bytes(name));

    private static byte[] Bytes(string name)
    {
        using var carried = typeof(ServedPage).Assembly.GetManifestResourceStream(name)!;
        using var read = new MemoryStream();

        carried.CopyTo(read);

        return read.ToArray();
    }

    // Each face KaTeX's stylesheet draws, as it names its three files.
    [GeneratedRegex(@"url\(fonts/(?<font>[\w-]+)\.woff2\) format\(""woff2""\),url\(fonts/\k<font>\.woff\) format\(""woff""\),url\(fonts/\k<font>\.ttf\) format\(""truetype""\)")]
    private static partial Regex Face();
}

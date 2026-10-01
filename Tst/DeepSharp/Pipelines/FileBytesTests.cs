// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A file's bytes, once they are in hand, read as text the way the file itself reads as text: in the encoding its
/// byte-order mark names, UTF-8 when it names none. So rows parsed from bytes that were read once to be fingerprinted are
/// the rows the file reads as, whichever encoding it was written in.
/// </summary>
public sealed class FileBytesTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-bytes-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    public static TheoryData<string> EveryEncodingAFileSays() =>
        new() { "UTF-8 without a mark", "UTF-8 with its mark", "UTF-16 LE", "UTF-16 BE" };

    private static Encoding Named(string encoding) => encoding switch
    {
        "UTF-8 without a mark" => new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
        "UTF-8 with its mark" => new UTF8Encoding(encoderShouldEmitUTF8Identifier: true),
        "UTF-16 LE" => new UnicodeEncoding(bigEndian: false, byteOrderMark: true),
        _ => new UnicodeEncoding(bigEndian: true, byteOrderMark: true),
    };

    [Theory]
    [MemberData(nameof(EveryEncodingAFileSays))]
    public void AFilesBytes_AsText_AreTheTextTheFileReadsAs(string encoding)
    {
        var written = Named(encoding);
        byte[] bytes = [.. written.GetPreamble(), .. written.GetBytes("name,town\nZoë,Cherbourg\nJosé,Queenstown\n")];
        var path = Path.Join(_folder, "passengers.csv");

        File.WriteAllBytes(path, bytes);

        Assert.Equal(File.ReadAllText(path), bytes.AsText());
        Assert.StartsWith("name,town\nZoë", bytes.AsText(), StringComparison.Ordinal);
    }

    [Fact]
    public void NoBytes_AreRefused()
    {
        Assert.Throws<ArgumentNullException>(() => ((byte[])null!).AsText());
    }
}

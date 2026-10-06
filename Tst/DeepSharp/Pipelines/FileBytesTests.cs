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

    [Theory]
    [InlineData("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")]
    [InlineData("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")]
    public void AFilesBytes_Fingerprint_IsTheLowercaseSha256OfThem(string text, string expected)
    {
        Assert.Equal(expected, Encoding.ASCII.GetBytes(text).Fingerprint());
    }

    [Fact]
    public void ManyBytes_Fingerprint_IsWhatTheSystemsOwnHashOfTheFileSays()
    {
        var bytes = new byte[10_000];

        new Random(20261006).NextBytes(bytes);
        File.WriteAllBytes(Path.Join(_folder, "random.bin"), bytes);

        Assert.Equal(Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(Path.Join(_folder, "random.bin")))).ToLowerInvariant(), bytes.Fingerprint());
    }

    [Fact]
    public void NoBytes_AreNotFingerprinted()
    {
        Assert.Throws<ArgumentNullException>(() => ((byte[])null!).Fingerprint());
    }
}

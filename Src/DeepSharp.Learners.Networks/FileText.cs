// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;
using DeepSharp.Networks;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The top of the one file, surveyed before any part of it is read: that it is JSON, one object, of a version this library
/// reads, holding its parts and nothing else — and where each part stands, so a fault about a part is placed at it.
/// </summary>
/// <remarks>
/// Each part is read by what reads it — the network by the network document, the pipeline as its own file is — and each
/// places its own faults; this places the file's.
/// </remarks>
internal sealed class FileText
{
    private static readonly string[] Parts = [TrainedNetwork.VersionKey, TrainedNetwork.NetworkKey, TrainedNetwork.PipelineKey, TrainedNetwork.TrainingKey];

    private readonly byte[] _bytes;
    private readonly int[] _lineStarts;
    private readonly Dictionary<string, int> _places = new(StringComparer.Ordinal);

    /// <summary>Surveys a file's top.</summary>
    /// <exception cref="NetworkFileException">The text is not JSON, not one object, of no version this library reads, or holds what is no part of it.</exception>
    public FileText(string json)
    {
        _bytes = Encoding.UTF8.GetBytes(json);
        _lineStarts = LineStarts(_bytes);

        var faults = Survey();

        if (faults.Count > 0)
        {
            throw new NetworkFileException(faults);
        }
    }

    /// <summary>A refusal of the file, placed at the part a key stands for.</summary>
    /// <remarks>Only ever asked of a part its reader has read, so the key stands in the file.</remarks>
    public NetworkFileException Refused(string key, string message) => new([Placed(_places[key], message)]);

    private List<NetworkFileFault> Survey()
    {
        var faults = new List<NetworkFileFault>();
        var reader = new Utf8JsonReader(_bytes);
        int? version = null;

        try
        {
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            {
                faults.Add(Placed(0, "A trained network's file is one JSON object: its version, its network and the pipeline it was trained behind."));

                return faults;
            }

            while (reader.Read() && reader.CurrentDepth > 0)
            {
                var key = reader.GetString()!;
                var at = (int)reader.TokenStartIndex;

                _places.TryAdd(key, at);
                reader.Read();

                if (key == TrainedNetwork.VersionKey)
                {
                    version = reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out var number) ? number : 0;
                }

                reader.Skip();

                if (!Parts.Contains(key))
                {
                    faults.Add(Placed(at, $"'{key}' is no part of a trained network's file: it holds its version, its network, the pipeline it was trained behind and, for a checkpoint, what the run needs to go on."));
                }
            }
        }
        catch (JsonException fault)
        {
            var line = (int)Math.Min(fault.LineNumber.GetValueOrDefault(), _lineStarts.Length - 1);

            faults.Add(Placed(
                (int)Math.Min(_lineStarts[line] + fault.BytePositionInLine.GetValueOrDefault(), _bytes.Length),
                $"The text stops being JSON here: {fault.Message.Split(" LineNumber:")[0]}"));

            return faults;
        }

        if (version is not >= 1)
        {
            faults.Insert(0, Placed(
                _places.GetValueOrDefault(TrainedNetwork.VersionKey),
                $"A trained network's file names the whole number of the version it was written against, from 1, under '{TrainedNetwork.VersionKey}'."));
        }
        else if (version > TrainedNetwork.Version)
        {
            faults.Insert(0, Placed(
                _places[TrainedNetwork.VersionKey],
                string.Create(CultureInfo.InvariantCulture, $"This file was written against version {version}, by a newer DeepSharp than this one, which reads up to version {TrainedNetwork.Version}: read it with that DeepSharp.")));
        }

        return faults;
    }

    // A fault at its line and column, both from one; the column in characters rather than bytes.
    private NetworkFileFault Placed(int offset, string message)
    {
        var line = Array.BinarySearch(_lineStarts, offset);

        if (line < 0)
        {
            line = ~line - 1;
        }

        return new NetworkFileFault(line + 1, Encoding.UTF8.GetCharCount(_bytes, _lineStarts[line], offset - _lineStarts[line]) + 1, message);
    }

    private static int[] LineStarts(byte[] text)
    {
        var starts = new List<int> { 0 };

        for (var at = 0; at < text.Length; at++)
        {
            if (text[at] == (byte)'\n')
            {
                starts.Add(at + 1);
            }
        }

        return [.. starts];
    }
}

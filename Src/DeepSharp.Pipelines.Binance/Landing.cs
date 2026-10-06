// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>
/// A window landed in a folder, once: what stands there decides what is done, and nothing that stands there is ever replaced.
/// </summary>
/// <remarks>
/// <para>
/// By a table. Nothing stands: the window is walked and landed. The file and its record both stand: the bytes are read once and
/// held to the fingerprint the record names, and reused — the venue is not asked. A record stands without its file, a file's
/// bytes are not the record's, a record is of a newer version, cannot be read or names another window: refused, touching
/// nothing, since the first landing is what a pipeline may already have been fitted behind. A file stands without its record,
/// as when a program stopped between the two writes: the window is walked again, and the record is written only if the venue
/// said the same; other numbers are refused and the first landing stays.
/// </para>
/// <para>
/// The move of the file is the commit: the file is written whole under a name of its own and moved into place without
/// replacing anything, so of two landings of one window racing each other one stands and the other either finds the same
/// bytes — and is the same landing — or is refused. That holds within a process on every system; between processes the
/// system's own refusal decides, and where it is a look and then a rename, a landing replaced all the same is caught by
/// the fingerprint its record names. The record is written after the commit, and not under the caller's
/// cancellation: a landing that has been moved into place is finished with, and a record is what says what it was. Nothing is
/// written before the whole window has been walked, so a landing that was cancelled or stopped leaves nothing behind.
/// </para>
/// </remarks>
/// <param name="ask">The window.</param>
/// <param name="folder">Where the landing goes.</param>
internal sealed class Landing(BinanceCandles ask, SourceFolder folder)
{
    private readonly string _file = folder.Resolve(ask.LandingName);
    private readonly string _record = folder.Resolve(ask.ManifestName);

    /// <summary>Lands the window if it is not landed, and reuses it if it is.</summary>
    /// <param name="cancellation">Ends the landing before the file is moved into place.</param>
    /// <returns>The name of the file, relative to the folder.</returns>
    /// <exception cref="DirectoryNotFoundException">The folder is not there.</exception>
    /// <exception cref="InvalidDataException">What stands in the folder is not a landing of this window that can be reused, or says another.</exception>
    public async Task<string> RunAsync(CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();

        if (!Directory.Exists(Path.GetDirectoryName(_file)))
        {
            throw new DirectoryNotFoundException($"The folder '{Path.GetDirectoryName(_file)}' that the landing goes in is not there; nothing is asked of Binance until it is.");
        }

        var fileStands = File.Exists(_file);

        if (File.Exists(_record))
        {
            return fileStands
                ? await ReuseAsync(cancellation)
                : throw new InvalidDataException($"'{_record}' says a landing was made, and its file '{ask.LandingName}' is not there. Nothing is fetched to take its place: put the file back, or remove the record to land the window afresh.");
        }

        var walked = await ask.WalkAsync(cancellation);
        var bytes = walked.Candles.AsCsv();

        await CommitAsync(bytes, cancellation);
        await RecordAsync(walked, bytes);

        return ask.LandingName;
    }

    private async Task<string> ReuseAsync(CancellationToken cancellation)
    {
        LandingManifest record;

        try
        {
            record = LandingManifest.FromJson(await File.ReadAllTextAsync(_record, cancellation));
        }
        catch (FormatException unreadable)
        {
            throw new InvalidDataException($"The record '{_record}' of a landing cannot be read: {unreadable.Message}", unreadable);
        }

        if (record.File != ask.LandingName || record.Symbol != ask.Symbol || record.Interval != ask.Interval || record.From != ask.From || record.To != ask.To)
        {
            throw new InvalidDataException($"The record '{_record}' is of another window than the one asked for: it names {record.Symbol} at {record.Interval} in a file called '{record.File}'.");
        }

        var actual = (await File.ReadAllBytesAsync(_file, cancellation)).Fingerprint();

        return actual == record.Fingerprint
            ? ask.LandingName
            : throw new InvalidDataException(
                $"'{_file}' holds other bytes than its record names: the record says {record.Fingerprint} and the file is {actual}. It was edited or torn, and it is left as it is; a pipeline fitted behind it would have learned from other numbers.");
    }

    // The move of the file is what lands it. Other bytes standing there already are somebody else's landing, which stays.
    private async Task CommitAsync(byte[] bytes, CancellationToken cancellation)
    {
        try
        {
            await bytes.WriteWholeOnceAsync(_file, cancellation);
        }
        catch (IOException failure) when (File.Exists(_file))
        {
            // The write has already taken the same bytes standing for the same landing, so a refusal with a file standing
            // there is other bytes. A move refused with nothing standing is a folder's own trouble, and passes as it is.
            throw new InvalidDataException(
                $"'{_file}' stands already, and holds other bytes than the venue answered with now: the first stays, and nothing is written beside it. Remove it, or land another window.", failure);
        }
    }

    // Written after the commit and not under the caller's cancellation. A record that stands already is accepted when it
    // names these same bytes: it is the record of the landing that this one found equal.
    private async Task RecordAsync(Walked walked, byte[] bytes)
    {
        var fingerprint = bytes.Fingerprint();
        var rows = walked.Candles.Count;
        var record = new LandingManifest(
            ask.Host,
            ask.Symbol,
            ask.Interval,
            ask.From,
            ask.To,
            walked.AskedAt,
            walked.ServerTime,
            (int)LandingPace.SettleAfterClose.TotalSeconds,
            walked.PagesEstimated,
            walked.Pages,
            rows,
            (int)(ask.Length.Candles(ask.From, ask.To) - rows),
            DateTimeOffset.FromUnixTimeMilliseconds(walked.Candles[0].OpenTime).UtcDateTime,
            DateTimeOffset.FromUnixTimeMilliseconds(walked.Candles[^1].OpenTime).UtcDateTime,
            ask.LandingName,
            fingerprint);

        try
        {
            await Encoding.UTF8.GetBytes(record.ToJson()).WriteWholeOnceAsync(_record, CancellationToken.None);
        }
        catch (IOException)
        {
            // The record that stands is this landing's own when it names these same bytes — another landing of the window
            // wrote it — and any other refusal is the folder's own trouble, and passes as it is.
            if (!StandsFor(fingerprint))
            {
                throw;
            }
        }
    }

    private bool StandsFor(string fingerprint)
    {
        try
        {
            return File.Exists(_record) && LandingManifest.FromJson(File.ReadAllText(_record)) is { } standing && standing.File == ask.LandingName && standing.Fingerprint == fingerprint;
        }
        catch (FormatException)
        {
            return false;
        }
    }
}

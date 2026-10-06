// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Reading candles from Binance: a window landed as a file, once, and read like any other.
/// </summary>
/// <remarks>
/// <para>
/// A source that answers differently every time it is asked cannot be read while a pipeline trains: the pipeline would train
/// on other numbers tomorrow while claiming to be the same pipeline. So the window is fetched once, over a closed window, and
/// kept as a file beside a record of what was asked and what came back, and the pipeline reads that file — an ordinary
/// <c>read.csv</c>, which is all the pipeline's file ever says of it. The declaration names the window, the landing names the
/// bytes.
/// </para>
/// <para>
/// This package brings the HTTP, the retries and the pacing, and a project that only serves a trained model never needs it:
/// the file such a pipeline saves holds a read of a landed file and nothing of Binance, so it is read, and served, by the
/// library alone.
/// </para>
/// </remarks>
public static class BinanceSourceExtensions
{
    extension(PipelineBuilder pipeline)
    {
        /// <summary>
        /// Lands a window of Binance's candles in the working directory, unless it is landed already, and declares that the rows
        /// come from the file.
        /// </summary>
        /// <param name="candles">The window: a symbol, an interval and a closed span of time.</param>
        /// <param name="cancellation">Ends the landing before anything is written.</param>
        /// <returns>The pipeline, with the read of the landing as its source, so the next verb can be written after it.</returns>
        /// <exception cref="ArgumentNullException">There is no pipeline or no window.</exception>
        /// <exception cref="BinanceException">Binance stopped the landing, asked it to wait longer than a landing waits, or refused it in its own words.</exception>
        /// <exception cref="InvalidOperationException">The window ends where Binance has not yet closed its last candle, or Binance holds no candle of it.</exception>
        /// <exception cref="InvalidDataException">What stands in the working directory under the landing's name is not a landing of this window that can be reused.</exception>
        /// <remarks>
        /// <para>
        /// Awaited, since landing a window is waiting — write it as
        /// <c>(await pipeline.ReadBinanceAsync(new BinanceCandles("BTCEUR", "1d", from, to))).Declare(...)</c>. The first run
        /// asks Binance, paced and within the share of its budget a one-off may take; every run after reads the landing and
        /// asks nothing. The window is the file's name, so another window is another file and a landing is never replaced.
        /// </para>
        /// <para>
        /// A relative path is read, when the pipeline runs, from the working directory for a pipeline written in code, which is
        /// where this lands it. The file is Binance's data, under Binance's terms, and not this library's.
        /// </para>
        /// </remarks>
        public Task<PipelineBuilder> ReadBinanceAsync(BinanceCandles candles, CancellationToken cancellation = default) =>
            pipeline.ReadBinanceAsync(candles, SourceFolder.WorkingDirectory, cancellation);

        internal async Task<PipelineBuilder> ReadBinanceAsync(BinanceCandles candles, SourceFolder folder, CancellationToken cancellation)
        {
            ArgumentNullException.ThrowIfNull(pipeline);
            ArgumentNullException.ThrowIfNull(candles);
            ArgumentNullException.ThrowIfNull(folder);

            return pipeline.ReadCsv(await candles.LandAsync(folder, cancellation));
        }
    }
}

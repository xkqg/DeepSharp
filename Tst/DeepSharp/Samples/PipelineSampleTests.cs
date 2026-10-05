// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Sample.Pipelines;
using Microsoft.Extensions.DependencyInjection;

namespace DeepSharp.Tests.Samples;

/// <summary>
/// The pipelines sample is run as it stands, through a real container as its program reaches it, so the page a person
/// reads cannot drift from the code: it asks the passenger list what each column holds before anything is declared, has a
/// profile say what should not be there and how each is answered, prepares the passengers and the price series, and saves
/// the passengers' pipeline and runs it again from its file.
/// </summary>
public class PipelineSampleTests
{
    [Fact]
    public void TheSample_AsksTheFileWhatItHolds_SaysWhatShouldNotBeThere_AndPreparesBothShapes()
    {
        using var services = new ServiceCollection().AddDeepSharpPipelines().BuildServiceProvider();
        using var output = new StringWriter();

        PipelineSample.Run(services.GetRequiredService<IPipelineFactory>(), Path.GetDirectoryName(Repository.Data("titanic.csv"))!, output);

        var said = output.ToString();

        Assert.Contains("  alive        boolean, 2 values", said, StringComparison.Ordinal);
        Assert.Contains("  pclass       integer, category offered, 3 values", said, StringComparison.Ordinal);
        Assert.Contains("  deck         category, 7 values, 688 gaps", said, StringComparison.Ordinal);
        Assert.Matches(@"  alive: .* so it hands a model the answer\. Answered by leaving it out\.", said);
        Assert.Matches(@"  class: .* so it says again what pclass says\. Answered by leaving it out\.", said);
        Assert.Matches(@"  fare: 12 of 623 rows hold 0, .* Answered by saying in the schema that 0 stands for a gap\.", said);
        Assert.Matches(@"  age: 133 of 623 rows are gaps\. settle\.gaps .* fill\.missing .* Answered by fill\.missing\.", said);
        Assert.Contains("=== Titanic ===", said, StringComparison.Ordinal);

        // The table's course, started from: every step but the network waiting for what only its person knows, and then said for
        // the passenger list, whose rows it divides as the passengers' own chain does.
        Assert.Contains("=== the table's course, before anything is said ===", said, StringComparison.Ordinal);
        Assert.Contains("  Step 1, 'read.csv': waits for what only you can say: 'path'.", said, StringComparison.Ordinal);
        Assert.Contains("  Step 10, 'normalise': waits for what only you can say: 'column'.", said, StringComparison.Ordinal);
        Assert.DoesNotContain("'evidence.report': waits", said, StringComparison.Ordinal);
        Assert.Contains("=== Titanic, from the table's course ===", said, StringComparison.Ordinal);
        Assert.Equal(2, said.Split("rows      891 (train 623, validation 133, test 135, predict 0)").Length - 1);
        Assert.Contains("=== Apple ===", said, StringComparison.Ordinal);
        // A scaling that names no kind lands the training rows between minus one and one, so age's centre is the middle
        // of its training range rather than its mean: the numbers the wiki's Titanic pipeline has always shown.
        Assert.Contains("normalise[6] centre = 40.21", said, StringComparison.Ordinal);

        // The sample says where its features land and names no kind for them, so every scaling it writes is the one
        // that lands them there: for a column whose training rows start at nought, midrange's centre and spread are
        // the same half of its range, which no other scale gives.
        Assert.Contains("normalise[7] centre = 256.1646", said, StringComparison.Ordinal);
        Assert.Contains("normalise[7] spread = 256.1646", said, StringComparison.Ordinal);
        Assert.Contains("normalise[8] centre = 5", said, StringComparison.Ordinal);
        Assert.Contains("normalise[8] spread = 5", said, StringComparison.Ordinal);
        Assert.Contains("normalise[6] spread = 39.79", said, StringComparison.Ordinal);
        Assert.Contains("identical: True", said, StringComparison.Ordinal);
    }
}

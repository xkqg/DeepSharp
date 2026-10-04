// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A block that names a learner offers the other learners a notebook knows.
/// </summary>
/// <remarks>
/// Comparing a tree with a network is the one gesture this library added a second learner for, and in a notebook that
/// gesture is swapping the verb of one block. A learner is a stage like any other, so the verbs of that stage are what
/// the block's own list offers.
/// </remarks>
public class LearnerBlockTests
{
    [Fact]
    public void ABlockNamingALearner_OffersTheOtherLearnersAsItsStage()
    {
        var network = (IPipelineStep)NotebookVerbs.Catalog().ReadStep(NotebookVerbs.Catalog().Describe("learn.network").Template);
        var tree = (IPipelineStep)NotebookVerbs.Catalog().ReadStep(NotebookVerbs.Catalog().Describe("learn.ml").Template);

        Assert.Equal(network.ActingCapability(), tree.ActingCapability());
        Assert.Equal(network.Stage(), tree.Stage());
    }
}

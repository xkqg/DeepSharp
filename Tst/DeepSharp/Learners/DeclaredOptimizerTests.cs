// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Text;
using System.Text.Json;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The optimizers a declared network may be moved by beyond Sgd and Adam — AdamW, RmsProp and Nadam — and the one rule that
/// keeps the two tables of a network's words together: every optimizer and loss the catalog ships has a declared word, and
/// every declared word is one the catalog ships.
/// </summary>
public class DeclaredOptimizerTests
{
    private static FittingBuilder Passengers() => Pdd.Create()
        .ReadCsv(Repository.Data("titanic.csv"))
        .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
        .SplitStratified("survived", 0.70, 0.15)
        .FillMissing("age", With.Median)
        .EncodeCategories()
        .Normalise("age", Scale.MidRange)
        .Normalise("fare", Scale.MidRange)
        .Normalise("sibsp", Scale.MidRange)
        .Normalise("parch", Scale.MidRange)
        .Target("survived");

    [Fact]
    public void EveryOptimizerAndLossTheCatalogShips_HasADeclaredWord_AndEveryDeclaredWordIsOneItShips()
    {
        // The two tables once drifted: a kind the network's file knew that no declaration could name. Held here, a kind added
        // to one and not the other fails at once, whichever it was added to.
        var catalog = NetworkCatalog.BuiltIn();

        Assert.Equal(Shipped<Optimizer>(catalog), NetworkWords.Optimizers.Select(kind => kind.Name).Order(StringComparer.Ordinal));
        Assert.Equal(Shipped<Loss>(catalog), NetworkWords.Losses.Select(kind => kind.Name).Order(StringComparer.Ordinal));
        Assert.Equal(Shipped<LearningRateSchedule>(catalog), NetworkWords.Schedules.Select(kind => kind.Name).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void TheNewOptimizers_AreDeclaredWithPyTorchsDefaults_AndBecomeTheOnesTheCoreHas()
    {
        var adamW = Assert.IsType<AdamW>(Word("adamw").Declared().Moved());
        var rmsProp = Assert.IsType<RmsProp>(Word("rmsprop").Declared().Moved());
        var nadam = Assert.IsType<Nadam>(Word("nadam").Declared().Moved());

        Assert.Equal([0.001, 0.9, 0.999, 1e-8, 0.01], new[] { adamW.Rate, adamW.Betas.First, adamW.Betas.Second, adamW.Epsilon, adamW.WeightDecay });
        Assert.Equal([0.01, 0.99, 1e-8, 0], new[] { rmsProp.Rate, rmsProp.Alpha, rmsProp.Epsilon, rmsProp.Momentum });
        Assert.Equal([0.002, 0.9, 0.999, 1e-8, 0.004], new[] { nadam.Rate, nadam.Betas.First, nadam.Betas.Second, nadam.Epsilon, nadam.MomentumDecay });
    }

    [Theory]
    [InlineData("adamw")]
    [InlineData("rmsprop")]
    [InlineData("nadam")]
    public void ANewOptimizer_IsWrittenUnderTheKeysItsFileWrites_ButForTheBetas_WhichFollowAdamsDeclaredWords(string kind)
    {
        // One key per setting: every number the network's file writes the optimizer with is declared under the same key,
        // except the two betas, which a declaration writes as Adam's are declared, each under a key of its own.
        var declared = Word(kind).Settings.Select(setting => setting.Key).ToHashSet(StringComparer.Ordinal);
        var written = WrittenKeys((ISaved)Word(kind).Declared().Moved());

        Assert.Equal(
            written.Where(key => key != "betas").Order(StringComparer.Ordinal),
            declared.Where(key => key is not ("firstMoment" or "secondMoment")).Order(StringComparer.Ordinal));
        Assert.Equal(written.Contains("betas"), declared.Contains("firstMoment") && declared.Contains("secondMoment"));
    }

    [Fact]
    public void TheDoors_WritePyTorchsDefaults_AndTheNumbersACallerNames()
    {
        var adamW = Learner(network => network.AdamW()).Optimizer;
        var rmsProp = Learner(network => network.RmsProp()).Optimizer;
        var nadam = Learner(network => network.Nadam()).Optimizer;
        var named = Learner(network => network.AdamW(rate: 0.5, weightDecay: 0.25, betas: new Betas(0.8, 0.9), epsilon: 1e-6)).Optimizer;
        var carried = Learner(network => network.RmsProp(rate: 0.5, alpha: 0.9, momentum: 0.25, epsilon: 1e-6)).Optimizer;
        var decayed = Learner(network => network.Nadam(rate: 0.5, betas: new Betas(0.8, 0.9), momentumDecay: 0.002, epsilon: 1e-6)).Optimizer;

        Assert.Equal("adamw", adamW.Kind);
        Assert.Equal(Word("adamw").Declared(), adamW);
        Assert.Equal(Word("rmsprop").Declared(), rmsProp);
        Assert.Equal(Word("nadam").Declared(), nadam);
        Assert.Equal([0.5, 0.8, 0.9, 1e-6, 0.25], Numbers(named, "rate", "firstMoment", "secondMoment", "epsilon", "weightDecay"));
        Assert.Equal([0.5, 0.9, 1e-6, 0.25], Numbers(carried, "rate", "alpha", "epsilon", "momentum"));
        Assert.Equal([0.5, 0.8, 0.9, 1e-6, 0.002], Numbers(decayed, "rate", "firstMoment", "secondMoment", "epsilon", "momentumDecay"));
    }

    [Fact]
    public void InKerassWords_AdamWIsDeclaredWithPyTorchsNumbers_AndRmsPropAndNadamAreRefused_NamingThePyTorchDoor()
    {
        // Keras's AdamW decays as PyTorch's does and only leaves other numbers unsaid, which a caller can name; Keras's
        // RMSprop adds its epsilon under the root and its Nadam holds its momentum decay fixed, so the Keras words cannot say
        // the optimizers this library runs, and a declaration in them is refused rather than written as something else.
        var keras = Assert.IsType<LearnNetworkStep>(Passengers().WithTensorflow(network => network.Dense(1).AdamW().MeanSquaredError().Run(1)).Build().Declaration.Learner);

        Assert.Equal(Word("adamw").Declared(), keras.Optimizer);
        Assert.Contains("WithTorch", Assert.Throws<NotSupportedException>(() => Passengers().WithTensorflow(network => network.Dense(1).RmsProp())).Message, StringComparison.Ordinal);
        Assert.Contains("WithTorch", Assert.Throws<NotSupportedException>(() => Passengers().WithTensorflow(network => network.Dense(1).Nadam())).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("adamw")]
    [InlineData("rmsprop")]
    [InlineData("nadam")]
    public void ANetworkDeclaredWithANewOptimizer_IsTheNetworkTheSameWordsWrittenAsCodeGive_NumberForNumber(string kind)
    {
        var declared = Passengers()
            .WithTorch(network => (kind switch
            {
                "adamw" => network.Dense(4).Relu().Dense(1).AdamW(0.01, weightDecay: 0.1),
                "rmsprop" => network.Dense(4).Relu().Dense(1).RmsProp(0.001, momentum: 0.9),
                _ => network.Dense(4).Relu().Dense(1).Nadam(0.01),
            })
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 3))
            .Build()
            .Train();
        Optimizer optimizer = kind switch
        {
            "adamw" => new AdamW(0.01) { WeightDecay = 0.1 },
            "rmsprop" => new RmsProp(0.001) { Momentum = 0.9 },
            _ => new Nadam(0.01),
        };
        var written = new Sequential().Dense(4).Relu().Dense(1)
            .Compile(optimizer, new BinaryCrossEntropy())
            .Fit(Passengers().Build().RunFor(Needs.OneScale), new FitOptions(20260929) { Epochs = 3 });

        Assert.Equal(written.History!.Epochs.Select(epoch => epoch.Loss), declared.History!.Epochs.Select(epoch => epoch.Loss));
        Assert.Equal(
            written.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()),
            declared.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()));
    }

    // Every kind of a role the core assembly holds, by the name it is registered under: each of them is in the catalog the
    // package ships, so a kind written and never registered fails here too.
    private static IEnumerable<string> Shipped<TRole>(NetworkCatalog catalog)
    {
        var names = typeof(TRole).Assembly.GetExportedTypes()
            .Where(type => !type.IsAbstract && typeof(TRole).IsAssignableFrom(type))
            .Select(type => (string)type.GetProperty("Name", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!)
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.All(names, name => Assert.Contains(name, catalog.Names));

        return names;
    }

    private static PartKind Word(string name) => NetworkWords.Optimizers.Single(kind => kind.Name == name);

    private static LearnNetworkStep Learner(Func<NetworkDeclaration, NetworkDeclaration> optimizer) =>
        Assert.IsType<LearnNetworkStep>(Passengers()
            .WithTorch(network => optimizer(network.Dense(1)).MeanSquaredError().Run(1))
            .Build().Declaration.Learner);

    private static double[] Numbers(PartDeclaration part, params string[] keys) => [.. keys.Select(part.Number)];

    // The keys a kind writes its settings under in a network's file.
    private static string[] WrittenKeys(ISaved saved)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            saved.WriteSettings(writer);
            writer.WriteEndObject();
        }

        return [.. JsonDocument.Parse(Encoding.UTF8.GetString(stream.ToArray())).RootElement.EnumerateObject().Select(property => property.Name)];
    }
}

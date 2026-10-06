// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Learners.ML;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What the Binance package brings into an application, and what it does not bring into any other: it is the one package
/// that carries HTTP and retries, and a project that only serves a trained model never does, since the pipeline it saves names
/// a landed file and nothing of Binance. Each is held by a test rather than remembered — and what the package publishes is
/// pinned to the member, because the first public API on nuget.org is the one every later release is held to.
/// </summary>
public class BinancePackageTests
{
    private static readonly Assembly Package = typeof(BinanceCandles).Assembly;

    private static XDocument Project() =>
        XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Pipelines.Binance", "DeepSharp.Pipelines.Binance.csproj"));

    private static string[] ReferencedOutsideTheRuntime(Assembly assembly)
    {
        // The runtime's own assemblies sit in one folder, so "part of .NET" is a fact looked up rather than a prefix.
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        return [.. assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Order(StringComparer.Ordinal)];
    }

    [Fact]
    public void ThePackageReferencesThePipelineAndPollyCoreAlone_AndPollyIsTheVersionItsTestsRunOn()
    {
        var packages = Project().Descendants("PackageReference").Select(reference => $"{reference.Attribute("Include")!.Value} {reference.Attribute("Version")!.Value}");
        var projects = Project().Descendants("ProjectReference").Select(reference => reference.Attribute("Include")!.Value);

        Assert.Equal(["Polly.Core 8.8.0"], packages);
        Assert.Equal([@"..\DeepSharp.Pipelines\DeepSharp.Pipelines.csproj"], projects);
        Assert.Equal(["DeepSharp.Pipelines", "Polly.Core"], ReferencedOutsideTheRuntime(Package));
    }

    [Fact]
    public void TheRuntimesHttpStackIsInTheRuntimeFolder_SoTheClosedListAboveCannotSeeIt_AndAPinOfItsOwnDoes()
    {
        // The reason this pin has its own test: System.Net.Http ships with the runtime, so the filter every closed-list test
        // uses to leave the runtime out leaves HTTP out too, and a package could take it up without any of them failing.
        Assert.Contains("System.Net.Http", Package.GetReferencedAssemblies().Select(reference => reference.Name));
        Assert.DoesNotContain("System.Net.Http", ReferencedOutsideTheRuntime(Package));
    }

    [Theory]
    [InlineData(typeof(Pdd))]
    [InlineData(typeof(DeepSharp.Tensors.Tensor))]
    [InlineData(typeof(TrainedNetwork))]
    [InlineData(typeof(MLModelFile))]
    public void NothingAProjectThatOnlyServesAModelBrings_ReferencesHttpOrPolly(Type brought)
    {
        // The pipeline library, the tensors, the network's learner and the declaration of the ML.NET one are what a model
        // server references. None of them reaches for HTTP or for a retry, so a server that loads a pipeline whose source was
        // landed from Binance carries nothing of how it was landed.
        var references = brought.Assembly.GetReferencedAssemblies().Select(reference => reference.Name!).ToArray();

        Assert.DoesNotContain("System.Net.Http", references);
        Assert.DoesNotContain(references, name => name.StartsWith("Polly", StringComparison.Ordinal));
        Assert.DoesNotContain(references, name => name.Contains("Binance", StringComparison.Ordinal));
    }

    [Fact]
    public void ThePackageIsTheOneThatCarriesHttpAndTheRetries_WhichIsWhatTheOtherPinsAreHeldAgainst()
    {
        // The control: the same walk over the package that does carry them finds them, so the pins above can fail.
        var references = Package.GetReferencedAssemblies().Select(reference => reference.Name!).ToArray();

        Assert.Contains("System.Net.Http", references);
        Assert.Contains("Polly.Core", references);
    }

    [Fact]
    public void NoTypeOfPollys_StandsInAnythingThePackagePublishes()
    {
        var polly = typeof(Polly.ResiliencePipeline).Assembly;
        var named = Package.GetExportedTypes().SelectMany(Named).Where(type => type.Assembly == polly).Select(type => type.FullName).Distinct();

        Assert.Empty(named);
    }

    [Fact]
    public void ThePollyWalkSeesWhatItIsMeantTo_ATypeThatDoesPublishOneIsFound()
    {
        // The control for the walk: a type that names one of Polly's in a signature is found by it.
        Assert.Contains(typeof(Polly.ResiliencePipeline), Named(typeof(PublishesAPipeline)));
        Assert.Contains(typeof(Polly.ResiliencePipeline), Named(typeof(PublishesAField)));
    }

    [Fact]
    public void ThePackagePublishesThreeTypes_AndNothingElse()
    {
        Assert.Equal(
            ["DeepSharp.Pipelines.BinanceCandles", "DeepSharp.Pipelines.BinanceException", "DeepSharp.Pipelines.BinanceSourceExtensions"],
            Package.GetExportedTypes().Where(type => !type.IsNested).Select(type => type.FullName!).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void WhatThePackagePublishes_IsPinnedToTheMember_BecauseTheFirstReleaseIsTheOneEveryLaterOneIsHeldTo()
    {
        Assert.Equal(
            [
                "BinanceCandles..ctor(String, String, DateTime, DateTime)",
                "BinanceCandles.At(Uri)",
                "BinanceCandles.From",
                "BinanceCandles.Interval",
                "BinanceCandles.LandAsync(SourceFolder, CancellationToken)",
                "BinanceCandles.MostPages",
                "BinanceCandles.On(TimeProvider)",
                "BinanceCandles.Symbol",
                "BinanceCandles.Through(HttpMessageHandler)",
                "BinanceCandles.To",
            ],
            Surface(typeof(BinanceCandles)));

        Assert.Equal(
            ["BinanceException..ctor(String, Nullable<HttpStatusCode>, Nullable<Int32>, Exception)", "BinanceException.Code", "BinanceException.Status"],
            Surface(typeof(BinanceException)));

        Assert.Equal(
            ["BinanceSourceExtensions.ReadBinanceAsync(PipelineBuilder, BinanceCandles, CancellationToken)"],
            Surface(typeof(BinanceSourceExtensions)));
    }

    [Fact]
    public void TheWindowIsANamedWindow_ItsFluentChoicesAreAboutHowItIsSent_NeverWhatItNames()
    {
        // The identity constructor takes the four things that name the window and nothing about how it is fetched.
        var constructor = Assert.Single(typeof(BinanceCandles).GetConstructors());

        Assert.Equal(["symbol", "interval", "from", "to"], constructor.GetParameters().Select(parameter => parameter.Name));
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags_AndItsDescriptionSaysWhoseDataItLands()
    {
        var project = Project();
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');
        var description = project.Descendants("Description").Single().Value;

        Assert.Equal("DeepSharp.Pipelines.Binance", project.Descendants("PackageId").Single().Value);
        Assert.Contains("binance", tags);
        Assert.Contains("pipeline", tags);
        Assert.Contains("Not affiliated with Binance", description, StringComparison.Ordinal);
        Assert.Contains("Binance's terms", description, StringComparison.Ordinal);
    }

    // The public and protected members of a type as a caller sees them, one line each: its constants, constructors,
    // properties and methods, with the types of what they take — a name and a signature, no more.
    private static string[] Surface(Type type)
    {
        const BindingFlags Seen = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        static string Spelled(Type each) => each.IsGenericType
            ? $"{each.Name[..each.Name.IndexOf('`', StringComparison.Ordinal)]}<{string.Join(", ", each.GetGenericArguments().Select(Spelled))}>"
            : each.Name;

        static string Signature(MethodBase method) => $"({string.Join(", ", method.GetParameters().Select(parameter => Spelled(parameter.ParameterType)))})";

        string[] lines =
        [
            .. type.GetConstructors(Seen).Select(constructor => $"{type.Name}..ctor{Signature(constructor)}"),
            .. type.GetFields(Seen).Select(field => $"{type.Name}.{field.Name}"),
            .. type.GetProperties(Seen).Select(property => $"{type.Name}.{property.Name}"),
            .. type.GetMethods(Seen).Where(method => !method.IsSpecialName).Select(method => $"{type.Name}.{method.Name}{Signature(method)}"),
        ];

        return [.. lines.Order(StringComparer.Ordinal)];
    }

    // Every type a published type names where a caller sees it: what it derives from and implements, and every public or
    // protected member's type, parameters and return, generic arguments included.
    private static IEnumerable<Type> Named(Type type)
    {
        const BindingFlags Seen = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        var members = type.GetMembers(Seen).Where(member => member switch
        {
            MethodBase method => method.IsPublic || method.IsFamily || method.IsFamilyOrAssembly,
            FieldInfo field => field.IsPublic || field.IsFamily || field.IsFamilyOrAssembly,
            PropertyInfo property => property.GetAccessors(nonPublic: true).Any(accessor => accessor.IsPublic || accessor.IsFamily || accessor.IsFamilyOrAssembly),
            EventInfo happening => happening.AddMethod is { } add && (add.IsPublic || add.IsFamily),
            Type nested => nested.IsNestedPublic || nested.IsNestedFamily,
            _ => false,
        });

        var named = new List<Type> { type };
        named.AddRange(type.GetInterfaces());

        if (type.BaseType is { } baseType)
        {
            named.Add(baseType);
        }

        foreach (var member in members)
        {
            named.AddRange(member switch
            {
                MethodInfo method => [method.ReturnType, .. method.GetParameters().Select(parameter => parameter.ParameterType)],
                ConstructorInfo constructor => constructor.GetParameters().Select(parameter => parameter.ParameterType),
                FieldInfo field => [field.FieldType],
                PropertyInfo property => [property.PropertyType, .. property.GetIndexParameters().Select(parameter => parameter.ParameterType)],
                EventInfo happening => [happening.EventHandlerType!],
                _ => [],
            });
        }

        return named.SelectMany(Unwrapped);
    }

    // A type and every type it is made of: an array's element, a reference's target, a generic's arguments.
    private static IEnumerable<Type> Unwrapped(Type type) =>
        type.HasElementType
            ? Unwrapped(type.GetElementType()!)
            : type.IsGenericType ? [type, .. type.GetGenericArguments().SelectMany(Unwrapped)] : [type];

    // Types that publish one of Polly's, only to see that the walk finds them.
    public sealed class PublishesAPipeline
    {
        public Polly.ResiliencePipeline Pipeline() => Polly.ResiliencePipeline.Empty;
    }

    public sealed class PublishesAField
    {
        public Polly.ResiliencePipeline? Held;
    }
}

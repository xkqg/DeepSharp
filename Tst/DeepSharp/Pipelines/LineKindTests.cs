// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Every kind a line offers is written, and writes the kind it is named after.
/// </summary>
/// <remarks>
/// A line is a door to a verb, and the kinds are its methods — so a method that writes the wrong kind, or one its verb
/// would throw on, is a line that reads right and does something else. Each is invoked here rather than described: the
/// kinds are found by reflection, so one added to any line is exercised the day it is written, and so is the guard
/// every line keeps against a column list that is not there.
/// </remarks>
public class LineKindTests
{
    public static TheoryData<string, string> EveryKind()
    {
        var kinds = new TheoryData<string, string>();

        foreach (var (line, method) in Kinds())
        {
            kinds.Add(line.Name, method.Name + "/" + method.GetParameters().Length);
        }

        return kinds;
    }

    [Theory]
    [MemberData(nameof(EveryKind))]
    public void EveryKindALineOffers_WritesOneStepAColumnCarryingTheKindItNames(string line, string method)
    {
        var (type, kind) = Find(line, method);
        var written = Steps(Written(type, kind, ["a", "b"]));

        Assert.Equal(2, written.Count);
        Assert.All(written, step => Assert.Equal(written[0].Verb, step.Verb));

        // The kind is named by the method, so the step it writes carries a value of that name: `MidRange` writes
        // `Scale.MidRange`, `Log` writes `Maths.Log`, `Mean` writes the strategy spelled `mean`. The one method that
        // names no kind says so in its name, and writes the one the pipeline says its features land in.
        var named = kind.Name == nameof(ScaleBuilder.Columns) ? NormaliseStep.DefaultScale.ToString() : kind.Name;

        Assert.Contains(
            Settings(written[0]),
            value => string.Equals(value, named, StringComparison.OrdinalIgnoreCase));
    }

    [Theory]
    [MemberData(nameof(EveryKind))]
    public void EveryKindALineOffers_RefusesAColumnListThatIsNotThere(string line, string method)
    {
        var (type, kind) = Find(line, method);

        Assert.Throws<ArgumentNullException>(() => Written(type, kind, null));
    }

    [Fact]
    public void EveryLineThereIs_IsExercised()
    {
        // The theories are only worth anything if they find every line, so the set is pinned: a line added or taken
        // away says so here first.
        Assert.Equal(
            ["BoundsLine", "CategoryPartLine", "GapLine", "MathsLine", "NotANumberLine", "NumberPartLine", "PeriodLine", "ScaleBuilder", "SettleLine"],
            Lines().Select(line => line.Name).Order(StringComparer.Ordinal));
    }

    // Every kind of every line: a method taking the columns last, which is how a kind is written.
    private static IEnumerable<(Type Line, MethodInfo Method)> Kinds() =>
        Lines().SelectMany(line => line
            .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Concat(line.BaseType is { IsAbstract: true } shared
                ? shared.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                : [])
            .Where(method => method.GetParameters() is [.., { ParameterType.IsArray: true } last]
                && last.ParameterType.GetElementType() == typeof(string))
            .Select(method => (line, method)));

    private static IEnumerable<Type> Lines() =>
        typeof(Pdd).Assembly.GetExportedTypes()
            .Where(type => type is { IsClass: true, IsAbstract: false }
                && type.GetInterfaces().Any(face => face.Name == "IDeclaresSteps"));

    private static (Type Line, MethodInfo Kind) Find(string line, string method)
    {
        var (name, count) = (method.Split('/')[0], int.Parse(method.Split('/')[1], null));

        return Kinds().Single(each =>
            each.Line.Name == line && each.Method.Name == name && each.Method.GetParameters().Length == count);
    }

    // The line written with one kind over these columns, and nothing else.
    private static object Written(Type line, MethodInfo kind, string[]? columns)
    {
        var written = Activator.CreateInstance(line)!;
        var given = kind.GetParameters();
        object?[] arguments = [.. given.Take(given.Length - 1).Select(Sensible), columns];

        try
        {
            kind.Invoke(written, arguments);
        }
        catch (TargetInvocationException thrown)
        {
            throw thrown.InnerException!;
        }

        return written;
    }

    // What a setting before the columns is given: the first word of its own kind, or a share a bound can take.
    private static object Sensible(ParameterInfo parameter) =>
        parameter.ParameterType == typeof(double)
            ? 0.25
            : parameter.ParameterType.IsEnum
                ? Enum.GetValues(parameter.ParameterType).GetValue(parameter.ParameterType == typeof(OutOfRange) ? 1 : 0)!
                : Enum.GetValues(parameter.ParameterType.GetElementType()!).GetValue(0)!;

    private static IReadOnlyList<IPipelineStep> Steps(object line) =>
        (IReadOnlyList<IPipelineStep>)line.GetType()
            .GetInterfaceMap(line.GetType().GetInterfaces().Single(face => face.Name == "IDeclaresSteps"))
            .TargetMethods.Single(method => method.Name.EndsWith("get_Steps", StringComparison.Ordinal))
            .Invoke(line, null)!;

    // What a step says about itself, as words: every value it writes under one of its own keys.
    private static IEnumerable<string> Settings(IPipelineStep step) =>
        step.GetType()
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Select(property => property.GetValue(step))
            .Where(value => value is not null)
            .Select(value => value is FillStrategy strategy ? strategy.Name : value!.ToString()!);
}

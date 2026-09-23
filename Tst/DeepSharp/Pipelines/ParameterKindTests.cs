// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The kinds of value a parameter can hold, each on its own: what it reads, what it writes, and what it
/// refuses. A package that brings a verb builds its parameters from these, so they are held to their word
/// here rather than only through the steps that happen to use them today.
/// </summary>
public class ParameterKindTests
{
    private static JsonElement Step(string json)
    {
        using var document = JsonDocument.Parse(json);

        return document.RootElement.Clone();
    }

    private static string Written<T>(StepParameter<T> parameter, T value)
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            parameter.Write(writer, value);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    [Fact]
    public void EveryKeyAFileHoldsBeyondAParameters_IsNamedOnce_AndWrittenUnderThatName()
    {
        // The keys inside a way of filling, the share worked out from the others, and the key a step's verb is
        // written under: each is said once, publicly, so a form or an editor never types it again.
        var with = new FillStrategyParameter("with", "How a gap is filled.", With.Constant(1), ["constant", "mean"], "filling a gap");
        var strategy = Step(Written(with, With.Constant(2.5))).GetProperty("with");

        Assert.Equal("constant", strategy.GetProperty(FillStrategyParameter.KindKey).GetString());
        Assert.Equal(2.5, strategy.GetProperty(FillStrategyParameter.ValueKey).GetDouble());
        Assert.Equal([FillStrategyParameter.KindKey, FillStrategyParameter.ValueKey], with.StrategyKeys);

        var shares = new SplitSharesParameter();

        Assert.Contains(SplitSharesParameter.TestKey, shares.Keys);
        Assert.Equal(0.15, Step(Written(shares, new SplitShares(0.70, 0.15, 0.15))).GetProperty(SplitSharesParameter.TestKey).GetDouble());
        Assert.Equal(StepCatalog.StepKey, Step(StepCatalog.BuiltIn().Describe("read.csv").Template).EnumerateObject().First().Name);
    }

    [Fact]
    public void ANumberWithNoBound_TakesAnyFiniteNumberAndNothingElse()
    {
        var number = new NumberParameter("weight", "How much it weighs.", 1.5);

        Assert.Equal(-3.25, number.Require(-3.25));
        Assert.Equal(-3.25, number.Read(Step("""{"weight":-3.25}""")));
        Assert.Equal("""{"weight":-3.25}""", Written(number, -3.25));

        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => number.Require(double.PositiveInfinity));

        Assert.Contains("a finite number", refused.Message, StringComparison.Ordinal);
        Assert.Null(number.Above);
    }

    [Fact]
    public void ANumberWithABound_RefusesTheBoundItself()
    {
        var number = new NumberParameter("at", "How far out.", 1.5, above: 0);

        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => number.Require(0));

        Assert.Contains("a number above 0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKindWithNoRuleOfItsOwn_TakesEveryValueItCanHold()
    {
        var choice = new OneOfParameter<Scale>("scale", "Which scale.", Scale.Standard);
        var flag = new TrueOrFalseParameter("asCategories", "Whether they are groups.", true);

        Assert.Equal(Scale.Robust, choice.Require(Scale.Robust));
        Assert.False(flag.Require(false));
        Assert.Equal(["standard", "minmax", "maxabs", "robust", "quantile", "power"], choice.Choices);
        Assert.Equal("""{"asCategories":false}""", Written(flag, false));
    }

    [Fact]
    public void AnOptionalNewColumnLeftOut_IsLeftToTheStep()
    {
        // The result of a reshaping goes into the column it came from unless a file says otherwise, and a
        // file that says nothing means exactly that.
        var reshaped = MathsStep.ReadFrom(Step("""{"step":"maths","column":"fare","maths":"log1p"}"""));

        Assert.Equal("fare", reshaped.Into);
        Assert.True(new NewColumnParameter("into", "Where it goes.", "column", optional: true).Optional);
        Assert.Null(new NewColumnParameter("into", "Where it goes.", "column", optional: true).Require(" "));
        Assert.Throws<ArgumentException>(() => new NewColumnParameter("name", "What it is called.", "feature").Require(" "));
    }

    [Fact]
    public void SeveralOfAKind_ReadsEveryOneAndRefusesAWordNobodyDefined()
    {
        var parts = new SeveralOfParameter<TimePart>("parts", "Which pieces.", [TimePart.Month]);

        Assert.Equal([TimePart.Hour, TimePart.Season], parts.Read(Step("""{"parts":["hour","Season"]}""")));
        Assert.Throws<FormatException>(() => parts.Read(Step("""{"parts":["fortnight"]}""")));
        Assert.Throws<FormatException>(() => parts.Read(Step("""{"parts":[7]}""")));
        Assert.Throws<FormatException>(() => parts.Read(Step("""{"parts":"hour"}""")));
        Assert.Throws<ArgumentException>(() => parts.Require([]));
        Assert.Equal(8, parts.Choices.Count);
    }

    [Fact]
    public void ColumnsAreNames_AndOnlyNames()
    {
        var columns = new ColumnsParameter("columns", "Which columns.", ["a"], ColumnKinds.Numbers);

        Assert.Equal(["a", "b"], columns.Read(Step("""{"columns":["a","b"]}""")));
        Assert.Throws<FormatException>(() => columns.Read(Step("""{"columns":["a",3]}""")));
        Assert.Throws<ArgumentException>(() => columns.Require(["a", " "]));
        Assert.Equal(ColumnKinds.Numbers, columns.Accepts);
    }

    [Fact]
    public void AColumnNamedTwiceInOneList_IsRefused_ByName()
    {
        // Leaving a column out twice would pass every rule and fail when the rows run, on a column already gone.
        var twice = Assert.Throws<ArgumentException>(() => new DropColumnsStep(["age", "fare", "age"]));

        Assert.Contains("'age'", twice.Message, StringComparison.Ordinal);
        Assert.Contains("twice", twice.Message, StringComparison.Ordinal);
        Assert.Equal("columns", twice.ParamName);
    }

    [Fact]
    public void AListWhosePlacesAreRoles_MayNameOneColumnInMoreThanOne()
    {
        // An indicator's columns are its high, its low and its close: rows with one price have it as all three.
        var roles = new ColumnsParameter("columns", "Roles.", ["high", "low", "close"], ColumnKinds.Numbers, repeatable: true);

        Assert.True(roles.Repeatable);
        Assert.Equal(["close", "close", "close"], roles.Require(["close", "close", "close"]));
        Assert.False(new ColumnsParameter("columns", "Which.", ["a"], ColumnKinds.Any).Repeatable);
    }

    [Fact]
    public void AColumnSaysWhichKindsOfColumnItCanWorkOn()
    {
        var column = new ColumnParameter("column", "The column.", "column", ColumnKinds.Moments);

        Assert.Equal([ColumnKind.Timestamp], column.Accepts);
        Assert.Contains(ColumnKind.Category, ColumnKinds.Any);
        Assert.Equal([ColumnKind.Number, ColumnKind.Integer], ColumnKinds.Fillable);
        Assert.Equal([ColumnKind.Number], ColumnKinds.Fractions);
        Assert.Equal([ColumnKind.Timestamp, ColumnKind.Integer, ColumnKind.Number], ColumnKinds.Ordered);
    }

    [Fact]
    public void ADeclaredColumnWithAKeyNobodyDefined_IsRefused()
    {
        var declared = new ColumnDeclarationsParameter("columns", "The columns.", [new ColumnDeclaration("a", ColumnKind.Number, false)]);

        var refused = Assert.Throws<FormatException>(
            () => declared.Read(Step("""{"columns":[{"name":"a","kind":"number","optional":false,"colour":"red"}]}""")));

        Assert.Contains("'colour'", refused.Message, StringComparison.Ordinal);
        Assert.Equal(["name", "kind", "optional"], declared.ColumnKeys);
    }

    [Fact]
    public void AFillStrategySaysWhichNamesItAllows()
    {
        var with = new FillStrategyParameter("with", "What goes in.", With.Mean, ["mean", "zero"], "filling a gap");

        Assert.Equal(["mean", "zero"], with.Allowed);
        Assert.Throws<ArgumentException>(() => with.Require(With.Median));
        Assert.Equal("""{"with":{"kind":"constant","value":-1}}""", Written(with, With.Constant(-1)));
    }

    [Fact]
    public void AParameterNeedsAKeyAndSaysWhatItMeans()
    {
        Assert.Throws<ArgumentException>(() => new TextParameter(" ", "Something.", "x"));
        Assert.Throws<ArgumentException>(() => new TextParameter("key", " ", "x"));
        Assert.Equal(["key"], new TextParameter("key", "Something.", "x").Keys);
        Assert.Equal(["train", "validation", "test", "predict"], new SplitSharesParameter().Keys);
    }

    [Theory]
    // A word is one of the words, in whatever case a person typed it — and nothing else. The reader used to
    // take whatever the runtime's enum parser took: a number standing for a place in the list, a list of
    // words joined by a comma that added up to a different one, a word with spaces round it.
    [InlineData("MinMax", true)]
    [InlineData("minmax", true)]
    [InlineData("1", false)]
    [InlineData("standard,minmax", false)]
    [InlineData(" minmax", false)]
    [InlineData("min max", false)]
    public void AWordIsOneOfItsChoices_InAnyCase_AndNothingElse(string written, bool read)
    {
        var scale = new OneOfParameter<Scale>("scale", "Which scale.", Scale.Standard);
        var parts = new SeveralOfParameter<Scale>("scales", "Which scales.", [Scale.Standard]);

        if (read)
        {
            Assert.Equal(Scale.MinMax, scale.Read(Step($$"""{"scale":"{{written}}"}""")));
            Assert.Equal([Scale.MinMax], parts.Read(Step($$"""{"scales":["{{written}}"]}""")));

            return;
        }

        var refused = Assert.Throws<FormatException>(() => scale.Read(Step($$"""{"scale":"{{written}}"}""")));

        Assert.Contains("standard, minmax, maxabs", refused.Message, StringComparison.Ordinal);
        Assert.Throws<FormatException>(() => parts.Read(Step($$"""{"scales":["{{written}}"]}""")));
    }

    [Fact]
    public void AValueThatIsNoneOfTheWords_IsRefusedAtTheCSharpDoorToo()
    {
        // A number cast to the set compiles, and was carried until the file was written — where it came out
        // as a number no reader takes. The C# door and the file door hold a word to one rule.
        Assert.Throws<ArgumentOutOfRangeException>(() => new NormaliseStep("a", (Scale)99));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimePartsStep("t", [TimePart.Hour, (TimePart)42]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeclareStep([new ColumnDeclaration("a", (ColumnKind)7, false)]));
    }

    [Theory]
    // A whole number written with a fraction of nought is that whole number: nothing is lost, and it is what
    // any JSON Schema calls an integer. A fraction of anything else is refused, and so is a size nobody can hold.
    [InlineData("3", 3)]
    [InlineData("3.0", 3)]
    [InlineData("1e2", 100)]
    [InlineData("-2147483648", int.MinValue)]
    public void AWholeNumber_IsReadWhenItIsOne(string written, int read)
    {
        var seed = new WholeNumberParameter("seed", "The seed.", 1);

        Assert.Equal(read, seed.Read(Step($$"""{"seed":{{written}}}""")));
    }

    [Theory]
    [InlineData("3.5")]
    [InlineData("2147483648")]
    [InlineData("1e10")]
    [InlineData("\"3\"")]
    public void AWholeNumberThatIsNotOne_IsRefusedNamingItsKey(string written)
    {
        var seed = new WholeNumberParameter("seed", "The seed.", 1);

        var refused = Assert.Throws<FormatException>(() => seed.Read(Step($$"""{"seed":{{written}}}""")));

        Assert.Contains("'seed'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANumberTooLargeToHold_IsRefusedNamingItsKey()
    {
        // 1e400 is a number to JSON and an infinity to a double; the reader used to let the runtime refuse it
        // in words that named neither the step nor the key.
        var at = new NumberParameter("at", "How far out.", 1.5);

        var refused = Assert.Throws<FormatException>(() => at.Read(Step("""{"at":1e400}""")));

        Assert.Contains("'at'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryKind_IsVisitedAsItself()
    {
        // The kinds are a closed set and the things built from them — a form, a schema, a reference page —
        // are open, so each of those visits the kinds rather than asking every value what type it is.
        StepParameter[] kinds =
        [
            new TextParameter("text", "Words.", "x"),
            new FilePathParameter("path", "A path.", "x.csv"),
            new ColumnParameter("column", "A column.", "x", ColumnKinds.Any),
            new NewColumnParameter("made", "A new column.", "x"),
            new ColumnsParameter("columns", "Columns.", ["x"], ColumnKinds.Any),
            new NumberParameter("number", "A number.", 1),
            new WholeNumberParameter("whole", "A whole number.", 1),
            new TrueOrFalseParameter("flag", "A flag.", true),
            new ShareParameter("share", "A share."),
            new OneOfParameter<Scale>("scale", "A scale.", Scale.Standard),
            new SeveralOfParameter<TimePart>("parts", "Parts.", [TimePart.Month]),
            new FillStrategyParameter("with", "A strategy.", With.Mean, ["mean"], "filling a gap"),
            new SplitSharesParameter(),
            new ColumnDeclarationsParameter("columns", "Declared columns.", [new ColumnDeclaration("x", ColumnKind.Text, false)]),
        ];

        Assert.Equal(
            ["Text", "FilePath", "Column", "NewColumn", "Columns", "Number", "WholeNumber", "TrueOrFalse", "Share", "OneOf",
             "SeveralOf", "FillStrategy", "SplitShares", "ColumnDeclarations"],
            kinds.Select(kind => kind.Accept(new KindName())));
    }

    /// <summary>Says which kind it was handed.</summary>
    private sealed class KindName : IStepParameterVisitor<string>
    {
        public string Visit(TextParameter parameter) => "Text";

        public string Visit(FilePathParameter parameter) => "FilePath";

        public string Visit(ColumnParameter parameter) => "Column";

        public string Visit(NewColumnParameter parameter) => "NewColumn";

        public string Visit(ColumnsParameter parameter) => "Columns";

        public string Visit(NumberParameter parameter) => "Number";

        public string Visit(WholeNumberParameter parameter) => "WholeNumber";

        public string Visit(TrueOrFalseParameter parameter) => "TrueOrFalse";

        public string Visit(ShareParameter parameter) => "Share";

        public string Visit<TEnum>(OneOfParameter<TEnum> parameter)
            where TEnum : struct, Enum => "OneOf";

        public string Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
            where TEnum : struct, Enum => "SeveralOf";

        public string Visit(FillStrategyParameter parameter) => "FillStrategy";

        public string Visit(SplitSharesParameter parameter) => "SplitShares";

        public string Visit(ColumnDeclarationsParameter parameter) => "ColumnDeclarations";
    }

    [Fact]
    public void AParameterIsWrittenOnlyWhereAWriterIsGiven()
    {
        Assert.Throws<ArgumentNullException>(() => new TextParameter("key", "Something.", "x").Write(null!, "x"));
        Assert.Throws<ArgumentNullException>(() => new StepParameters<ReadCsvStep>().With<string>(null!, step => step.Path));
        Assert.Throws<ArgumentNullException>(
            () => new StepParameters<ReadCsvStep>().With(new FilePathParameter("path", "Where.", "x.csv"), null!));
    }
}

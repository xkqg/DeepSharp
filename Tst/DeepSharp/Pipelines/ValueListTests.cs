// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A list that compares and hashes by what it holds, so a step holding one is equal to a step holding another list of the same items.
/// </summary>
public class ValueListTests
{
    [Fact]
    public void TwoListsOfTheSameItemsInTheSameOrder_AreEqualAndHashAlike()
    {
        var one = new ValueList<string>(["a", "b"]);
        var other = new ValueList<string>(["a", "b"]);

        Assert.True(one.Equals(other));
        Assert.True(one.Equals((object)other));
        Assert.Equal(one.GetHashCode(), other.GetHashCode());
    }

    [Fact]
    public void TwoListsOfAnotherOrderOrAnotherLength_AreNotEqual()
    {
        var one = new ValueList<int>([1, 2]);

        Assert.False(one.Equals(new ValueList<int>([2, 1])));
        Assert.False(one.Equals(new ValueList<int>([1, 2, 3])));
        Assert.False(one.Equals(new ValueList<int>([])));
    }

    [Fact]
    public void AListIsNotEqualToNothingOrToSomethingElse()
    {
        var one = new ValueList<int>([1]);

        // The same numbers in a list of another kind, handed over as anything at all.
        object another = new List<int> { 1 };

        Assert.False(one.Equals((ValueList<int>?)null));
        Assert.False(one.Equals((object?)null));
        Assert.False(one.Equals(another));
        Assert.True(one.Equals(one));
    }

    [Fact]
    public void TwoEmptyLists_AreEqual()
    {
        Assert.Equal(new ValueList<string>([]), new ValueList<string>([]));
        Assert.Equal(new ValueList<string>([]).GetHashCode(), new ValueList<string>([]).GetHashCode());
    }

    [Fact]
    public void AListReadsLikeAnyOtherReadOnlyList()
    {
        IReadOnlyList<string> list = new ValueList<string>(["x", "y", "z"]);

        Assert.Equal(3, list.Count);
        Assert.Equal("y", list[1]);
        Assert.Equal(["x", "y", "z"], list);
        Assert.Equal(["x", "y", "z"], ((System.Collections.IEnumerable)list).Cast<object>());
    }

    [Fact]
    public void AListCannotBeChanged_ThroughAnyListItIsCastTo()
    {
        var list = (IList<int>)new ValueList<int>([1]);

        Assert.True(list.IsReadOnly);
        Assert.Throws<NotSupportedException>(() => list.Add(2));
        Assert.Throws<NotSupportedException>(() => list[0] = 2);
    }

    [Fact]
    public void TheListOfNothing_IsEmptyAndEqualToAnyOtherEmptyList()
    {
        Assert.Empty(ValueList<int>.None);
        Assert.Equal(new ValueList<int>([]), ValueList<int>.None);
    }

    [Fact]
    public void AListKeepsWhatItWasMadeFrom_EvenWhenTheSourceChangesAfterwards()
    {
        var source = new List<string> { "a" };
        var list = new ValueList<string>(source);

        source.Add("b");

        Assert.Equal(["a"], list);
    }

    [Fact]
    public void AListOfItems_ThatCompareByValueThemselves_IsEqualToo()
    {
        Assert.Equal(
            new ValueList<Pair>([new Pair(1, "a")]),
            new ValueList<Pair>([new Pair(1, "a")]));
    }

    [Fact]
    public void ValuedGivesBackTheSameListWhenItIsOneAlready_AndACopyOtherwise()
    {
        var made = new ValueList<int>([1, 2]);
        IReadOnlyList<int> array = [1, 2];

        Assert.Same(made, made.Valued());
        Assert.NotSame(array, array.Valued());
        Assert.Equal(made, array.Valued());
    }

    [Fact]
    public void ValuedOfNothing_IsRefused()
    {
        IEnumerable<int> nothing = null!;

        Assert.Throws<ArgumentNullException>(() => nothing.Valued());
        Assert.Throws<ArgumentNullException>(() => new ValueList<int>(null!));
    }

    private readonly record struct Pair(int Number, string Word);
}

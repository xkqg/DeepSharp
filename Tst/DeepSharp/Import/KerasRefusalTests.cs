// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using PureHDF;

namespace DeepSharp.Tests.Import;

/// <summary>
/// What a Keras file says that no network here is built of is refused where the file says it — each layer's fault at that
/// layer, every one at once — and numbers that do not fit the layers its description builds are refused by the path of each
/// in the file, by the same rule and in the same words as every other file of numbers. A file that is no model Keras saved
/// is refused as such.
/// </summary>
public class KerasRefusalTests
{
    private const string Titanic = "keras-titanic.keras";

    private const string Images = "keras-images.keras";

    private const string Kinds = "a Keras model is read here when each of its layers is a Dense, Conv2D, BatchNormalization, LayerNormalization, Dropout, Flatten, Reshape, Activation or ReLU.";

    private const string Activations = "linear, relu, tanh and sigmoid, and softmax as the last of a model a categorical cross-entropy trained.";

    [Fact]
    public void AStrideForEachAxis_ADilatedWindow_ChannelsInGroups_AndPooling_AreEachRefusedAtTheirLayer_AllAtOnce()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Open("keras-refused.keras")));

        Assert.Equal(
            [
                "config.json, layer 'conv2d_2' (Conv2D): it strides 2 down and 1 across, and a window here walks one stride down and across alike.",
                "config.json, layer 'conv2d_3' (Conv2D): its window is dilated by 2 down and 2 across, and a window here covers neighbouring places.",
                "config.json, layer 'conv2d_4' (Conv2D): it splits its channels into 2 groups, and a convolution here takes every channel of a place at once.",
                $"config.json, layer 'max_pooling2d' (MaxPooling2D): 'MaxPooling2D' is no kind a network here is built of: {Kinds}",
            ],
            refused.Message.Split(Environment.NewLine));
    }

    [Theory]
    // What the model is, what it takes, and what it holds.
    [InlineData(Titanic, new[] { "class_name=\"Functional\"" }, "config.json: the model is a Functional, and a Keras model is read here when it is a Sequential: a stack of layers, each handed what the one before it made.")]
    [InlineData(Titanic, new[] { "class_name=\"Functional\\r\\nFAKE LINE\"" }, "config.json: the model is a Functional\\r\\nFAKE LINE, and a Keras model is read here when it is a Sequential: a stack of layers, each handed what the one before it made.")]
    [InlineData(Titanic, new[] { "config.layers.0=-", "config.build_input_shape=-" }, "config.json: the model states no shape for its input — it begins with no InputLayer and says no build_input_shape — and every width its layers take is worked out from that shape.")]
    [InlineData(Titanic, new[] { "config.layers.0.config.batch_shape=[null, 14, null]" }, "config.json: the model's input, [null, 14, null], does not state every length of an example, and a network here is built for examples of one shape.")]
    [InlineData(Titanic, new[] { "config.layers.0.config.batch_shape=-" }, "config.json, layer 'input_layer' (InputLayer): it says no 'batch_shape'.")]
    [InlineData(Titanic, new[] { "config.layers.2=-", "config.layers.1=-" }, "config.json: the model holds no layer a network here is built of.")]
    [InlineData(Titanic, new[] { "config.layers=-" }, "config.json: the model holds no layer a network here is built of.")]
    [InlineData(Images, new[] { "config.layers.6=-" }, "config.json: This network cannot be lowered at 84: word 8, dense, takes each example as a row of numbers, and each reaching it is 2x3x3: flatten it first. (Parameter 'input')")]
    // The loss it was trained with.
    [InlineData(Titanic, new[] { "compile_config=-" }, "config.json, compile_config: the model was saved without the loss it was trained with, and a network here answers through its loss.")]
    [InlineData(Titanic, new[] { "compile_config=null" }, "config.json, compile_config: the model was saved without the loss it was trained with, and a network here answers through its loss.")]
    [InlineData(Titanic, new[] { "compile_config.loss=-" }, "config.json, compile_config: the model was saved without the loss it was trained with, and a network here answers through its loss.")]
    [InlineData(Titanic, new[] { "compile_config.loss=\"hinge\"" }, "config.json, compile_config: the loss is hinge, and a Keras model is read here when it was trained with a binary or a categorical cross-entropy or a mean squared error.")]
    [InlineData(Titanic, new[] { "compile_config.loss=[\"binary_crossentropy\"]" }, "config.json, compile_config: the loss is [\"binary_crossentropy\"], and a Keras model is read here when it was trained with a binary or a categorical cross-entropy or a mean squared error.")]
    [InlineData(Titanic, new[] { "compile_config.loss={\"class_name\": \"BinaryCrossentropy\", \"config\": {\"label_smoothing\": 0.1}}" }, "config.json, compile_config: the loss smooths its labels by 0.1, and no loss here smooths them.")]
    [InlineData(Titanic, new[] { "compile_config.loss={\"class_name\": \"BinaryCrossentropy\", \"config\": {\"reduction\": \"sum\"}}" }, "config.json, compile_config: the loss reduces a batch's losses by 'sum', and a loss here takes their mean.")]
    [InlineData(Titanic, new[] { "compile_config.loss={\"class_name\": \"BinaryCrossentropy\", \"config\": {\"from_logits\": \"yes\"}}" }, "config.json, compile_config: its 'from_logits' is written as \"yes\", which is not what Keras writes there.")]
    [InlineData(Titanic, new[] { "compile_config.loss={\"class_name\": \"BinaryCrossentropy\", \"config\": {\"from_logits\": true}}" }, "config.json, layer 'dense_1' (Dense): the model ends in a sigmoid, and its loss takes logits and applies the sigmoid itself: every prediction would go through it twice.")]
    [InlineData(Titanic, new[] { "config.layers.2.config.activation=\"linear\"" }, "config.json, layer 'dense_1' (Dense): its loss takes the chances a last sigmoid gives, and the model ends in linear: a network here ends before the sigmoid its loss applies itself, so it is read when the model ends in one.")]
    // A layer's activation, bias, settings and precision.
    [InlineData(Titanic, new[] { "config.layers.1.config.activation=\"softmax\"" }, $"config.json, layer 'dense' (Dense): its activation, softmax, is none of those read here: {Activations}")]
    [InlineData(Titanic, new[] { "config.layers.1.config.activation=\"gelu\"" }, $"config.json, layer 'dense' (Dense): its activation, gelu, is none of those read here: {Activations}")]
    [InlineData(Titanic, new[] { "config.layers.1.config.use_bias=false" }, "config.json, layer 'dense' (Dense): it adds no bias, and a dense layer or a convolution here always adds one.")]
    [InlineData(Titanic, new[] { "config.layers.1.config.units=-" }, "config.json, layer 'dense' (Dense): it says no 'units'.")]
    [InlineData(Titanic, new[] { "config.layers.1.config.units=\"16\"" }, "config.json, layer 'dense' (Dense): its 'units' is written as \"16\", which is not what Keras writes there.")]
    [InlineData(Titanic, new[] { "config.layers.1.config=-" }, "config.json, layer '' (Dense): it says no 'units'.")]
    [InlineData(Titanic, new[] { "config.layers.1.class_name=-" }, $"config.json, layer 'dense' (): '' is no kind a network here is built of: {Kinds}")]
    [InlineData(Titanic, new[] { "config.layers.1.config.dtype.config.name=\"mixed_float16\"" }, "config.json, layer 'dense' (Dense): it works in mixed_float16, and a network here works in single precision, float32.")]
    [InlineData(Titanic, new[] { "config.layers.1.config.dtype=\"float64\"" }, "config.json, layer 'dense' (Dense): it works in float64, and a network here works in single precision, float32.")]
    [InlineData(Titanic, new[] { "config.layers.1.config.dtype={\"class_name\": \"DTypePolicy\"}" }, "config.json, layer 'dense' (Dense): it works in {\"class_name\":\"DTypePolicy\"}, and a network here works in single precision, float32.")]
    [InlineData(Titanic, new[] { "config.layers.1.config.dtype={\"config\": {}}" }, "config.json, layer 'dense' (Dense): it works in {\"config\":{}}, and a network here works in single precision, float32.")]
    // A convolution.
    [InlineData(Images, new[] { "config.layers.2.config.padding=\"causal\"" }, "config.json, layer 'conv2d' (Conv2D): it pads as 'causal', and a window here pads as 'valid' or 'same'.")]
    [InlineData(Images, new[] { "config.layers.2.config.data_format=\"channels_first\"" }, "config.json, layer 'conv2d' (Conv2D): it lays its images out 'channels_first', and images here are laid out with their channels last.")]
    [InlineData(Images, new[] { "config.layers.2.config.kernel_size=[3]" }, "config.json, layer 'conv2d' (Conv2D): its 'kernel_size' is written as [3], and Keras writes a pair there.")]
    [InlineData(Images, new[] { "config.layers.2.config.kernel_size=-" }, "config.json, layer 'conv2d' (Conv2D): it says no 'kernel_size'.")]
    [InlineData(Images, new[] { "config.layers.2.config.dilation_rate=[1, 2]" }, "config.json, layer 'conv2d' (Conv2D): its window is dilated by 1 down and 2 across, and a window here covers neighbouring places.")]
    [InlineData(Images, new[] { "config.layers.2.config.use_bias=false" }, "config.json, layer 'conv2d' (Conv2D): it adds no bias, and a dense layer or a convolution here always adds one.")]
    // The normalisations.
    [InlineData(Images, new[] { "config.layers.3.config.axis=1" }, "config.json, layer 'batch_normalization' (BatchNormalization): it normalises over axis 1, and a normalisation here works over the last axis.")]
    [InlineData(Images, new[] { "config.layers.3.config.center=false" }, "config.json, layer 'batch_normalization' (BatchNormalization): it leaves out its shift or its scale, and a normalisation here learns both.")]
    [InlineData(Images, new[] { "config.layers.3.config.scale=false" }, "config.json, layer 'batch_normalization' (BatchNormalization): it leaves out its shift or its scale, and a normalisation here learns both.")]
    [InlineData(Images, new[] { "config.layers.10.config.axis=[1]" }, "config.json, layer 'layer_normalization' (LayerNormalization): it normalises over axis [1], and a normalisation here works over the last axis.")]
    [InlineData(Images, new[] { "config.layers.10.config.center=false" }, "config.json, layer 'layer_normalization' (LayerNormalization): it leaves out its shift or its scale, and a normalisation here learns both.")]
    [InlineData(Images, new[] { "config.layers.10.config.rms_scaling=true" }, "config.json, layer 'layer_normalization' (LayerNormalization): it scales by the root mean square alone, which no normalisation here does.")]
    // A dropout, a flatten, a relu and a reshape.
    [InlineData(Images, new[] { "config.layers.7.config.noise_shape=[null, 1]" }, "config.json, layer 'dropout' (Dropout): it leaves values out by a noise shape, [null, 1], and a dropout here leaves out each value on its own.")]
    [InlineData(Images, new[] { "config.layers.7.config.rate=1" }, "config.json, layer 'dropout' (Dropout): A dropout leaves out a share of the values, from nothing to below one. (Parameter 'rate') Actual value was 1.")]
    [InlineData(Images, new[] { "config.layers.6.config.data_format=\"channels_first\"" }, "config.json, layer 'flatten' (Flatten): it lays its images out 'channels_first', and images here are laid out with their channels last.")]
    [InlineData(Images, new[] { "config.layers.5.config.max_value=6" }, "config.json, layer 're_lu' (ReLU): it caps its values, lets some below nothing through or starts above nothing, and a relu here keeps each value above nothing as it is and makes the rest nothing.")]
    [InlineData(Images, new[] { "config.layers.5.config.negative_slope=0.1" }, "config.json, layer 're_lu' (ReLU): it caps its values, lets some below nothing through or starts above nothing, and a relu here keeps each value above nothing as it is and makes the rest nothing.")]
    [InlineData(Images, new[] { "config.layers.5.config.threshold=0.5" }, "config.json, layer 're_lu' (ReLU): it caps its values, lets some below nothing through or starts above nothing, and a relu here keeps each value above nothing as it is and makes the rest nothing.")]
    [InlineData(Images, new[] { "config.layers.1.config.target_shape=[-1, 7, 2]" }, "config.json, layer 'reshape' (Reshape): its target shape, [-1, 7, 2], leaves a length to be worked out, and a reshape here states every length.")]
    [InlineData(Images, new[] { "config.layers.1.config.target_shape=-" }, "config.json, layer 'reshape' (Reshape): it says no 'target_shape'.")]
    // A setting written as what Keras never writes there.
    [InlineData(Titanic, new[] { "config.layers.1.config.units=16.5" }, "config.json, layer 'dense' (Dense): its 'units' is written as 16.5, which is not what Keras writes there.")]
    [InlineData(Titanic, new[] { "config.layers.0.config.batch_shape=\"x\"" }, "config.json, layer 'input_layer' (InputLayer): its 'batch_shape' is written as \"x\", which is not what Keras writes there.")]
    [InlineData(Titanic, new[] { "config.layers.0.config.batch_shape=[null, \"a\"]" }, "config.json, layer 'input_layer' (InputLayer): its 'batch_shape' is written as [null, \"a\"], which is not what Keras writes there.")]
    [InlineData(Images, new[] { "config.layers.3.config.momentum=\"0.9\"" }, "config.json, layer 'batch_normalization' (BatchNormalization): its 'momentum' is written as \"0.9\", which is not what Keras writes there.")]
    [InlineData(Images, new[] { "config.layers.2.config.padding=5" }, "config.json, layer 'conv2d' (Conv2D): its 'padding' is written as 5, which is not what Keras writes there.")]
    [InlineData(Images, new[] { "config.layers.2.config.kernel_size=3" }, "config.json, layer 'conv2d' (Conv2D): its 'kernel_size' is written as 3, which is not what Keras writes there.")]
    [InlineData(Images, new[] { "config.layers.2.config.kernel_size=[2.5, 2]" }, "config.json, layer 'conv2d' (Conv2D): its 'kernel_size' is written as [2.5, 2], which is not what Keras writes there.")]
    [InlineData(Images, new[] { "config.layers.1.config.target_shape=[\"a\", 7, 2]" }, "config.json, layer 'reshape' (Reshape): its 'target_shape' is written as [\"a\", 7, 2], which is not what Keras writes there.")]
    public void WhatADescriptionSays_ThatNoNetworkHereIsBuiltOf_IsRefusedWhereItSaysIt(string file, string[] edits, string refusal)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited(file, edits)));

        Assert.Equal(refusal, refused.Message);
    }

    [Fact]
    public void AClassNameOfExtremeLength_IsCutShortInTheRefusal_SayingHowLongItWas()
    {
        var huge = new string('m', 10_000);

        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited(Titanic, $"class_name=\"{huge}\"")));

        Assert.True(refused.Message.Length < 1000, $"The message is {refused.Message.Length} characters long.");
        Assert.Contains("(10000 characters)", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain(huge, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryFaultOfADescription_IsNamedAtOnce()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited(
            Images, "config.layers.2.config.padding=\"causal\"", "config.layers.3.config.center=false", "config.layers.9.config.activation=\"elu\"")));

        Assert.Equal(
            [
                "config.json, layer 'conv2d' (Conv2D): it pads as 'causal', and a window here pads as 'valid' or 'same'.",
                "config.json, layer 'batch_normalization' (BatchNormalization): it leaves out its shift or its scale, and a normalisation here learns both.",
                $"config.json, layer 'activation' (Activation): its activation, elu, is none of those read here: {Activations}",
            ],
            refused.Message.Split(Environment.NewLine));
    }

    [Fact]
    public void NumbersOfOtherLengths_ThanTheLayersTheDescriptionBuilds_AreRefusedAtTheirDatasets_InTheLoadsOwnWords()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.Edited(Titanic, "config.layers.1.config.units=8")));

        Assert.Equal(
            [
                new SlotLoadFault("model.weights.h5, layers/dense/vars/0", "0.weight", "'0.weight' is a 14x8 slot here, and is written as 14x16."),
                new SlotLoadFault("model.weights.h5, layers/dense/vars/1", "0.bias", "'0.bias' is a 8 slot here, and is written as 16."),
                new SlotLoadFault("model.weights.h5, layers/dense_1/vars/0", "2.weight", "'2.weight' is a 8x1 slot here, and is written as 16x1."),
            ],
            Sorted(refused.Faults));
    }

    [Fact]
    public void AKernelWrittenForAnotherWindow_IsRefusedAsWritten_RatherThanLaidOutForThisOne()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.Edited(Images, "config.layers.2.config.kernel_size=[2, 3]")));

        Assert.Equal(
            [new SlotLoadFault("model.weights.h5, layers/conv2d/vars/0", "1.weight", "'1.weight' is a 12x4 slot here, and is written as 3x2x2x4.")],
            refused.Faults);
    }

    [Fact]
    public void NumbersOfAnotherPrecision_BeyondALayersSlots_OfALayerTheModelDoesNotName_OrMissing_AreEachNamedWhereTheFileHoldsThem()
    {
        var weights = new H5File
        {
            ["layers"] = new H5Group
            {
                ["dense"] = new H5Group
                {
                    ["vars"] = new H5Group
                    {
                        ["0"] = new H5Dataset<double[]>(new double[14 * 16], fileDims: [14, 16]),
                        ["1"] = KerasFixtures.Floats([16], 0.5f),
                        ["2"] = KerasFixtures.Floats([3], 0.5f),
                        Attributes = { ["name"] = "dense" },
                    },
                },
                ["dense_1"] = new H5Group
                {
                    ["vars"] = new H5Group { ["0"] = new H5Dataset<int[]>(new int[16], fileDims: [16, 1]), Attributes = { ["name"] = "dense_1" } },
                },
                ["extra"] = new H5Group
                {
                    ["vars"] = new H5Group { ["0"] = KerasFixtures.Floats([2], 1f), Attributes = { ["name"] = "dense_9" } },
                },
                ["unnamed"] = new H5Group
                {
                    ["vars"] = new H5Group { ["0"] = KerasFixtures.Floats([1], 1f) },
                },
                ["weightless"] = new H5Group(),
            },
        };

        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.WithWeights(Titanic, weights)));

        Assert.Equal(
            [
                new SlotLoadFault(null, "2.bias", "'2.bias' is missing: every slot of the network is written."),
                new SlotLoadFault("model.weights.h5, layers/dense/vars/0", "0.weight", "'0.weight' holds single-precision numbers here, and is written as FloatingPoint of 8 bytes each."),
                new SlotLoadFault("model.weights.h5, layers/dense/vars/2", "dense.2", "'dense.2' is no slot of this network."),
                new SlotLoadFault("model.weights.h5, layers/dense_1/vars/0", "2.weight", "'2.weight' holds single-precision numbers here, and is written as FixedPoint of 4 bytes each."),
                new SlotLoadFault("model.weights.h5, layers/extra/vars/0", "dense_9.0", "'dense_9.0' is no slot of this network."),
                new SlotLoadFault("model.weights.h5, layers/unnamed/vars/0", "layers/unnamed.0", "'layers/unnamed.0' is no slot of this network."),
            ],
            Sorted(refused.Faults));
    }

    [Fact]
    public void AWeightsFileHoldingNoLayers_LeavesEverySlotMissing()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.WithWeights(Titanic, new H5File())));

        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias"], refused.Faults.Select(fault => fault.Slot));
        Assert.All(refused.Faults, fault => Assert.Null(fault.Source));
    }

    [Fact]
    public void AnHdf5FileKerasSavedAModelTo_IsHeldToTheSameRules_ItsNumbersByTheWeightNamesOfEachLayer()
    {
        var weights = new H5Group
        {
            ["dense"] = new H5Group
            {
                ["kernel"] = KerasFixtures.Floats([14, 16], 0.5f),
                ["bias"] = KerasFixtures.Floats([16], 0.5f),
                Attributes = { ["weight_names"] = new[] { "kernel", "bias", "gone" } },
            },
            ["dense_1"] = new H5Group { ["kernel"] = KerasFixtures.Floats([16, 1], 0.5f) },
            ["top_level_model_weights"] = new H5Group
            {
                ["extra"] = KerasFixtures.Floats([1], 0.5f),
                Attributes = { ["weight_names"] = new[] { "extra" } },
            },
        };

        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(Legacy(TrainingConfig, weights)));

        Assert.Equal(
            [
                new SlotLoadFault(null, "2.bias", "'2.bias' is missing: every slot of the network is written."),
                new SlotLoadFault(null, "2.weight", "'2.weight' is missing: every slot of the network is written."),
                new SlotLoadFault("model_weights/top_level_model_weights/extra", "top_level_model_weights.extra", "'top_level_model_weights.extra' is no slot of this network."),
            ],
            Sorted(refused.Faults));
    }

    [Fact]
    public void AnHdf5FileHoldingNoWeightsGroup_LeavesEverySlotMissing()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(Legacy(TrainingConfig, weights: null)));

        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias"], refused.Faults.Select(fault => fault.Slot));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("{\"optimizer_config\": {}}")]
    public void AnHdf5FileSavedWithoutItsLoss_IsRefusedAtItsTrainingConfig(string? trainingConfig)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(Legacy(trainingConfig, weights: null)));

        Assert.Equal("training_config: the model was saved without the loss it was trained with, and a network here answers through its loss.", refused.Message);
    }

    [Fact]
    public void AnHdf5FileOfNumbersAlone_IsRefusedAsNoModelKerasSavedWhole()
    {
        var file = new H5File { ["model_weights"] = new H5Group() };

        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(new MemoryStream(KerasFixtures.Written(file))));

        Assert.Equal(
            "This HDF5 file holds no model_config: a model Keras saved whole describes itself there, and a file of numbers alone names no layers to hold them.",
            refused.Message);
    }

    [Fact]
    public void AFileThatIsNeitherAKerasArchiveNorAnHdf5File_IsRefusedAsSuch()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(new MemoryStream(Encoding.UTF8.GetBytes("{\"network\": {}}"))));

        Assert.Equal(
            "A Keras file is the archive Keras 3 saves a model to, .keras, or the HDF5 file it saved one to before, .h5, and this is neither.",
            refused.Message);
        Assert.Throws<FormatException>(() => new KerasFile().Read(new MemoryStream()));
        Assert.Throws<ArgumentNullException>(() => new KerasFile().Read(null!));
    }

    [Theory]
    [InlineData("config.json", "This archive holds no config.json, where Keras 3 describes a model.")]
    [InlineData("model.weights.h5", "This archive holds no model.weights.h5, where Keras 3 keeps a model's numbers: a model whose numbers were kept otherwise is not read here.")]
    public void AnArchiveWithoutWhatKerasWritesIntoOne_IsRefusedNamingWhatIsMissing(string missing, string refusal)
    {
        var entries = KerasFixtures.Entries(Titanic);
        entries.Remove(missing);

        Assert.Equal(refusal, Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Archive(entries))).Message);
    }

    [Fact]
    public void ADescriptionThatIsNotJson_IsRefusedWhereItStands()
    {
        var entries = KerasFixtures.Entries(Titanic);
        entries["config.json"] = Encoding.UTF8.GetBytes("a model");

        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Archive(entries)));

        Assert.StartsWith("config.json is not JSON: ", refused.Message, StringComparison.Ordinal);
    }

    // The faults in an order of their own, where the file holds them and then the slot, so a test names them regardless of
    // the order the file lists its groups in: a slot the file holds no number for first.
    private static SlotLoadFault[] Sorted(IEnumerable<SlotLoadFault> faults) =>
        [.. faults.OrderBy(fault => fault.Source ?? string.Empty, StringComparer.Ordinal).ThenBy(fault => fault.Slot, StringComparer.Ordinal)];

    // The Titanic model's training_config, as Keras saved it to an HDF5 file.
    private static string TrainingConfig => Saved("training_config");

    // An attribute of the HDF5 file Keras saved the Titanic model to.
    private static string Saved(string attribute)
    {
        using var saved = H5File.Open(KerasFixtures.Open("keras-titanic.h5"));

        return saved.Attribute(attribute).Read<string>();
    }

    // An HDF5 file as Keras saved the Titanic model to one before: its description, the training_config given, or none, and
    // the numbers given, or none.
    private static MemoryStream Legacy(string? trainingConfig, H5Group? weights)
    {
        var file = new H5File { Attributes = { ["model_config"] = Saved("model_config") } };

        if (trainingConfig is not null)
        {
            file.Attributes["training_config"] = trainingConfig;
        }

        if (weights is not null)
        {
            file["model_weights"] = weights;
        }

        return new MemoryStream(KerasFixtures.Written(file));
    }
}

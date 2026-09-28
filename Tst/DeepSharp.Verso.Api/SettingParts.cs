// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>A toolbar button with a setting of its own, which a notebook saves for it — as any part may have.</summary>
internal sealed class SettingPart : IToolbarAction, IExtensionSettings
{
    public const string Id = "deepsharp.tests.setting-part";

    public const string Plain = "plain";

    public string ExtensionId => Id;

    public string Name => "A button with a colour of its own";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string ActionId => "deepsharp.tests.setting";

    public string DisplayName => "Setting";

    public string? Icon => null;

    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    public int Order => 0;

    /// <summary>Its setting as it stands.</summary>
    public string Colour { get; private set; } = Plain;

    public IReadOnlyList<SettingDefinition> SettingDefinitions => [new SettingDefinition("colour", "Colour", "What colour it is.", SettingType.String, Plain)];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task<bool> IsEnabledAsync(IToolbarActionContext context) => Task.FromResult(false);

    public Task ExecuteAsync(IToolbarActionContext context) => Task.CompletedTask;

    public IReadOnlyDictionary<string, object?> GetSettingValues() => new Dictionary<string, object?> { ["colour"] = Colour };

    public Task ApplySettingsAsync(IReadOnlyDictionary<string, object?> values)
    {
        if (values.TryGetValue("colour", out var colour))
        {
            Colour = colour is JsonElement saved ? saved.GetString()! : (string)colour!;
        }

        return Task.CompletedTask;
    }

    public Task OnSettingChangedAsync(string name, object? value)
    {
        Colour = (string)value!;

        return Task.CompletedTask;
    }
}

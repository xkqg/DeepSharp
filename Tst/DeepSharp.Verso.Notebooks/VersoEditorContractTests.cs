// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// Verso's browser editor, the one `verso serve` starts, asks no extension anything when it opens a file or saves one: it
/// reads a file with the format's own reader and writes it with the format's own writer, where Verso's VS Code host and
/// DeepSharp's own server ask every extension first. So the guard that keeps a notebook of blocks in .verso never runs
/// there — a Jupyter copy of one opens as text, and saving it writes a .verso over the one beside it. That is written down
/// as a limit; this reads the editor's own code and pins it, so a Verso whose editor starts asking is noticed.
/// </summary>
public sealed class VersoEditorContractTests
{
    private static readonly string Editor = Path.Join(AppContext.BaseDirectory, "verso-editor", "Verso.Blazor.dll");

    [Fact]
    public void VersosBrowserEditor_AsksNoExtensionWhenItOpensOrSavesAFile_ThoughItAsksWhenItComparesWithTheFileSaved()
    {
        using var file = File.OpenRead(Editor);
        using var editor = new PEReader(file);
        var metadata = editor.GetMetadataReader();

        // The one place the editor does ask, reading the file a notebook is compared with: the scan finds a call when
        // there is one to find.
        Assert.Contains("PostDeserializeAsync", Called(editor, metadata, "<DeserializeBaselineAsync>"));

        // Opening reads the file with the format's reader and asks nothing after it; saving writes with the format's
        // writer and asks nothing before it.
        var opening = Called(editor, metadata, "<OpenAsync>");
        var saving = Called(editor, metadata, "<PrepareSerializedContentAsync>");

        Assert.Contains("DeserializeAsync", opening);
        Assert.DoesNotContain("PostDeserializeAsync", opening);
        Assert.Contains("SerializeAsync", saving);
        Assert.DoesNotContain("PreSerializeAsync", saving);
    }

    // The names of every method the steps of an async method of the editor's notebook service call: the compiler writes
    // an async method's body into a type of its own, named after it.
    private static HashSet<string> Called(PEReader editor, MetadataReader metadata, string asyncMethod)
    {
        var service = metadata.TypeDefinitions.Select(metadata.GetTypeDefinition)
            .Single(type => metadata.GetString(type.Name) == "ServerNotebookService" && metadata.GetString(type.Namespace).StartsWith("Verso.Blazor", StringComparison.Ordinal));
        var body = service.GetNestedTypes().Select(metadata.GetTypeDefinition)
            .Where(type => metadata.GetString(type.Name).StartsWith(asyncMethod, StringComparison.Ordinal))
            .SelectMany(type => type.GetMethods().Select(metadata.GetMethodDefinition))
            .Single(method => metadata.GetString(method.Name) == "MoveNext");
        var il = editor.GetMethodBody(body.RelativeVirtualAddress).GetILBytes()!;
        var called = new HashSet<string>(StringComparer.Ordinal);

        // A call, a virtual call and a constructor call are each one byte followed by the token of what they call.
        for (var at = 0; at + 4 < il.Length; at++)
        {
            if (il[at] is 0x28 or 0x6F or 0x73 && Named(metadata, BitConverter.ToInt32(il, at + 1)) is { } name)
            {
                called.Add(name);
            }
        }

        return called;
    }

    private static string? Named(MetadataReader metadata, int token)
    {
        var handle = MetadataTokens.EntityHandle(token);

        return handle.Kind switch
        {
            HandleKind.MemberReference when MetadataTokens.GetRowNumber(handle) <= metadata.GetTableRowCount(TableIndex.MemberRef)
                => metadata.GetString(metadata.GetMemberReference((MemberReferenceHandle)handle).Name),
            HandleKind.MethodDefinition when MetadataTokens.GetRowNumber(handle) <= metadata.GetTableRowCount(TableIndex.MethodDef)
                => metadata.GetString(metadata.GetMethodDefinition((MethodDefinitionHandle)handle).Name),
            _ => null,
        };
    }
}

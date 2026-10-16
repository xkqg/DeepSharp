// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// Verso's browser editor, the one `verso serve` starts, asks every extension that takes a format when it opens a file
/// and when it saves one, as Verso's VS Code host and DeepSharp's own server do, so the guard that keeps a notebook of
/// blocks out of a format that loses them runs there too. Until Verso 1.2.3 it asked nothing either way, and a notebook of
/// blocks could be saved there in any format; this reads the editor's own code and pins what it does now, so a Verso
/// whose editor stops asking is noticed.
/// </summary>
public sealed class VersoEditorContractTests
{
    private static readonly string Editor = Path.Join(AppContext.BaseDirectory, "verso-editor", "Verso.Blazor.dll");

    [Fact]
    public void VersosBrowserEditor_AsksEveryExtensionWhenItOpensOrSavesAFile_AndWhenItComparesWithTheFileSaved()
    {
        using var file = File.OpenRead(Editor);
        using var editor = new PEReader(file);
        var metadata = editor.GetMetadataReader();

        // Opening reads the file with the format's reader and then hands what it read to every post-processor that takes
        // the format; saving hands the notebook to them first and then writes it; and the file a notebook is compared with
        // is read the way it is opened.
        var opening = Called(editor, metadata, "<OpenAsync>");
        var saving = Called(editor, metadata, "<PrepareSerializedContentAsync>");
        var comparing = Called(editor, metadata, "<DeserializeBaselineAsync>");

        Assert.Contains("DeserializeAsync", opening);
        Assert.Contains("AfterDeserializeAsync", opening);
        Assert.Contains("SerializeAsync", saving);
        Assert.Contains("BeforeSerializeAsync", saving);
        Assert.Contains("AfterDeserializeAsync", comparing);
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
        // Only a method's own row or a reference to one names a method; any other four bytes after the opcode are no token
        // of a call, and asking the reader to make a handle of them throws.
        if ((token >> 24) is not (0x06 or 0x0A))
        {
            return null;
        }

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

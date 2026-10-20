// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Emit;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// Every method one method calls, read off its compiled body: what the source says is what the compiler wrote down, and a
/// conversion the compiler inserts — a number into the scalar libtorch takes — is a call nobody typed. Calls made through a
/// delegate or a lambda are the lambda's own: its body is a method of a class the compiler wrote.
/// </summary>
internal sealed class CallSites(MethodBase method)
{
    private static readonly Dictionary<short, OpCode> Codes = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(field => (OpCode)field.GetValue(null)!)
        .ToDictionary(code => code.Value);

    /// <summary>The methods this one calls, constructors included, in the order they are called.</summary>
    /// <returns>Each method.</returns>
    public IReadOnlyList<MethodBase> Calls()
    {
        var body = method.GetMethodBody();

        if (body?.GetILAsByteArray() is not { } il)
        {
            return [];
        }

        var typeArguments = method.DeclaringType is { IsGenericType: true } owner ? owner.GetGenericArguments() : null;
        var methodArguments = method.IsGenericMethod ? method.GetGenericArguments() : null;
        var calls = new List<MethodBase>();

        for (var at = 0; at < il.Length;)
        {
            var code = il[at] == 0xFE ? Codes[(short)(0xFE00 | il[at + 1])] : Codes[il[at]];

            at += code.Size;

            if (code.OperandType == OperandType.InlineMethod)
            {
                try
                {
                    calls.Add(method.Module.ResolveMethod(BitConverter.ToInt32(il, at), typeArguments, methodArguments)!);
                }
                catch (ArgumentException)
                {
                    // A token that names nothing this module can resolve is a call to nobody a test could ban.
                }
            }

            at += code.OperandType switch
            {
                OperandType.InlineNone => 0,
                OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
                OperandType.InlineVar => 2,
                OperandType.InlineI8 or OperandType.InlineR => 8,
                OperandType.InlineSwitch => 4 + (4 * BitConverter.ToInt32(il, at)),
                _ => 4,
            };
        }

        return calls;
    }

    /// <summary>Every method of a type and of the classes the compiler wrote inside it for its lambdas.</summary>
    /// <param name="type">The type.</param>
    /// <returns>Each method, with its calls.</returns>
    public static IEnumerable<CallSites> Within(Type type) =>
        type.GetNestedTypes(BindingFlags.NonPublic | BindingFlags.Public)
            .SelectMany(Within)
            .Concat(type.GetMembers(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .OfType<MethodBase>()
                .Select(found => new CallSites(found)));

    /// <summary>Names the method the calls are made in.</summary>
    /// <returns>The declaring type and the method.</returns>
    public override string ToString() => $"{method.DeclaringType?.Name}.{method.Name}";
}

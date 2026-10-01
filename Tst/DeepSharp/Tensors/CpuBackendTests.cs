// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// The one that ships: .NET's own SIMD tensor primitives, no native library behind it. What every engine is held to runs
/// on it from the contract; this is what is its own — its name, its version and its device, and totals kept in double
/// precision.
/// </summary>
public class CpuBackendTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void TheBackend_SaysWhichOneItIs()
    {
        Assert.Equal("cpu", _backend.Name);
    }

    [Fact]
    public void TheBackend_NamesDeepSharpsOwnVersion_ForItsArithmeticIsDeepSharpsOwn_AndThisMachinesProcessor()
    {
        // The version a release is handed, the one Directory.Build.props declares, as the package names it: without the
        // commit Source Link adds to what the assembly says of itself, so every build of one release names the same one.
        var named = Assert.IsAssignableFrom<INamesItsVersionAndDevice>(_backend);
        var declared = XDocument.Load(Path.Join(Repository.Root, "Directory.Build.props")).Descendants("Version").Single().Value.Trim();
        var said = typeof(CpuBackend).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion;

        Assert.Equal(declared, named.Version);
        Assert.Matches($@"^{Regex.Escape(declared)}(\+.+)?$", said);
        Assert.Equal("cpu", named.Device);
    }

    [Fact]
    public void TheMeanOfTenMillionTenths_IsATenth()
    {
        // Added up in single precision, the running total stops taking small values in once it is large and the
        // mean comes out as 0.1087937; the total is kept in double precision so it does not.
        var tenths = _backend.Fill(new Shape(10_000_000), 0.1f);

        Assert.Equal(0.1f, _backend.Mean(tenths).Values[0]);
    }
}

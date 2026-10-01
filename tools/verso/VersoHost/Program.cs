// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// A Verso host of its own, holding Verso's engine and nothing of DeepSharp. It has Verso's own installer install the
// notebook package just made, as the Extensions panel installs a package from a file, and Verso's own loader load it; then
// it opens a notebook as Verso's editors do and asks it what a person asks: "Show the data here" on its report block, its
// C# cell run in Verso's own kernel, and the report block again. Last, it asks the notebook's own load context where each
// assembly the notebook refers to comes from, and compiles every method of DeepSharp's against what it finds. It says what
// it found a line at a time, and ends with 1 when anything was otherwise.
//
//   VersoHost <the notebook package> <a notebook> <the folder of the lists of what an install lays out>
//
// tools/verso/check.sh starts it in the notebook's folder, whose nuget.config names where the C# cell's packages come from,
// and with a temporary folder of its own, where Verso keeps what it downloads and this host keeps its install.

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Verso;
using Verso.Abstractions;
using Verso.Execution;
using Verso.Extensions;
using Verso.Extensions.Marketplace;
using Verso.Serializers;

if (args is not [var package, var notebookFile, var lists])
{
    Console.Error.WriteLine("VersoHost <the notebook package> <a notebook> <the folder of the lists of what an install lays out>");
    return 2;
}

// The notebook package's own names for its parts. Nothing of the package is referenced here, so they are written out.
const string Renderer = "io.github.xkqg.deepsharp.notebooks.renderer";
const string Block = "deepsharp.step";
const string Show = "deepsharp.show";

var failures = 0;
var temp = Path.GetTempPath();

// Nothing of DeepSharp in this process or beside it, and nothing downloaded yet: whatever the notebook finds, it finds in
// what Verso installs for it.
string[] preloaded =
[
    .. AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetName().Name!)
        .Concat(Directory.GetFiles(AppContext.BaseDirectory, "*.dll").Select(file => Path.GetFileNameWithoutExtension(file)))
        .Where(name => name.StartsWith("DeepSharp", StringComparison.Ordinal) || name.StartsWith("MatPlotLibNet", StringComparison.Ordinal)
                       || name == "System.Numerics.Tensors")
        .Distinct(),
];

if (preloaded.Length > 0 || Directory.Exists(Path.Join(temp, "verso-nuget-packages")))
{
    Fails(preloaded.Length > 0
        ? $"This host already holds {string.Join(", ", preloaded)}, so the notebook would not depend on its install alone."
        : $"Verso has downloaded packages into {temp} before installing anything: start this with a temporary folder of its own.");
    return 1;
}

Holds($"Verso {typeof(ExtensionHost).Assembly.GetName().Version!.ToString(3)} on .NET {Environment.Version}, holding nothing of DeepSharp, with nothing downloaded");

// Installed as the Extensions panel installs a package from a file: asked for, trusted, installed with every package it
// depends on into Verso's folder for it, and loaded by Verso's own loader, each file in a context of its own.
var managed = Path.Join(temp, "extensions");
var asked = new List<string>();
await using var host = new ExtensionHost();

host.ConsentHandler = (extensions, _) =>
{
    asked.AddRange(extensions.Select(extension => $"{extension.PackageId} {extension.Version}"));
    return Task.FromResult(true);
};

await host.LoadBuiltInExtensionsAsync();

MarketplaceLoader.LocalInstallOutcome installed;

try
{
    installed = await MarketplaceLoader.InstallLocalFileAsync(
        host, new NuGetMarketplaceService(), ExtensionTrustStore.Load(Path.Join(temp, "trust.json")), package, managed);
}
catch (ReflectionTypeLoadException refused)
{
    Fails($"Verso's loader could not load what its installer laid out: {string.Join(" | ", refused.LoaderExceptions.Select(fault => fault?.Message).Distinct().Take(5))}");
    return 1;
}

if (!installed.Success || installed.PackageId is null || installed.ResolvedVersion is null)
{
    Fails($"Verso's installer did not install {Path.GetFileName(package)}: {installed.ErrorMessage}");
    return 1;
}

Holds($"installed by Verso's own installer once asked for {string.Join(", ", asked)}, and loaded by its own loader: {installed.ExtensionsRegistered} parts");

// Verso keeps one install for each runtime, in a folder named after it, and the list for that folder names every file.
if (Directory.GetDirectories(Path.Join(managed, installed.PackageId, installed.ResolvedVersion)) is not [var folder])
{
    Fails($"Verso's installer made no one folder for this runtime under {Path.Join(managed, installed.PackageId, installed.ResolvedVersion)}");
    return 1;
}

var runtime = Path.GetFileName(folder);
string[] laidOut = [.. Directory.GetFiles(folder, "*", SearchOption.AllDirectories)
    .Select(file => Path.GetRelativePath(folder, file).Replace('\\', '/')).Order(StringComparer.Ordinal)];
var list = Path.Join(lists, $"installed.{runtime}.txt");
string[] listed = File.Exists(list) ? [.. File.ReadLines(list).Where(line => line.Length > 0)] : [];
string[] missing = [.. listed.Except(laidOut)];
string[] unlisted = [.. laidOut.Except(listed)];

if (missing.Length + unlisted.Length > 0)
{
    Fails($"Verso laid the package out for {runtime} otherwise than {list} says. Not laid out: {Named(missing)}. Laid out and not named: {Named(unlisted)}."
          + " The package lost or gained a dependency: if it was meant to, write the list again from what Verso laid out, and the notebook's PackageTests hold it to what the build resolves.");
}
else
{
    Holds($"laid out for {runtime} as installed.{runtime}.txt says: {laidOut.Length} files, {VersionOf("DeepSharp.dll")} and {VersionOf("System.Numerics.Tensors.dll")} among them");
}

var contexts = AssemblyLoadContext.All.ToArray();
string[] unloaded = [.. laidOut.Where(file => file.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
    .Where(file => !contexts.Any(context => context.Name == $"VersoExt:{Path.GetFileNameWithoutExtension(file)}"
                                            && context.Assemblies.Any(assembly => From(assembly, Path.Join(folder, file)))))];

if (unloaded.Length > 0)
{
    Fails($"Verso's loader did not load {Named(unloaded)} from where its installer laid them out.");
}
else
{
    Holds("every assembly laid out loaded by Verso's own loader, each in a context of its own, from where it was laid out");
}

// The notebook, opened as Verso's editors open one: through the serializer of its format, beside its file.
var path = Path.GetFullPath(notebookFile);
await using var scaffold = new Scaffold(await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(path)), host, path);

scaffold.InitializeSubsystems();

var handler = host.GetInteractionHandler(Renderer);
CellModel[] blocks = [.. scaffold.Cells.Where(cell => cell.Type == Block)];

if (handler is null || AssemblyLoadContext.GetLoadContext(handler.GetType().Assembly) is not { } notebook || notebook == AssemblyLoadContext.Default
    || blocks.Length == 0 || scaffold.Cells.Where(cell => cell.Type == "code").ToArray() is not [var trains])
{
    Fails($"{Path.GetFileName(path)} opened without the installed package's blocks and one C# cell to run.");
    return 1;
}

ICellInteractionHandler part = handler;
var report = blocks[^1];

Holds($"{Path.GetFileName(path)} opened through Verso's own serializer: {blocks.Length} blocks and a C# cell, the blocks drawn by the installed package's part");

// "Show the data here" on the report block, as a click reaches the part: its card, the rows there, and — before any cell
// has trained a model — where its measures come from.
async Task<CellOutput[]> ShownAsync()
{
    await part.OnCellInteractionAsync(new CellInteractionContext
    {
        CellId = report.Id,
        ExtensionId = Renderer,
        InteractionType = Show,
        Payload = string.Empty,
        Region = CellRegion.Output,
        Variables = scaffold.Variables,
        Notebook = scaffold.NotebookOps,
        NotebookModel = scaffold.Notebook,
    });

    return [.. report.Outputs];
}

var shown = await ShownAsync();

if (shown.Length != 3 || shown.Any(output => output.IsError))
{
    Fails($"\"Show the data here\" on the report block showed {Described(shown)}");
}
else
{
    Holds("\"Show the data here\" on the report block: its card, the rows there, and where its measures come from");
}

// The C# cell names DeepSharp's packages by their NuGet ids, reads the pipeline the blocks handed over, trains a network,
// hands its predictions back, and ends with the loss curve.
var ran = await scaffold.ExecuteCellAsync(trains.Id);
var curve = trains.Outputs.FirstOrDefault(output => output.Content.Contains("verso-svg-output", StringComparison.Ordinal)
                                                    && output.Content.Contains("<svg", StringComparison.Ordinal));

if (ran.Status != ExecutionResult.ExecutionStatus.Success || trains.Outputs.Any(output => output.IsError) || curve is null)
{
    Fails($"The C# cell ran {ran.Status} and showed {Described([.. trains.Outputs])}");
}
else
{
    Holds($"the C# cell, in Verso's own kernel, trained a network on the packages as the cell names them and drew its loss as Verso draws a picture: {curve.Content.Length} characters");
}

shown = await ShownAsync();

if (shown is not [_, _, var drawn] || drawn.IsError || !drawn.Content.Contains("<svg", StringComparison.Ordinal))
{
    Fails($"\"Show the data here\" on the report block, after the cell handed its predictions back, showed {Described(shown)}");
}
else
{
    Holds($"the report block again: the measures the cell handed back, drawn: {drawn.Content.Length} characters");
}

// Where the notebook's own context finds each assembly the notebook refers to, however far down: in the folder Verso
// installed it to, or — Verso's abstractions — in the host, or in the runtime. Anything else was found only because this
// host downloaded it this session, and a host that did not has none.
var runtimeFolder = RuntimeEnvironment.GetRuntimeDirectory();
var abstractions = typeof(IExtension).Assembly;
var bound = new List<Assembly> { handler.GetType().Assembly };
var outside = new List<string>();
var named = new HashSet<string>(StringComparer.Ordinal);
var next = new Queue<Assembly>(bound);

while (next.TryDequeue(out var assembly))
{
    foreach (var reference in assembly.GetReferencedAssemblies().Where(reference => named.Add(reference.Name!)))
    {
        try
        {
            var found = notebook.LoadFromAssemblyName(reference);

            if (AssemblyLoadContext.GetLoadContext(found) == notebook && From(found, folder))
            {
                bound.Add(found);
                next.Enqueue(found);
            }
            else if (found != abstractions && !From(found, runtimeFolder))
            {
                outside.Add($"{reference.Name} from {found.Location}");
            }
        }
        catch (Exception fault) when (fault is FileNotFoundException or FileLoadException or BadImageFormatException)
        {
            outside.Add($"{reference.Name}, which it cannot load ({fault.Message})");
        }
    }
}

if (outside.Count > 0)
{
    Fails($"The notebook took from outside the folder Verso installed it to: {string.Join("; ", outside)}. The package does not carry what it needs.");
}
else
{
    Holds($"the notebook's own context found all {bound.Count} of its assemblies in the folder Verso installed it to, {BoundVersion("DeepSharp")} and {BoundVersion("System.Numerics.Tensors")} among them");
}

// Every method of DeepSharp's own assemblies, compiled against what the notebook's context found: a member only another
// version of a dependency has fails here rather than in the middle of somebody's notebook.
const BindingFlags Declared = BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
var compiled = 0;
var uncompiled = new List<string>();

foreach (var type in bound.Where(assembly => assembly.GetName().Name!.StartsWith("DeepSharp", StringComparison.Ordinal))
             .SelectMany(TypesOf).Where(type => !type.ContainsGenericParameters))
{
    foreach (var method in type.GetMethods(Declared).Cast<MethodBase>().Concat(type.GetConstructors(Declared))
                 .Where(method => !method.IsAbstract && !method.ContainsGenericParameters && method.GetMethodBody() is not null))
    {
        try
        {
            RuntimeHelpers.PrepareMethod(method.MethodHandle);
            compiled++;
        }
        catch (Exception fault) when (fault is TypeLoadException or MissingMemberException or FileNotFoundException or FileLoadException or BadImageFormatException)
        {
            uncompiled.Add($"{type.FullName}.{method.Name} ({fault.Message})");
        }
    }
}

if (uncompiled.Count > 0)
{
    Fails($"{uncompiled.Count} methods of DeepSharp's do not compile against what the install holds: {string.Join("; ", uncompiled.Take(5))}");
}
else
{
    Holds($"every method of DeepSharp's own assemblies compiled against what the install holds: {compiled}");
}

return failures == 0 ? 0 : 1;

void Holds(string what) => Console.WriteLine($"    {what}");

// A line GitHub's runner shows as an error, and a failure this host ends with.
void Fails(string what)
{
    failures++;
    Console.WriteLine($"::error::{what}");
}

string Named(string[] files) => files.Length == 0 ? "none" : string.Join(", ", files);

string VersionOf(string file) => $"{Path.GetFileNameWithoutExtension(file)} {AssemblyName.GetAssemblyName(Path.Join(folder, file)).Version}";

string BoundVersion(string name) => $"{name} {bound.Single(assembly => assembly.GetName().Name == name).GetName().Version}";

bool From(Assembly assembly, string place) =>
    assembly.Location.Length > 0 && Path.GetFullPath(assembly.Location).StartsWith(Path.GetFullPath(place), OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

static string Described(CellOutput[] outputs) =>
    $"{outputs.Length} outputs{string.Concat(outputs.Where(output => output.IsError).Take(1).Select(output => $", the first error: {output.Content.ReplaceLineEndings(" ")}"))}";

static IEnumerable<Type> TypesOf(Assembly assembly)
{
    try
    {
        return assembly.GetTypes();
    }
    catch (ReflectionTypeLoadException partly)
    {
        return partly.Types.OfType<Type>();
    }
}

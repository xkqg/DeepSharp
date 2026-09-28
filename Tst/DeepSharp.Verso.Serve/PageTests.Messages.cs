// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tests.Serve.Parts;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// What the page says, as Verso's editor keeps it: a notice — saved, a kernel started afresh — gone after three seconds, a
// newer one starting the three again; a sentence that stands while what it says holds; and an error in a banner of its
// own, which a notice never takes away, kept until it is dismissed or another takes its place, and cleared as a save
// begins.
public sealed partial class PageTests
{
    [Fact]
    public async Task ANotice_GoesAfterThreeSeconds_ANewerOneStartingTheThreeAgain()
    {
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(Code("1 + 1"));
        await SaveAsync("noticed.verso", notebook);
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var notice = page.Locator("#notice");

        page.Dialog += async (_, dialog) => await dialog.AcceptAsync();

        // The page's clock stands still once the notebook is drawn, and runs on only as far as the test says.
        await page.Clock.InstallAsync(new() { TimeDate = DateTime.UnixEpoch });
        await page.GotoAsync($"{served.Address}?token={Token}&notebook=noticed.verso");
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(1);
        await page.Clock.PauseAtAsync(DateTime.UnixEpoch.AddHours(1));

        await page.Locator("#save").ClickAsync();
        await Expect(notice).ToHaveTextAsync("Saved to noticed.verso");

        await page.Clock.RunForAsync(2_000);
        await Expect(notice).ToHaveTextAsync("Saved to noticed.verso");

        // A newer notice two seconds on stands three seconds from then, not from the first.
        await page.Locator("#toolbar button[data-button='verso.action.restart-kernel']").ClickAsync();
        await Expect(notice).ToHaveTextAsync("Kernel restarted", new() { Timeout = 30_000 });
        await page.Clock.RunForAsync(2_000);
        await Expect(notice).ToHaveTextAsync("Kernel restarted");

        await page.Clock.RunForAsync(1_100);
        await Expect(notice).ToBeEmptyAsync();
    }

    [Fact]
    public async Task AnError_StandsInABannerOfItsOwn_WhichANoticeLeaves_ASaveClears_AndADismissTakesAway()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference(FailingLayout.Part, FailingLayout.Id), DefaultKernelId = "csharp" };

        notebook.Cells.Add(Showing("1 + 1", "2"));
        await SaveAsync("failing.verso", notebook);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "failing.verso");
        var error = page.Locator("#error");

        await Expect(error.Locator(".text")).ToHaveTextAsync($"The layout could not be drawn: {FailingLayout.Why}");
        await Expect(page.Locator("#status")).ToBeEmptyAsync();

        // A notice leaves it where it is.
        page.Dialog += async (_, dialog) => await dialog.AcceptAsync();
        await page.Locator("#toolbar button[data-button='verso.action.restart-kernel']").ClickAsync();
        await Expect(page.Locator("#notice")).ToHaveTextAsync("Kernel restarted", new() { Timeout = 30_000 });
        await Expect(error).ToBeVisibleAsync();

        // A save clears it as the save begins.
        await page.Locator("#save").ClickAsync();
        await Expect(error).ToBeHiddenAsync();

        // Dismissed, it goes.
        var again = await OpenAsync(served, "failing.verso");

        await Expect(again.Locator("#error .text")).ToHaveTextAsync($"The layout could not be drawn: {FailingLayout.Why}");
        await again.Locator("#error button.dismiss").ClickAsync();
        await Expect(again.Locator("#error")).ToBeHiddenAsync();
    }
}

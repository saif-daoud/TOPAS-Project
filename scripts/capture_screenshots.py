from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


BASE_URL = "http://127.0.0.1:5173"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def capture(page: Page, output_dir: Path, name: str, *, full_page: bool = True) -> None:
    page.screenshot(path=output_dir / name, full_page=full_page)


def main() -> None:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots-updated").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME_PATH, headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page = context.new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(BASE_URL, wait_until="networkidle")
        page.evaluate("localStorage.clear()")
        page.reload(wait_until="networkidle")

        capture(page, output_dir, "01-access.png")

        page.locator("#gate-email").fill("screenshots.therapy@qcri.org")
        page.locator("#gate-code").fill("topas-preview")
        page.get_by_role("button", name="Enter workspace").click()
        page.get_by_role("heading", name="Overview").wait_for()
        page.wait_for_timeout(4500)
        capture(page, output_dir, "02-dashboard.png")

        page.get_by_role("button", name="New project").click()
        page.get_by_role("heading", name="What are we building?").wait_for()
        page.wait_for_timeout(600)
        capture(page, output_dir, "03-new-project.png")
        page.get_by_role("button", name="Close").click()

        page.locator(".project-card").filter(has_text="Showcase").click()
        page.locator(".project-page").wait_for()
        capture(page, output_dir, "04-extraction-overview.png")

        page.locator(".project-tabs button").nth(1).click()
        page.locator(".summary-content").wait_for()
        page.wait_for_timeout(1500)
        capture(page, output_dir, "05-summaries.png")

        page.locator(".project-tabs button").nth(2).click()
        page.locator(".confidence").first.wait_for()
        capture(page, output_dir, "06-components-confidence.png")

        page.locator(".component-nav button").filter(has_text="Knowledge graph").click()
        page.locator(".kg-canvas").wait_for()
        capture(page, output_dir, "07-knowledge-graph.png")

        page.get_by_role("button", name="Open graph workspace").first.click()
        page.locator(".graph-editor-window").wait_for()
        page.wait_for_timeout(500)
        capture(page, output_dir, "08-knowledge-graph-editor.png", full_page=False)
        page.get_by_role("button", name="Close graph workspace").click()

        page.locator(".pipeline-header button").nth(1).click()
        page.locator(".refinement-main").wait_for()
        page.wait_for_timeout(500)
        capture(page, output_dir, "09-refinement-overview.png")

        page.locator(".refinement-main .project-tabs button").nth(1).click()
        page.locator(".refinement-main .component-content").wait_for()
        page.locator(".refinement-main .button--edit").click()
        page.locator(".component-editor-window").wait_for()
        page.wait_for_timeout(500)
        capture(page, output_dir, "10-component-editor.png", full_page=False)
        page.get_by_role("button", name="Save changes").click()
        page.locator(".component-editor-window").wait_for(state="detached")

        page.locator(".refinement-main .project-tabs button").nth(2).click()
        page.locator(".refinement-report").wait_for()
        capture(page, output_dir, "11-refinement-report.png")

        mobile = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile_page = mobile.new_page()
        mobile_page.on("pageerror", lambda error: page_errors.append(str(error)))
        mobile_page.goto(BASE_URL, wait_until="networkidle")
        mobile_page.evaluate("localStorage.clear()")
        mobile_page.reload(wait_until="networkidle")
        capture(mobile_page, output_dir, "12-mobile-access.png")

        mobile.close()
        context.close()
        browser.close()

    if page_errors:
        raise RuntimeError("Browser page errors:\n" + "\n".join(page_errors))

    print(f"Captured screenshots in {output_dir}")


if __name__ == "__main__":
    main()

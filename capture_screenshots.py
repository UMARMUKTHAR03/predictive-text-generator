import asyncio
import os
import subprocess
import time
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "docs", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

async def capture_all_screenshots():
    print("Starting backend and frontend dev servers...")
    
    # 1. Start FastAPI server
    backend_proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "backend.app.main:app", "--port", "8000"],
        cwd=r"C:\Users\Personal\.gemini\antigravity\scratch\predictive-text-generator"
    )

    # 2. Start Vite preview/dev server
    frontend_proc = subprocess.Popen(
        ["cmd.exe", "/c", "npm run dev -- --port 5173"],
        cwd=r"C:\Users\Personal\.gemini\antigravity\scratch\predictive-text-generator\frontend"
    )

    time.sleep(4)  # Wait for servers to spin up

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1440, "height": 900})
            page = await context.new_page()

            base_url = "http://localhost:5173"
            print(f"Navigating to {base_url}...")
            await page.goto(base_url)
            await page.wait_for_timeout(2500)

            # 1. Dashboard
            print("Capturing 1. dashboard.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "dashboard.png"), full_page=False)

            # 2. Writing Studio - General Mode
            print("Navigating to Writing Studio...")
            await page.click("text=Writing Studio")
            await page.wait_for_timeout(2000)

            # Click General English mode
            await page.click("text=General English (DistilGPT-2)")
            await page.wait_for_timeout(1000)

            # Type General prompt
            textarea = page.locator("textarea")
            await textarea.fill("The future of artificial intelligence")
            await page.wait_for_timeout(2000)  # wait for debounced inference

            print("Capturing 2. writing-studio-general.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "writing-studio-general.png"), full_page=False)

            # 3. General Text Generation
            print("Clicking Generate Continuation...")
            await page.click("text=Generate Continuation")
            await page.wait_for_timeout(3500)
            print("Capturing 3. general-text-generation.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "general-text-generation.png"), full_page=False)

            # 4. Writing Studio - Technical Mode
            print("Switching to Technical Writing (BiLSTM)...")
            await page.click("text=Technical Writing (BiLSTM)")
            await page.wait_for_timeout(1000)
            await textarea.fill("Deep learning architectures for")
            await page.wait_for_timeout(2000)
            print("Capturing 4. writing-studio-technical.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "writing-studio-technical.png"), full_page=False)

            # 5. Technical Text Generation
            print("Generating technical continuation...")
            await page.click("text=Generate Continuation")
            await page.wait_for_timeout(3500)
            print("Capturing 5. technical-text-generation.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "technical-text-generation.png"), full_page=False)

            # 6. Model Intelligence
            print("Navigating to Model Intelligence...")
            await page.click("text=Model Intelligence")
            await page.wait_for_timeout(1500)
            print("Capturing 6. model-intelligence.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "model-intelligence.png"), full_page=False)

            # 7. Live Analytics
            print("Navigating to Live Analytics...")
            await page.click("text=Live Analytics")
            await page.wait_for_timeout(1500)
            print("Capturing 7. live-analytics.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "live-analytics.png"), full_page=False)

            # 8. Settings
            print("Navigating to Settings...")
            await page.click("text=Settings")
            await page.wait_for_timeout(1500)
            print("Capturing 8. settings.png...")
            await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "settings.png"), full_page=False)

            # 9. Mobile View
            print("Capturing 9. mobile-view.png...")
            mobile_context = await browser.new_context(viewport={"width": 390, "height": 844})
            mobile_page = await mobile_context.new_page()
            await mobile_page.goto(f"{base_url}")
            await mobile_page.wait_for_timeout(2000)
            await mobile_page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "mobile-view.png"), full_page=False)

            await browser.close()
            print("All 9 screenshots captured successfully!")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    asyncio.run(capture_all_screenshots())


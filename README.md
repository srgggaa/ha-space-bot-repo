# Space Bot Home Assistant Add-ons

This repository contains the **Space Image Discord Bot** add-on for Home Assistant OS / Supervisor.

## Installation

1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**.
2. Click the **⋮** menu (top right) → **Repositories**.
3. Paste this repository's URL:
   `https://github.com/YOUR_USERNAME/YOUR_REPO`
4. Click **Add**, then close the dialog.
5. The store will refresh and show a new section, "Space Bot Add-ons" — find
   **Space Image Discord Bot** inside it and click **Install**.
6. Go to the **Configuration** tab of the add-on and set:
   - `discord_bot_token` — your Discord bot's token (required)
   - `nasa_api_key` — your api.nasa.gov key (optional, defaults to `DEMO_KEY`)
7. On the **Info** tab, enable **Start on boot** and **Watchdog**, then click **Start**.

Logs are visible in the add-on's **Log** tab. Run `!setup_space` once in your
Discord server to trigger the first channel setup pass (it will otherwise
wait for the first hourly auto-fetch).

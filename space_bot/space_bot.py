"""
Space Image Discord Bot
========================
Pulls real space photography (NASA image library, ESA/SpaceX Flickr feeds,
ISRO/Roscosmos via Wikimedia Commons, and Mars rover feeds) into per-topic
Discord channels, filtering out:
  1. Anything that metadata suggests shows people, events, portraits, etc.
  2. Illustrations / artist concepts / renders / diagrams (not real photos).
  3. Anything that actually contains a detected human face or body, checked
     by decoding the image and running it through OpenCV cascade detectors
     (a real visual check, not just a keyword guess).

JAXA is not included: it has no public photo API, no official Flickr, and
barely any openly-licensed photography on Wikimedia Commons either (its
archive is a manual, per-request licensing portal) - see the comment above
its TARGETS entry for details.

Setup
-----
1. pip install -r requirements.txt
2. Set environment variables:
     DISCORD_BOT_TOKEN   - your bot's token (required)
     NASA_API_KEY        - your api.nasa.gov key (optional, defaults to
                            the shared DEMO_KEY, which is heavily rate
                            limited - get a free key at api.nasa.gov)
3. python space_bot.py

Note: ESA/SpaceX (Flickr's public feed) and ISRO/Roscosmos (Wikimedia
Commons) all use free, keyless public endpoints - no extra credentials
needed for any of them.
"""

import os
import io
import json
import re
import asyncio
import logging
import tempfile
import urllib.parse
from datetime import datetime

import aiohttp
import discord
from discord.ext import commands, tasks

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

DEFAULT_CATEGORY_NAME = "NASA"  # kept as the default/fallback so existing
                                 # NASA channels from before this update stay
                                 # in the same category and aren't orphaned
LOG_FILE = "bot_downloaded_log.json"
HUMAN_LOG_FILE = "bot_human_detected_log.json"
FETCH_INTERVAL_MINUTES = 60
MAX_PAGES = 5
PAGE_SIZE = 100
DOWNLOAD_CONCURRENCY = 4          # simultaneous image downloads for CV checks
MAX_IMAGE_BYTES = 25 * 1024 * 1024  # skip absurdly large files
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF_SECONDS = 2

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("space_bot")

TARGETS = {
    "apollo": {
        "type": "nasa_api", "query": "Apollo program", "name": "🚀-apollo-mission",
        "description": "Real photography from the Apollo Moon missions (1961-1972), NASA Image Library.",
    },
    "artemis": {
        "type": "nasa_api", "query": "Artemis program", "name": "🚀-artemis-mission",
        "description": "Real photography from NASA's Artemis Moon program, NASA Image Library.",
    },
    "voyager": {
        "type": "nasa_api", "query": "Voyager spacecraft", "name": "🛰️-voyager-mission",
        "description": "Real imagery from the Voyager 1 & 2 probes, NASA Image Library.",
    },
    "cassini": {
        "type": "nasa_api", "query": "Cassini Saturn", "name": "🛰️-cassini-mission",
        "description": "Real imagery from the Cassini-Huygens mission to Saturn, NASA Image Library.",
    },
    "hubble": {
        "type": "nasa_api", "query": "Hubble Space Telescope", "name": "🔭-hubble",
        "description": "Real deep-space photography captured by the Hubble Space Telescope.",
    },
    "james-webb": {
        "type": "nasa_api", "query": "JWST Webb Space Telescope", "name": "🔭-james-webb",
        "description": "Real deep-space photography captured by the James Webb Space Telescope.",
    },
    "roman": {
        "type": "nasa_api", "query": "Nancy Grace Roman Space Telescope", "name": "🔭-roman-telescope",
        "description": "Imagery and mission photos related to the Nancy Grace Roman Space Telescope.",
    },
    "iss": {
        "type": "nasa_api", "query": "International Space Station", "name": "🛰️-iss",
        "description": "Real photography of and from the International Space Station, NASA Image Library.",
    },
    "earth": {
        "type": "nasa_api", "query": "Earth from space", "name": "🌍-earth-from-space",
        "description": "Real photographs of Earth taken from orbit or deep space.",
    },
    "nebula": {
        "type": "nasa_api", "query": "nebula", "name": "🌌-nebulae",
        "description": "Real telescope photography of nebulae.",
    },
    "galaxy": {
        "type": "nasa_api", "query": "galaxy", "name": "🌌-galaxies",
        "description": "Real telescope photography of galaxies.",
    },
    # NOTE: These used to hit api.nasa.gov/mars-photos (type: "mars_api"),
    # which proxies a community-run project (mars-photos.herokuapp.com)
    # that's been unmaintained since Heroku killed free dynos in 2022. That
    # endpoint is now returning 404s outright, so these are pointed at the
    # NASA Image & Video Library search instead (same working mechanism as
    # every other channel above). The old "mars_api" code path (get_mars_photos
    # / fetch_mars_api) is left in place below in case NASA ever restores
    # a working rover-photos endpoint - switch "type" back to "mars_api"
    # if so.
    "mars-curiosity": {
        "type": "nasa_api", "query": "Curiosity rover Mars surface Gale Crater",
        "name": "🔴-mars-curiosity",
        "description": "Real surface photography from NASA's Curiosity rover (Gale Crater, Mars).",
    },
    "mars-perseverance": {
        "type": "nasa_api", "query": "Perseverance rover Mars surface Jezero Crater",
        "name": "🔴-mars-perseverance",
        "description": "Real surface photography from NASA's Perseverance rover (Jezero Crater, Mars).",
    },

    # ----------------------------------------------------------------
    # ESA (European Space Agency) - real photography pulled from ESA's
    # official Flickr photostream (flickr.com/photos/europeanspaceagency)
    # via Flickr's free public feed (see fetch_flickr for why: Flickr's
    # paid-only REST search API is deliberately not used). The public feed
    # is a single per-account "latest photos" stream with no full-text
    # search, so unlike NASA this is one general channel rather than
    # several mission-specific ones.
    # ----------------------------------------------------------------
    "esa": {
        "type": "flickr", "category": "ESA", "flickr_user": "europeanspaceagency",
        "name": "🛰️-esa",
        "description": "Real space photography from ESA's official Flickr photostream.",
    },

    # ----------------------------------------------------------------
    # SpaceX - real mission photography pulled from SpaceX's official
    # Flickr account (flickr.com/photos/spacex), which SpaceX uses to
    # release its own photos into the public domain, via the same free
    # public feed mechanism as ESA above.
    # ----------------------------------------------------------------
    "spacex": {
        "type": "flickr", "category": "SpaceX", "flickr_user": "spacex",
        "name": "🚀-spacex",
        "description": "Real mission photography from SpaceX's official Flickr photostream.",
    },

    # ----------------------------------------------------------------
    # ISRO (India) and Roscosmos (Russia) - neither publishes a public
    # photo API or its own open-license photo feed, so these pull from
    # Wikimedia Commons instead: a free, keyless MediaWiki API that hosts
    # openly-licensed real photography of both agencies' missions,
    # organized into browsable categories. See fetch_commons_category.
    # ----------------------------------------------------------------
    "isro": {
        "type": "commons", "category": "ISRO", "commons_category": "Images from ISRO",
        "name": "🛰️-isro",
        "description": "Real space photography from ISRO (Indian Space Research Organisation), via Wikimedia Commons.",
    },
    "roscosmos": {
        "type": "commons", "category": "Roscosmos", "commons_category": "Baikonur Cosmodrome",
        "name": "🚀-roscosmos",
        "description": "Real launch photography of Roscosmos/Russian missions from the Baikonur Cosmodrome, via Wikimedia Commons.",
    },

    # ----------------------------------------------------------------
    # JAXA (Japan) was requested but is still left unimplemented on
    # purpose: it has no public photo API, no official Flickr gallery,
    # and (unlike ISRO/Roscosmos) barely any real mission photography on
    # Wikimedia Commons either - its "JAXA Digital Archives"
    # (jda.jaxa.jp) is a manual, per-request licensing portal that keeps
    # its imagery out of the openly-licensed pool the other sources here
    # draw from. If that ever changes, add a "flickr" or "commons"-style
    # entry here the same way the others were added.
    # ----------------------------------------------------------------
}

# Metadata-level exclusions. Two separate lists so we can log *why* an item
# was skipped.
HUMAN_TERMS = {
    "press conference", "briefing", "portrait", "selfie of the crew", "group photo",
    "standing", "sitting", "posing", "handshake", "award", "ceremony",
    "auditorium", "headquarters", "administrator", "speech", "podium",
    "panelist", "qa session", "q&a", "news conference", "tour", "astronaut candidate",
    "crew member", "crew photo", "training session", "classroom", "interview",
    "visitor", "spectator", "reporter", "photographer poses", "employee", "staff photo",
    "engineer poses", "technician poses", "scientist poses", "president", "senator",
    "congress", "dignitary", "media day", "ribbon cutting", "graduation", "commencement",
    "suit up", "walkout", "family", "children visit", "student", "classroom visit",
}

RENDER_TERMS = {
    "illustration", "artist concept", "artist's concept", "artists concept",
    "rendering", "cgi", "concept art", "conceptual image", "animation still",
    "graphic", "infographic", "diagram", "chart", "poster", "logo", "patch design",
    "artist rendition", "computer generated", "simulated image", "not an actual photograph",
}

EXCLUDE_TERMS = HUMAN_TERMS | RENDER_TERMS

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        downloaded_ids = set(json.load(f))
else:
    downloaded_ids = set()

if os.path.exists(HUMAN_LOG_FILE):
    with open(HUMAN_LOG_FILE, "r") as f:
        try:
            human_detections = json.load(f)
        except json.JSONDecodeError:
            human_detections = []
else:
    human_detections = []

# Migration: earlier versions of this bot only added an id to
# `downloaded_ids` once an image was actually *posted*, never when it was
# skipped (metadata match or visual human detection). That meant every
# skipped image got re-downloaded and re-run through OpenCV on every single
# fetch cycle / restart, forever. Backfill from the existing human-detection
# log so upgrading doesn't leave those already-caught images in the "will
# be rescanned forever" state.
downloaded_ids.update(entry["id"] for entry in human_detections if "id" in entry)

_log_lock = asyncio.Lock()
_human_log_lock = asyncio.Lock()


async def _atomic_json_write(path: str, data, lock: asyncio.Lock):
    """Atomically persist JSON (write-to-temp-file then rename) so a crash
    mid-save can't corrupt the file."""
    async with lock:
        dir_name = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


async def save_log():
    """Atomically persist the downloaded-id log."""
    await _atomic_json_write(LOG_FILE, list(downloaded_ids), _log_lock)


async def log_human_detection(source: str, item_id: str, title: str, image_url: str, channel_name: str):
    """Records an image that was skipped because a human face/body was
    visually detected in it, so you can audit what the CV filter is
    catching (and spot-check for false positives) without digging through
    console logs."""
    human_detections.append({
        "id": item_id,
        "source": source,
        "title": title,
        "image_url": image_url,
        "target_channel": channel_name,
        "detected_at": datetime.utcnow().isoformat() + "Z",
    })
    await _atomic_json_write(HUMAN_LOG_FILE, human_detections, _human_log_lock)

    # Mark as processed so this exact image is never re-downloaded and
    # re-run through OpenCV on a future fetch cycle or restart. Without
    # this, human-detected images were skipped from posting but never
    # remembered, so they got rescanned every single run forever.
    downloaded_ids.add(item_id)
    await save_log()


def clean_url(url_str: str) -> str:
    """Ensures URLs use https and escapes special characters or spaces."""
    if not url_str:
        return ""
    if url_str.startswith("http://"):
        url_str = "https://" + url_str[7:]
    return urllib.parse.quote(url_str, safe=":/%?&=#+")


def metadata_looks_human_or_fake(item_data: dict) -> str | None:
    """Returns the matched exclusion term if the metadata suggests a human
    subject or a non-photographic render, else None."""
    title = (item_data.get("title") or "").lower()
    description = (item_data.get("description") or "").lower()
    keywords = [k.lower() for k in item_data.get("keywords", [])]
    blob = f"{title} {description} {' '.join(keywords)}"
    for term in EXCLUDE_TERMS:
        if term in blob:
            return term
    return None


# --------------------------------------------------------------------------
# Real visual human-detection (not just keyword guessing)
# --------------------------------------------------------------------------

_face_cascade = None
_body_cascade = None

if CV2_AVAILABLE:
    try:
        # Alpine's py3-opencv package (used in the Docker build) doesn't
        # ship the cv2.data submodule that pip's opencv-python bundles, so
        # cv2.data.haarcascades doesn't exist there. Prefer a cascade dir
        # baked into the image (see Dockerfile), falling back to cv2.data
        # for environments (e.g. pip install) where it *is* available.
        _cascade_dir = os.getenv("CASCADE_DIR")
        if not _cascade_dir:
            _cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if not _cascade_dir:
            raise RuntimeError(
                "No cascade directory available (set CASCADE_DIR or install "
                "opencv-python, not just opencv's C++ bindings)"
            )
        if not _cascade_dir.endswith("/"):
            _cascade_dir += "/"

        _face_cascade = cv2.CascadeClassifier(
            _cascade_dir + "haarcascade_frontalface_default.xml"
        )
        _body_cascade = cv2.CascadeClassifier(
            _cascade_dir + "haarcascade_upperbody.xml"
        )
        if _face_cascade.empty() or _body_cascade.empty():
            raise RuntimeError(f"Cascade XML files failed to load from {_cascade_dir}")
    except Exception as e:
        log.warning("Could not load OpenCV cascades, disabling visual human check: %s", e)
        CV2_AVAILABLE = False


def _detect_human_sync(image_bytes: bytes) -> bool:
    """CPU-bound face/body detection. Run via asyncio.to_thread.

    Decodes directly at reduced resolution (IMREAD_REDUCED_COLOR_4) instead
    of decoding at full native resolution and resizing afterwards. Some NASA
    imagery (Hubble/JWST mosaics, stitched panoramas, etc.) is tens of
    megapixels even when the compressed file is well under MAX_IMAGE_BYTES;
    a full-resolution decode of one such image can spike memory by hundreds
    of MB to 1GB+, which is enough to get the whole add-on OOM-killed on a
    memory-constrained host. Detection accuracy doesn't need full res
    anyway (the old code immediately downscaled to max_dim=1000 after
    decoding), so decode small from the start.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        # Reduced-resolution decode isn't supported for every codec/file;
        # fall back to a normal decode for those cases.
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False

    # Downscale further if still large; detection accuracy doesn't need
    # full resolution.
    h, w = img.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
    )
    if len(faces) > 0:
        return True

    bodies = _body_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
    )
    return len(bodies) > 0


async def image_contains_human(session: aiohttp.ClientSession, image_url: str) -> bool:
    """Downloads the image and checks it for faces/bodies. Fails open
    (treats as 'no human detected') if the check can't run, since the
    metadata filter already ran first."""
    if not CV2_AVAILABLE:
        return False
    try:
        log.debug("Downloading for human-check: %s", image_url)
        async with session.get(image_url) as resp:
            if resp.status != 200:
                return False
            content_length = resp.content_length
            if content_length and content_length > MAX_IMAGE_BYTES:
                log.info(
                    "Skipping human-check download for %s: declared size %s bytes exceeds cap",
                    image_url, content_length,
                )
                return False

            # Stream the body and enforce MAX_IMAGE_BYTES as data actually
            # arrives, instead of calling resp.read() and checking the size
            # afterwards. Without a Content-Length header (chunked
            # responses, which some CDNs use), resp.read() would buffer an
            # unbounded amount of data into memory before the size check
            # ever ran - which is exactly the kind of thing that gets a
            # memory-constrained container OOM-killed.
            chunks = []
            total = 0
            async for chunk in resp.content.iter_chunked(65536):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    log.info(
                        "Aborting human-check download for %s: exceeded %s bytes cap mid-stream",
                        image_url, MAX_IMAGE_BYTES,
                    )
                    return False
                chunks.append(chunk)
            data = b"".join(chunks)
        return await asyncio.to_thread(_detect_human_sync, data)
    except Exception as e:
        log.warning("Human-detection check failed for %s: %s", image_url, e)
        return False


# --------------------------------------------------------------------------
# HTTP helper with retry/backoff
# --------------------------------------------------------------------------

async def get_json_with_retries(session: aiohttp.ClientSession, url: str):
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    wait = HTTP_BACKOFF_SECONDS * attempt
                    log.warning("Rate limited on %s, waiting %ss", url, wait)
                    await asyncio.sleep(wait)
                    continue
                log.warning("Request to %s failed with status %s", url, resp.status)
                return None
        except aiohttp.ClientError as e:
            log.warning("Request error on %s (attempt %s): %s", url, attempt, e)
            await asyncio.sleep(HTTP_BACKOFF_SECONDS * attempt)
    return None


async def get_text_with_retries(session: aiohttp.ClientSession, url: str) -> str | None:
    """Like get_json_with_retries, but for endpoints (like Flickr's public
    feeds) that don't reliably send a JSON content-type, so resp.json()
    can't be used directly."""
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
                if resp.status == 429:
                    wait = HTTP_BACKOFF_SECONDS * attempt
                    log.warning("Rate limited on %s, waiting %ss", url, wait)
                    await asyncio.sleep(wait)
                    continue
                log.warning("Request to %s failed with status %s", url, resp.status)
                return None
        except aiohttp.ClientError as e:
            log.warning("Request error on %s (attempt %s): %s", url, attempt, e)
            await asyncio.sleep(HTTP_BACKOFF_SECONDS * attempt)
    return None


# --------------------------------------------------------------------------
# Discord role & channel management
# --------------------------------------------------------------------------

BOT_ROLE_NAME = "🛰️ Space Bot"

# Cache of guild_id -> discord.Role so we don't re-fetch/create every call.
_bot_role_cache: dict[int, discord.Role] = {}


async def get_or_create_bot_role(guild: discord.Guild):
    """Ensures a dedicated role for the bot exists and is assigned to it.
    This role is what gets permission to post in the space channels -
    everyone else is read-only there."""
    cached = _bot_role_cache.get(guild.id)
    if cached and cached in guild.roles:
        return cached

    role = discord.utils.get(guild.roles, name=BOT_ROLE_NAME)
    if not role:
        try:
            role = await guild.create_role(
                name=BOT_ROLE_NAME,
                color=discord.Color.blue(),
                reason="Dedicated role for the space image bot",
                mentionable=False,
            )
        except discord.Forbidden:
            log.error("Missing permission to create role '%s' in %s", BOT_ROLE_NAME, guild.name)
            return None
        except Exception as e:
            log.error("Failed to create role %s: %s", BOT_ROLE_NAME, e)
            return None

    me = guild.me
    if me and role not in me.roles:
        try:
            await me.add_roles(role, reason="Assigning dedicated space bot role")
        except discord.Forbidden:
            log.error(
                "Missing permission to assign role '%s' to myself in %s "
                "(check the bot's role is high enough in the role list)",
                BOT_ROLE_NAME, guild.name,
            )
        except Exception as e:
            log.error("Failed to assign role %s to bot: %s", BOT_ROLE_NAME, e)

    _bot_role_cache[guild.id] = role
    return role


def _locked_overwrites(guild: discord.Guild, bot_role: discord.Role):
    """Permission overwrites that let everyone view/read the channel but
    only the bot's role can post."""
    return {
        guild.default_role: discord.PermissionOverwrite(
            view_channel=True,
            read_message_history=True,
            send_messages=False,
            add_reactions=True,
            create_public_threads=False,
            create_private_threads=False,
        ),
        bot_role: discord.PermissionOverwrite(
            view_channel=True,
            send_messages=True,
            embed_links=True,
            attach_files=True,
            read_message_history=True,
            manage_messages=True,
        ),
    }


async def get_or_create_channel(
    guild: discord.Guild,
    channel_name: str,
    bot_role: discord.Role,
    topic: str | None = None,
    category_name: str = DEFAULT_CATEGORY_NAME,
):
    overwrites = _locked_overwrites(guild, bot_role) if bot_role else None

    category = discord.utils.get(guild.categories, name=category_name)
    if not category:
        try:
            category = await guild.create_category(name=category_name, overwrites=overwrites)
        except discord.Forbidden:
            log.error("Missing permission to create category '%s' in %s", category_name, guild.name)
            return None
        except Exception as e:
            log.error("Failed to create category %s: %s", category_name, e)
            return None
    elif overwrites and category.overwrites != overwrites:
        try:
            await category.edit(overwrites=overwrites, reason="Lock category to bot-only posting")
        except discord.Forbidden:
            log.warning("Missing permission to update permissions on category '%s'", category_name)
        except Exception as e:
            log.warning("Failed to update category permissions: %s", e)

    channel = discord.utils.get(category.text_channels, name=channel_name)
    if not channel:
        try:
            channel = await guild.create_text_channel(
                name=channel_name, category=category, overwrites=overwrites, topic=topic
            )
        except discord.Forbidden:
            log.error("Missing permission to create channel '%s' in %s", channel_name, guild.name)
            return None
        except Exception as e:
            log.error("Failed to create channel %s: %s", channel_name, e)
            return None
    else:
        edit_kwargs = {}
        if overwrites and channel.overwrites != overwrites:
            # Channel already existed (e.g. from before this feature) - lock it down.
            edit_kwargs["overwrites"] = overwrites
        if topic is not None and channel.topic != topic:
            edit_kwargs["topic"] = topic
        if edit_kwargs:
            try:
                await channel.edit(reason="Sync space-bot channel settings", **edit_kwargs)
            except discord.Forbidden:
                log.warning("Missing permission to update channel '%s'", channel_name)
            except Exception as e:
                log.warning("Failed to update channel %s: %s", channel_name, e)

    return channel


# --------------------------------------------------------------------------
# NASA image library
# --------------------------------------------------------------------------

def parse_date(item_obj: dict) -> datetime:
    date_str = item_obj["data"][0].get("date_created", "1900-01-01T00:00:00Z")
    try:
        return datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        pass
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except ValueError:
        return datetime.min


async def fetch_nasa_api(session: aiohttp.ClientSession, guild: discord.Guild, target_info: dict, bot_role: discord.Role):
    channel = await get_or_create_channel(guild, target_info["name"], bot_role, topic=target_info.get("description"), category_name=target_info.get("category", DEFAULT_CATEGORY_NAME))
    if not channel:
        return

    query = urllib.parse.quote(target_info["query"])
    all_items = []

    for page in range(1, MAX_PAGES + 1):
        url = (
            f"https://images-api.nasa.gov/search?q={query}&media_type=image"
            f"&year_start=1950&page={page}&page_size={PAGE_SIZE}"
        )
        data = await get_json_with_retries(session, url)
        if not data:
            break
        items = data.get("collection", {}).get("items", [])
        if not items:
            break
        all_items.extend(items)

    items_sorted = sorted(all_items, key=parse_date)
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async def handle_item(item):
        item_data = item["data"][0]
        nasa_id = item_data.get("nasa_id")
        if not nasa_id or nasa_id in downloaded_ids:
            return

        skip_reason = metadata_looks_human_or_fake(item_data)
        if skip_reason:
            downloaded_ids.add(nasa_id)
            await save_log()
            return

        disp_title = item_data.get("title", "N/A")
        date_created = item_data.get("date_created", "N/A")
        center = item_data.get("center", "NASA")
        nasa_web_url = clean_url(f"https://images.nasa.gov/details-{nasa_id}")

        asset_data = await get_json_with_retries(session, f"https://images-api.nasa.gov/asset/{nasa_id}")
        if not asset_data:
            return
        asset_items = asset_data.get("collection", {}).get("items", [])
        img_urls = [a["href"] for a in asset_items if a.get("href", "").lower().endswith((".jpg", ".jpeg", ".png"))]
        if not img_urls:
            return

        image_url = clean_url(img_urls[0])

        log.debug("Checking %s (%s) -> %s", nasa_id, disp_title[:40], image_url)
        async with sem:
            if await image_contains_human(session, image_url):
                log.info("Skipping %s (%s): human detected in image", nasa_id, disp_title[:40])
                await log_human_detection(
                    source="nasa_image_library",
                    item_id=nasa_id,
                    title=disp_title,
                    image_url=image_url,
                    channel_name=target_info["name"],
                )
                return

        embed = discord.Embed(title=disp_title[:256], url=nasa_web_url, color=discord.Color.blue())
        embed.add_field(name="Publication Date", value=str(date_created)[:10], inline=True)
        embed.add_field(name="Mission / Center", value=str(center), inline=True)
        embed.add_field(name="NASA Source", value=f"[View Asset]({nasa_web_url})", inline=False)
        embed.set_image(url=image_url)
        embed.set_footer(text=f"NASA ID: {nasa_id}")

        try:
            await channel.send(embed=embed)
            log.info("Posted: %s... to %s", disp_title[:30], target_info["name"])
            downloaded_ids.add(nasa_id)
            await save_log()
        except discord.HTTPException as e:
            log.warning("Failed to post image (%s): %s", nasa_id, e)

    for item in items_sorted:
        await handle_item(item)


# --------------------------------------------------------------------------
# Flickr (used for ESA and SpaceX - neither has a public REST image API of
# its own, but both publish their real mission photography through Flickr)
#
# NOTE: Flickr's REST API (flickr.photos.search etc.) now requires a paid
# Flickr Pro subscription just to issue a new API key, so this deliberately
# does NOT use it. Instead it uses Flickr's public feed endpoint
# (services/feeds/photos_public.gne), which is still free/keyless - it's
# what Flickr itself designed for exactly this kind of external polling.
# The tradeoff: the public feed only returns each account's ~20 most
# recent public photos and has no free-text search, only a per-user feed
# (and an ALL/ANY tag filter, which isn't reliable here since we can't
# assume either agency tags photos consistently). So instead of the
# mission-specific sub-channels NASA gets, ESA and SpaceX each get one
# channel of their latest real photos. Since the bot polls hourly and
# dedupes by photo ID, it still catches everything each account posts
# going forward - it just can't backfill their full multi-thousand-photo
# archives the way the paid search API could have.
# --------------------------------------------------------------------------

# Numeric Flickr NSIDs for each account, looked up once by hand (Flickr's
# username->NSID lookup method also now requires the paid API tier). If an
# account ever migrates to a new NSID, update it here.
FLICKR_NSIDS = {
    "europeanspaceagency": "37472264@N04",
    "spacex": "130608600@N05",
}

_FLICKR_FEED_RE = re.compile(r"^\s*jsonFlickrFeed\((.*)\)\s*;?\s*$", re.DOTALL)


def _parse_flickr_feed(raw_text: str) -> dict | None:
    """The public feed's format=json response is JSONP - a jsonFlickrFeed(...)
    call wrapping the actual JSON object - not plain JSON, so it can't be
    read with resp.json(). Strip the wrapper and parse what's left."""
    match = _FLICKR_FEED_RE.match(raw_text)
    body = match.group(1) if match else raw_text
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        log.warning("Could not parse Flickr feed response: %s", e)
        return None


async def fetch_flickr(session: aiohttp.ClientSession, guild: discord.Guild, target_info: dict, bot_role: discord.Role):
    channel = await get_or_create_channel(
        guild, target_info["name"], bot_role, topic=target_info.get("description"),
        category_name=target_info.get("category", DEFAULT_CATEGORY_NAME),
    )
    if not channel:
        return

    username = target_info["flickr_user"]
    nsid = FLICKR_NSIDS.get(username)
    if not nsid:
        log.error("No known Flickr NSID for username '%s' - add one to FLICKR_NSIDS", username)
        return

    url = (
        "https://www.flickr.com/services/feeds/photos_public.gne"
        f"?id={urllib.parse.quote(nsid)}&format=json&lang=en-us"
    )
    raw_text = await get_text_with_retries(session, url)
    if not raw_text:
        return
    feed = _parse_flickr_feed(raw_text)
    if not feed:
        return

    items = feed.get("items", [])
    # Feed order is newest-first; post oldest-of-the-batch first so a
    # channel reads chronologically, same as the NASA channels.
    items = list(reversed(items))

    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async def handle_item(item: dict):
        link = item.get("link", "")
        photo_id_match = re.search(r"/(\d+)/?$", link.rstrip("/"))
        if not photo_id_match:
            return
        flickr_id = f"flickr_{photo_id_match.group(1)}"
        if flickr_id in downloaded_ids:
            return

        title = item.get("title") or "Untitled"
        # The feed's "description" field is an HTML snippet (thumbnail +
        # caption), not plain text - strip tags before running it through
        # the same keyword filter used for NASA metadata.
        description_html = item.get("description") or ""
        description = re.sub(r"<[^>]+>", " ", description_html)
        tags = (item.get("tags") or "").split()

        skip_reason = metadata_looks_human_or_fake(
            {"title": title, "description": description, "keywords": tags}
        )
        if skip_reason:
            downloaded_ids.add(flickr_id)
            await save_log()
            return

        media = item.get("media", {})
        image_url = media.get("m", "")
        if not image_url:
            return
        # The feed's media.m URL points at a small "_m" thumbnail - swap in
        # the "_b" (large, ~1024px) size, which Flickr serves at the same
        # path with a different suffix, so embeds aren't postage-stamp sized.
        image_url = re.sub(r"_m(\.[a-zA-Z]+)$", r"_b\1", image_url)
        image_url = clean_url(image_url)

        log.debug("Checking %s (%s) -> %s", flickr_id, title[:40], image_url)
        async with sem:
            if await image_contains_human(session, image_url):
                log.info("Skipping %s (%s): human detected in image", flickr_id, title[:40])
                await log_human_detection(
                    source=f"flickr_{username}",
                    item_id=flickr_id,
                    title=title,
                    image_url=image_url,
                    channel_name=target_info["name"],
                )
                return

        embed = discord.Embed(title=title[:256], url=link, color=discord.Color.gold())
        date_taken = item.get("date_taken", "N/A")
        embed.add_field(name="Date Taken", value=str(date_taken)[:10], inline=True)
        embed.add_field(name="Source", value=f"[View on Flickr]({link})", inline=False)
        embed.set_image(url=image_url)
        embed.set_footer(text=f"Flickr ID: {photo_id_match.group(1)}")

        try:
            await channel.send(embed=embed)
            log.info("Posted: %s... to %s", title[:30], target_info["name"])
            downloaded_ids.add(flickr_id)
            await save_log()
        except discord.HTTPException as e:
            log.warning("Failed to post image (%s): %s", flickr_id, e)

    for item in items:
        await handle_item(item)


# --------------------------------------------------------------------------
# Wikimedia Commons (used for ISRO and Roscosmos - neither publishes a
# public photo API or an official open-license photo feed of its own, but
# Commons hosts openly-licensed real photography of their missions,
# organized into browsable categories, via a free, keyless MediaWiki API)
# --------------------------------------------------------------------------

async def fetch_commons_category(session: aiohttp.ClientSession, guild: discord.Guild, target_info: dict, bot_role: discord.Role):
    channel = await get_or_create_channel(
        guild, target_info["name"], bot_role, topic=target_info.get("description"),
        category_name=target_info.get("category", DEFAULT_CATEGORY_NAME),
    )
    if not channel:
        return

    commons_category = target_info["commons_category"]
    all_pages = []
    gcmcontinue = None
    for _ in range(MAX_PAGES):
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{commons_category}",
            "gcmtype": "file",
            "gcmlimit": "500",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": "1600",
            "format": "json",
        }
        if gcmcontinue:
            params["gcmcontinue"] = gcmcontinue
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
        data = await get_json_with_retries(session, url)
        if not data:
            break
        pages = data.get("query", {}).get("pages", {})
        all_pages.extend(pages.values())
        gcmcontinue = data.get("continue", {}).get("gcmcontinue")
        if not gcmcontinue:
            break

    # Sort oldest-first when a capture date is available, same as the other
    # sources; files with no known date sort first (empty string < any date).
    def _sort_key(page):
        info = (page.get("imageinfo") or [{}])[0]
        return info.get("extmetadata", {}).get("DateTimeOriginal", {}).get("value", "")

    all_pages.sort(key=_sort_key)
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async def handle_page(page: dict):
        page_id = page.get("pageid")
        title = page.get("title", "")
        if not page_id or not title:
            return
        commons_id = f"commons_{page_id}"
        if commons_id in downloaded_ids:
            return

        infos = page.get("imageinfo") or []
        if not infos:
            return
        info = infos[0]
        mime = info.get("mime", "")
        # Commons categories mix in videos, SVG diagrams/logos, and PDFs
        # alongside real photos - only real raster photography should post.
        if not mime.startswith("image/") or mime == "image/svg+xml":
            return

        extmeta = info.get("extmetadata", {})
        description_html = extmeta.get("ImageDescription", {}).get("value", "")
        description = re.sub(r"<[^>]+>", " ", description_html)
        categories_str = extmeta.get("Categories", {}).get("value", "")
        keywords = categories_str.split("|") if categories_str else []
        disp_title = title[len("File:"):] if title.startswith("File:") else title
        disp_title = re.sub(r"\.[a-zA-Z0-9]+$", "", disp_title).replace("_", " ")

        skip_reason = metadata_looks_human_or_fake(
            {"title": disp_title, "description": description, "keywords": keywords}
        )
        if skip_reason:
            downloaded_ids.add(commons_id)
            await save_log()
            return

        image_url = info.get("thumburl") or info.get("url")
        if not image_url:
            return
        image_url = clean_url(image_url)
        page_url = f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        artist_html = extmeta.get("Artist", {}).get("value", "")
        artist = re.sub(r"<[^>]+>", " ", artist_html).strip() or "Wikimedia Commons"
        license_name = extmeta.get("LicenseShortName", {}).get("value", "")

        log.debug("Checking %s (%s) -> %s", commons_id, disp_title[:40], image_url)
        async with sem:
            if await image_contains_human(session, image_url):
                log.info("Skipping %s (%s): human detected in image", commons_id, disp_title[:40])
                await log_human_detection(
                    source=f"commons_{commons_category}",
                    item_id=commons_id,
                    title=disp_title,
                    image_url=image_url,
                    channel_name=target_info["name"],
                )
                return

        embed = discord.Embed(title=disp_title[:256], url=page_url, color=discord.Color.dark_teal())
        embed.add_field(name="Credit", value=artist[:1024], inline=True)
        if license_name:
            embed.add_field(name="License", value=license_name, inline=True)
        embed.add_field(name="Source", value=f"[View on Wikimedia Commons]({page_url})", inline=False)
        embed.set_image(url=image_url)
        embed.set_footer(text=f"Commons Page ID: {page_id}")

        try:
            await channel.send(embed=embed)
            log.info("Posted: %s... to %s", disp_title[:30], target_info["name"])
            downloaded_ids.add(commons_id)
            await save_log()
        except discord.HTTPException as e:
            log.warning("Failed to post image (%s): %s", commons_id, e)

    for page in all_pages:
        await handle_page(page)


# --------------------------------------------------------------------------
# Mars rover photos
# --------------------------------------------------------------------------

async def get_mars_photos(session: aiohttp.ClientSession, rover: str) -> list[dict]:
    """Fetches photos for a rover, robust to the fact that /latest_photos
    only covers the single most recent sol and can legitimately come back
    empty on a sol with no downlinked imagery (this is not necessarily an
    error - the underlying mars-photos API is a community project that's
    now archived/unmaintained, so treat outages as possible too).

    Falls back to the mission manifest to find the actual most recent sol
    with photos, then queries that sol directly."""
    base = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}"

    latest_url = f"{base}/latest_photos?api_key={NASA_API_KEY}"
    data = await get_json_with_retries(session, latest_url)
    photos = (data or {}).get("latest_photos", [])
    if photos:
        return photos

    log.info("latest_photos for %s was empty, falling back to mission manifest", rover)
    manifest_url = f"{base}/manifests/{rover}?api_key={NASA_API_KEY}"
    manifest_data = await get_json_with_retries(session, manifest_url)
    sols_with_photos = [
        entry["sol"] for entry in (manifest_data or {}).get("photo_manifest", {}).get("photos", [])
        if entry.get("total_photos", 0) > 0
    ]
    if not sols_with_photos:
        return []

    max_sol = max(sols_with_photos)
    sol_url = f"{base}/photos?sol={max_sol}&api_key={NASA_API_KEY}"
    sol_data = await get_json_with_retries(session, sol_url)
    return (sol_data or {}).get("photos", [])


async def fetch_mars_api(session: aiohttp.ClientSession, guild: discord.Guild, target_info: dict, bot_role: discord.Role):
    channel = await get_or_create_channel(guild, target_info["name"], bot_role, topic=target_info.get("description"), category_name=target_info.get("category", DEFAULT_CATEGORY_NAME))
    if not channel:
        return

    rover = target_info["rover"]
    photos = await get_mars_photos(session, rover)
    if not photos:
        log.warning(
            "No Mars photos found for %s (checked latest_photos and manifest fallback). "
            "This can happen if the (community-maintained, now-archived) mars-photos API "
            "is having an outage - check https://api.nasa.gov status if this persists.",
            rover,
        )
        return
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async def handle_photo(photo):
        photo_id = f"mars_{rover}_{photo['id']}"
        if photo_id in downloaded_ids:
            return

        camera_name = photo.get("camera", {}).get("full_name", "Rover Camera")
        earth_date = photo.get("earth_date", "N/A")
        sol = photo.get("sol", "N/A")
        img_url = clean_url(photo.get("img_src", ""))
        if not img_url:
            return

        log.debug("Checking %s -> %s", photo_id, img_url)
        async with sem:
            if await image_contains_human(session, img_url):
                log.info("Skipping mars photo %s: human detected in image", photo_id)
                await log_human_detection(
                    source=f"mars_{rover}",
                    item_id=photo_id,
                    title=f"{rover.capitalize()} Sol {sol} ({camera_name})",
                    image_url=img_url,
                    channel_name=target_info["name"],
                )
                return

        embed = discord.Embed(
            title=f"{rover.capitalize()} Surface Capture - Sol {sol}",
            color=discord.Color.red(),
        )
        embed.add_field(name="Earth Date", value=str(earth_date), inline=True)
        embed.add_field(name="Martian Sol", value=str(sol), inline=True)
        embed.add_field(name="Camera Instrument", value=str(camera_name), inline=False)
        embed.set_image(url=img_url)
        embed.set_footer(text=f"Mars Rover Asset ID: {photo['id']}")

        try:
            await channel.send(embed=embed)
            log.info("Posted Mars Photo: %s to %s", photo_id, target_info["name"])
            downloaded_ids.add(photo_id)
            await save_log()
        except discord.HTTPException as e:
            log.warning("Failed to post Mars photo (%s): %s", photo_id, e)

    for photo in photos:
        await handle_photo(photo)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

async def process_all_targets(guild: discord.Guild):
    bot_role = await get_or_create_bot_role(guild)
    if not bot_role:
        log.error(
            "Could not create/assign the bot's role in %s - channels won't be "
            "locked to bot-only posting until this is fixed (check the bot has "
            "'Manage Roles' and is positioned above where it needs to create roles).",
            guild.name,
        )

    headers = {"User-Agent": "Mozilla/5.0 SpaceBot/2.0 (+discord bot)"}
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        for target_key, target_info in TARGETS.items():
            try:
                if target_info["type"] == "nasa_api":
                    await fetch_nasa_api(session, guild, target_info, bot_role)
                elif target_info["type"] == "mars_api":
                    await fetch_mars_api(session, guild, target_info, bot_role)
                elif target_info["type"] == "flickr":
                    await fetch_flickr(session, guild, target_info, bot_role)
                elif target_info["type"] == "commons":
                    await fetch_commons_category(session, guild, target_info, bot_role)
            except Exception as e:
                log.exception("Error processing target %s: %s", target_key, e)


# --------------------------------------------------------------------------
# Bot setup
# --------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@tasks.loop(minutes=FETCH_INTERVAL_MINUTES)
async def auto_fetch_space_images():
    for guild in bot.guilds:
        await process_all_targets(guild)


@bot.event
async def on_ready():
    log.info("Logged in as %s", bot.user.name)
    if not CV2_AVAILABLE:
        log.warning(
            "OpenCV is not installed - visual human detection is DISABLED. "
            "Only metadata-based filtering will run. Install opencv-python-headless "
            "and numpy to enable it."
        )
    if not auto_fetch_space_images.is_running():
        auto_fetch_space_images.start()


@bot.event
async def on_command_error(ctx: commands.Context, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need the **Manage Channels** permission to run this command.")
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"Please wait {error.retry_after:.0f}s before running this again.")
    else:
        log.exception("Command error: %s", error)
        await ctx.send("Something went wrong running that command - check the logs.")


@bot.command(name="setup_space")
@commands.has_permissions(manage_channels=True)
@commands.cooldown(1, 300, commands.BucketType.guild)
async def setup_space(ctx: commands.Context):
    await ctx.send(
        "Starting search across NASA Media/Mars archives, ESA Flickr, SpaceX Flickr, "
        "and ISRO/Roscosmos via Wikimedia Commons "
        f"(filtering out humans and non-photo renders{' + visual face check' if CV2_AVAILABLE else ''})..."
    )
    await process_all_targets(ctx.guild)
    await ctx.send("Processing complete.")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Set it as an environment variable "
            "(or in a .env file) before running the bot."
        )
    bot.run(DISCORD_TOKEN)
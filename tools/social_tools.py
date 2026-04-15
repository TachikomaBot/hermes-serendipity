"""Social media tools for Hermes.

Post, reply, quote, read feeds/threads/profiles, check notifications,
like, and follow on social media platforms.  Currently supports Bluesky
via the AT Protocol (atproto SDK).  The ``platform`` parameter on every
tool enables future backends (Mastodon, Moltbook, etc.) without schema
changes.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tools.social_tools")

# ---------------------------------------------------------------------------
# Bluesky backend
# ---------------------------------------------------------------------------

# Named feed shortcuts.  "following" uses the default home timeline;
# other names are resolved lazily to at:// URIs.
_FEED_SHORTCUTS = {
    "following": None,  # sentinel — use client.get_timeline()
}

# Feed creator handles for lazy URI resolution
_FEED_CREATORS = {
    "for-you": ("spacecowboy17.bsky.social", "for-you"),
}


def _is_bot(labels) -> bool:
    """Detect bot accounts from AT Protocol profile labels."""
    for label in (labels or []):
        val = getattr(label, "val", "") or ""
        if val == "!no-unauthenticated" or "bot" in val.lower():
            return True
    return False


def _is_following_me(viewer) -> bool:
    """Check if the profile author follows the authenticated user."""
    if not viewer:
        return False
    return bool(getattr(viewer, "followed_by", None))


def _whitelist_path() -> Path:
    """Path to the social interaction whitelist file."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "state" / "social_whitelist.json"
    except ImportError:
        return Path.home() / ".hermes" / "state" / "social_whitelist.json"


def _load_whitelist() -> set:
    path = _whitelist_path()
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {h.lower() for h in data.get("handles", [])}
    except Exception:
        return set()


def _save_whitelist(handles: set):
    path = _whitelist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"handles": sorted(handles)}, indent=2),
        encoding="utf-8",
    )


def _is_whitelisted(handle: str) -> bool:
    return handle.lower() in _load_whitelist()


def _extract_images(post_view) -> List[str]:
    """Extract image URLs from post embeds (best-effort)."""
    images: List[str] = []
    embed = getattr(post_view, "embed", None)
    if not embed:
        return images
    try:
        if hasattr(embed, "images") and embed.images:
            for img in embed.images:
                url = getattr(img, "fullsize", None) or getattr(img, "thumb", None)
                if url:
                    images.append(url)
        elif hasattr(embed, "media") and hasattr(embed.media, "images"):
            for img in embed.media.images:
                url = getattr(img, "fullsize", None) or getattr(img, "thumb", None)
                if url:
                    images.append(url)
    except Exception:
        pass
    return images


def _enrich_post(post_view) -> Dict[str, Any]:
    """Convert a PostView into a JSON-serialisable dict with bot/follower info."""
    author = post_view.author
    record = post_view.record
    result = {
        "uri": post_view.uri,
        "cid": post_view.cid,
        "author": {
            "handle": author.handle,
            "display_name": getattr(author, "display_name", "") or author.handle,
            "is_bot": _is_bot(getattr(author, "labels", [])),
            "is_following_me": _is_following_me(getattr(author, "viewer", None)),
            "is_whitelisted": _is_whitelisted(author.handle),
        },
        "text": getattr(record, "text", "") if record else "",
        "created_at": str(getattr(record, "created_at", "")) if record else "",
        "like_count": getattr(post_view, "like_count", 0) or 0,
        "repost_count": getattr(post_view, "repost_count", 0) or 0,
        "reply_count": getattr(post_view, "reply_count", 0) or 0,
    }
    imgs = _extract_images(post_view)
    if imgs:
        result["images"] = imgs
    return result


def _build_facets(text: str, client) -> List:
    """Detect @mentions and URLs in text and return Bluesky facets.

    Facets use *byte* offsets into the UTF-8 encoded text, not character
    offsets — this matches what the Bluesky app itself produces.
    """
    from atproto import models

    facets = []
    text_bytes = text.encode("utf-8")

    # --- @mentions (e.g. @alice.bsky.social) ---
    for m in re.finditer(r"(?<!\w)@([\w.-]+\.[\w.-]+)", text):
        handle = m.group(1)
        # Resolve handle → DID
        try:
            profile = client.app.bsky.actor.get_profile({"actor": handle})
            did = profile.did
        except Exception:
            continue  # skip unresolvable handles

        # Byte offsets for the full @handle span
        start = len(text[: m.start()].encode("utf-8"))
        end = len(text[: m.end()].encode("utf-8"))

        facets.append(
            models.AppBskyRichtextFacet.Main(
                index=models.AppBskyRichtextFacet.ByteSlice(
                    byte_start=start, byte_end=end,
                ),
                features=[models.AppBskyRichtextFacet.Mention(did=did)],
            )
        )

    # --- URLs (https://... or http://...) ---
    for m in re.finditer(r"https?://[^\s<>\)\]]+", text):
        url = m.group(0)
        start = len(text[: m.start()].encode("utf-8"))
        end = len(text[: m.end()].encode("utf-8"))

        facets.append(
            models.AppBskyRichtextFacet.Main(
                index=models.AppBskyRichtextFacet.ByteSlice(
                    byte_start=start, byte_end=end,
                ),
                features=[models.AppBskyRichtextFacet.Link(uri=url)],
            )
        )

    return facets if facets else None


class _BlueskyBackend:
    """Lazy-initialised Bluesky / AT Protocol backend."""

    def __init__(self):
        self._client = None
        self._resolved_feeds: Dict[str, str] = {}  # name → at:// URI cache

    # -- connection ----------------------------------------------------------

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from atproto import Client

        self._client = Client()
        self._client.login(
            os.environ["BLUESKY_HANDLE"],
            os.environ["BLUESKY_APP_PASSWORD"],
        )
        logger.info("Logged in to Bluesky as %s", os.environ["BLUESKY_HANDLE"])
        return self._client

    @property
    def own_handle(self) -> str:
        return os.environ.get("BLUESKY_HANDLE", "")

    # -- feed URI resolution -------------------------------------------------

    def _resolve_feed_uri(self, name: str) -> Optional[str]:
        """Resolve a feed shortcut name to an at:// URI, caching the result."""
        if name in self._resolved_feeds:
            return self._resolved_feeds[name]
        if name in _FEED_CREATORS:
            creator_handle, feed_rkey = _FEED_CREATORS[name]
            client = self._ensure_client()
            profile = client.app.bsky.actor.get_profile({"actor": creator_handle})
            uri = f"at://{profile.did}/app.bsky.feed.generator/{feed_rkey}"
            self._resolved_feeds[name] = uri
            return uri
        # Assume it's already an at:// URI
        return name

    # -- posting -------------------------------------------------------------

    def post(self, text: str, image_path: Optional[str] = None, image_alt: Optional[str] = None) -> Dict:
        client = self._ensure_client()
        if len(text) > 300:
            return {
                "error": f"Post is {len(text)} characters but the limit is 300. Shorten your text and try again.",
            }

        embed = None
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
            blob = client.upload_blob(img_data)
            from atproto import models
            # Use provided alt text, or auto-generate from the post text
            alt_text = image_alt or (text[:1000] if text else "Image")
            embed = models.AppBskyEmbedImages.Main(
                images=[
                    models.AppBskyEmbedImages.Image(
                        alt=alt_text,
                        image=blob.blob,
                    )
                ]
            )

        facets = _build_facets(text, client)
        response = client.send_post(text=text, embed=embed, facets=facets)
        logger.info("social_post: posted '%s'", text[:80])
        return {"status": "posted", "uri": response.uri, "cid": response.cid}

    def reply(self, text: str, parent_uri: str, parent_cid: str) -> Dict:
        client = self._ensure_client()
        if len(text) > 300:
            return {
                "error": f"Reply is {len(text)} characters but the limit is 300. Shorten your text and try again.",
            }

        from atproto import models
        from atproto_client.models.com.atproto.repo.strong_ref import Main as StrongRef

        # Resolve thread root (v1 pattern)
        root_uri, root_cid = parent_uri, parent_cid
        try:
            parent_posts = client.get_posts([parent_uri])
            if parent_posts.posts:
                parent_record = parent_posts.posts[0].record
                parent_reply = getattr(parent_record, "reply", None)
                if parent_reply and getattr(parent_reply, "root", None):
                    candidate = parent_reply.root
                    try:
                        check = client.get_posts([candidate.uri])
                        if check.posts:
                            root_uri = candidate.uri
                            root_cid = candidate.cid
                    except Exception:
                        pass  # keep parent as root
        except Exception:
            logger.debug("Could not resolve thread root, using parent", exc_info=True)

        parent_ref = StrongRef(uri=parent_uri, cid=parent_cid)
        root_ref = StrongRef(uri=root_uri, cid=root_cid)

        facets = _build_facets(text, client)
        response = client.send_post(
            text=text,
            reply_to=models.AppBskyFeedPost.ReplyRef(
                parent=parent_ref,
                root=root_ref,
            ),
            facets=facets,
        )
        logger.info("social_reply: replied to %s", parent_uri)
        return {"status": "replied", "uri": response.uri, "cid": response.cid}

    def quote_post(self, text: str, quoted_uri: str, quoted_cid: str) -> Dict:
        client = self._ensure_client()
        if len(text) > 300:
            return {
                "error": f"Quote post is {len(text)} characters but the limit is 300. Shorten your text and try again.",
            }

        from atproto_client.models.com.atproto.repo.strong_ref import Main as StrongRef
        from atproto_client.models.app.bsky.embed.record import Main as EmbedRecord

        embed = EmbedRecord(record=StrongRef(uri=quoted_uri, cid=quoted_cid))
        facets = _build_facets(text, client)
        response = client.send_post(text=text, embed=embed, facets=facets)
        logger.info("social_quote: quoted %s", quoted_uri)
        return {"status": "quoted", "uri": response.uri, "cid": response.cid}

    def thread(self, posts: List[str]) -> Dict:
        """Post a thread (series of connected posts). All posts are validated
        before any are published, so the thread either goes out in full or
        not at all."""
        if not posts:
            return {"error": "No posts provided."}
        for i, text in enumerate(posts):
            if len(text) > 300:
                return {
                    "error": (
                        f"Post {i + 1} of {len(posts)} is {len(text)} characters "
                        f"but the limit is 300. Shorten it and try again."
                    ),
                }

        client = self._ensure_client()
        results = []

        # First post
        facets = _build_facets(posts[0], client)
        resp = client.send_post(text=posts[0], facets=facets)
        root_uri, root_cid = resp.uri, resp.cid
        prev_uri, prev_cid = root_uri, root_cid
        results.append({"index": 1, "uri": resp.uri, "cid": resp.cid})
        logger.info("social_thread: posted 1/%d '%s'", len(posts), posts[0][:60])

        # Subsequent posts reply to the previous one
        from atproto import models
        from atproto_client.models.com.atproto.repo.strong_ref import Main as StrongRef

        for i, text in enumerate(posts[1:], start=2):
            facets = _build_facets(text, client)
            resp = client.send_post(
                text=text,
                reply_to=models.AppBskyFeedPost.ReplyRef(
                    parent=StrongRef(uri=prev_uri, cid=prev_cid),
                    root=StrongRef(uri=root_uri, cid=root_cid),
                ),
                facets=facets,
            )
            prev_uri, prev_cid = resp.uri, resp.cid
            results.append({"index": i, "uri": resp.uri, "cid": resp.cid})
            logger.info("social_thread: posted %d/%d '%s'", i, len(posts), text[:60])

        return {"status": "thread_posted", "count": len(results), "posts": results}

    # -- reading -------------------------------------------------------------

    def get_timeline(self, feed: Optional[str] = None, limit: int = 25) -> List[Dict]:
        client = self._ensure_client()
        limit = max(1, min(limit, 100))

        feed_lower = (feed or "following").lower().strip()

        if feed_lower == "following" or feed_lower in ("", "home"):
            response = client.get_timeline(limit=limit)
            return [_enrich_post(item.post) for item in response.feed]

        # Named feed or raw URI
        feed_uri = self._resolve_feed_uri(feed_lower)
        if feed_uri:
            response = client.app.bsky.feed.get_feed(
                {"feed": feed_uri, "limit": limit}
            )
            return [_enrich_post(item.post) for item in response.feed]

        return []

    def get_thread(self, uri: str, depth: int = 6) -> List[Dict]:
        client = self._ensure_client()
        own = self.own_handle
        out: List[Dict] = []

        def walk(node, depth_remaining: int):
            if node is None:
                return
            parent = getattr(node, "parent", None)
            if parent is not None:
                walk(parent, depth_remaining)
            post = getattr(node, "post", None)
            if post is not None:
                entry = _enrich_post(post)
                entry["is_self"] = (post.author.handle == own)
                out.append(entry)
            if depth_remaining > 0:
                for r in (getattr(node, "replies", None) or []):
                    walk(r, depth_remaining - 1)

        try:
            response = client.app.bsky.feed.get_post_thread(
                {"uri": uri, "depth": depth, "parent_height": 10}
            )
            walk(response.thread, depth)
            # De-duplicate
            seen = set()
            unique = []
            for p in out:
                if p["uri"] not in seen:
                    seen.add(p["uri"])
                    unique.append(p)
            return unique
        except Exception:
            logger.exception("get_thread failed for %s", uri)
            return []

    def get_profile(self, handle: str) -> Dict:
        client = self._ensure_client()
        profile = client.app.bsky.actor.get_profile({"actor": handle})
        result = {
            "handle": profile.handle,
            "display_name": getattr(profile, "display_name", "") or profile.handle,
            "description": getattr(profile, "description", "") or "",
            "followers_count": getattr(profile, "followers_count", 0) or 0,
            "follows_count": getattr(profile, "follows_count", 0) or 0,
            "posts_count": getattr(profile, "posts_count", 0) or 0,
            "is_bot": _is_bot(getattr(profile, "labels", [])),
            "is_following_me": _is_following_me(getattr(profile, "viewer", None)),
        }
        # Fetch recent posts
        try:
            feed_resp = client.app.bsky.feed.get_author_feed(
                {"actor": handle, "limit": 10}
            )
            result["recent_posts"] = [_enrich_post(item.post) for item in feed_resp.feed]
        except Exception:
            result["recent_posts"] = []
        return result

    def get_notifications(self, limit: int = 25, unread_only: bool = True) -> List[Dict]:
        client = self._ensure_client()
        limit = max(1, min(limit, 100))

        response = client.app.bsky.notification.list_notifications({"limit": limit})

        # Filter to unread notifications by default to prevent duplicate replies
        raw_notifs = response.notifications
        if unread_only:
            raw_notifs = [n for n in raw_notifs if not n.is_read]

        # Batch-resolve post content for reply/mention/quote notifications
        post_uris = [
            n.uri
            for n in raw_notifs
            if n.reason in ("reply", "mention", "quote")
        ]
        posts_by_uri: Dict = {}
        if post_uris:
            try:
                posts_resp = client.get_posts(post_uris)
                for p in posts_resp.posts:
                    posts_by_uri[p.uri] = p
            except Exception:
                logger.debug("Failed to batch-resolve notification posts")

        notifs = []
        for n in raw_notifs:
            author = n.author
            entry: Dict[str, Any] = {
                "reason": n.reason,
                "author": {
                    "handle": author.handle,
                    "display_name": getattr(author, "display_name", "") or author.handle,
                    "is_bot": _is_bot(getattr(author, "labels", [])),
                    "is_following_me": _is_following_me(getattr(author, "viewer", None)),
                    "is_whitelisted": _is_whitelisted(author.handle),
                },
                "is_read": n.is_read,
                "uri": n.uri,
                "cid": getattr(n, "cid", None),
            }
            # Enrich with post content
            post = posts_by_uri.get(n.uri)
            if post is not None:
                record = post.record
                entry["text"] = getattr(record, "text", "") or ""
                entry["cid"] = post.cid
                reply_ref = getattr(record, "reply", None)
                if reply_ref:
                    parent = getattr(reply_ref, "parent", None)
                    if parent:
                        entry["parent_uri"] = getattr(parent, "uri", None)
                        entry["parent_cid"] = getattr(parent, "cid", None)
            notifs.append(entry)

        # Mark notifications as seen so next call doesn't return the same ones
        if notifs:
            try:
                from datetime import datetime, timezone
                client.app.bsky.notification.update_seen(
                    {"seen_at": datetime.now(timezone.utc).isoformat()}
                )
            except Exception:
                logger.debug("Failed to mark notifications as seen")

        return notifs

    # -- engagement ----------------------------------------------------------

    def like(self, uri: str, cid: str) -> Dict:
        client = self._ensure_client()
        from atproto_client.models.com.atproto.repo.strong_ref import Main as StrongRef
        from atproto_client.models.app.bsky.feed.like import Record as LikeRecord
        from datetime import datetime, timezone

        subject = StrongRef(uri=uri, cid=cid)
        record = LikeRecord(
            subject=subject,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        client.app.bsky.feed.like.create(
            repo=client.me.did,
            record=record,
        )
        logger.info("social_like: liked %s", uri)
        return {"status": "liked", "uri": uri}

    def follow(self, handle: str) -> Dict:
        client = self._ensure_client()
        profile = client.app.bsky.actor.get_profile({"actor": handle})
        client.follow(profile.did)
        logger.info("social_follow: followed %s (%s)", handle, profile.did)
        return {"status": "followed", "handle": handle, "did": profile.did}


# Module-level singleton (lazy — no login until first tool call)
_backend = _BlueskyBackend()


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def social_post(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    try:
        result = _backend.post(
            text=args.get("text", ""),
            image_path=args.get("image_path"),
            image_alt=args.get("image_alt"),
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("social_post error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def _check_whitelist(uri: str) -> Optional[str]:
    """Check if the author of a post is whitelisted. Returns a JSON error
    string if not whitelisted, or None if the interaction is allowed."""
    try:
        client = _backend._ensure_client()
        posts = client.get_posts([uri])
        if posts.posts:
            handle = posts.posts[0].author.handle
            if not _is_whitelisted(handle):
                rkey = uri.split("/")[-1]
                link = f"https://bsky.app/profile/{handle}/post/{rkey}"
                return json.dumps({
                    "error": (
                        f"@{handle} is not whitelisted for interaction. "
                        f"Ask your handler for approval before replying/quoting. "
                        f"Post: {link}"
                    ),
                })
    except Exception as e:
        logger.debug("Whitelist check failed for %s: %s", uri, e)
    return None


def social_reply(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    uri = args.get("uri", "")
    blocked = _check_whitelist(uri)
    if blocked:
        return blocked
    try:
        result = _backend.reply(
            text=args.get("text", ""),
            parent_uri=uri,
            parent_cid=args.get("cid", ""),
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("social_reply error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_quote(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    uri = args.get("uri", "")
    blocked = _check_whitelist(uri)
    if blocked:
        return blocked
    try:
        result = _backend.quote_post(
            text=args.get("text", ""),
            quoted_uri=uri,
            quoted_cid=args.get("cid", ""),
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("social_quote error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_thread(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    posts = args.get("posts", [])
    if not isinstance(posts, list):
        return json.dumps({"error": "posts must be a list of strings."})
    try:
        result = _backend.thread(posts=posts)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("social_thread error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_read_timeline(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    try:
        posts = _backend.get_timeline(
            feed=args.get("feed"),
            limit=args.get("limit", 25),
        )
        return json.dumps({"status": "ok", "count": len(posts), "posts": posts}, ensure_ascii=False)
    except Exception as e:
        logger.error("social_read_timeline error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_read_thread(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    try:
        posts = _backend.get_thread(
            uri=args.get("uri", ""),
            depth=args.get("depth", 6),
        )
        return json.dumps({"status": "ok", "count": len(posts), "posts": posts}, ensure_ascii=False)
    except Exception as e:
        logger.error("social_read_thread error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_read_profile(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    try:
        profile = _backend.get_profile(handle=args.get("handle", ""))
        return json.dumps({"status": "ok", **profile}, ensure_ascii=False)
    except Exception as e:
        logger.error("social_read_profile error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_read_notifications(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    try:
        notifs = _backend.get_notifications(
            limit=args.get("limit", 25),
            unread_only=args.get("unread_only", True),
        )
        return json.dumps({"status": "ok", "count": len(notifs), "notifications": notifs}, ensure_ascii=False)
    except Exception as e:
        logger.error("social_read_notifications error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_like(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    try:
        result = _backend.like(uri=args.get("uri", ""), cid=args.get("cid", ""))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("social_like error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_follow(args: dict, **kw) -> str:
    platform = args.get("platform", "bluesky")
    if platform != "bluesky":
        return json.dumps({"error": f"Unsupported platform: {platform}"})
    try:
        result = _backend.follow(handle=args.get("handle", ""))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("social_follow error: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})


def social_whitelist(args: dict, **kw) -> str:
    action = args.get("action", "")
    handle = args.get("handle", "").strip().lower()

    if action == "list":
        handles = sorted(_load_whitelist())
        return json.dumps({"whitelisted": handles, "count": len(handles)})

    if action == "check":
        if not handle:
            return json.dumps({"error": "handle is required for check action"})
        return json.dumps({"handle": handle, "whitelisted": _is_whitelisted(handle)})

    if action == "add":
        if not handle:
            return json.dumps({"error": "handle is required for add action"})
        wl = _load_whitelist()
        wl.add(handle)
        _save_whitelist(wl)
        return json.dumps({"ok": True, "added": handle})

    if action == "remove":
        if not handle:
            return json.dumps({"error": "handle is required for remove action"})
        wl = _load_whitelist()
        wl.discard(handle)
        _save_whitelist(wl)
        return json.dumps({"ok": True, "removed": handle})

    return json.dumps({"error": f"Unknown action: {action}"})


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def _check_social_requirements() -> bool:
    """Social tools require the atproto SDK and Bluesky credentials."""
    try:
        import atproto  # noqa: F401
    except ImportError:
        return False
    return bool(os.getenv("BLUESKY_HANDLE") and os.getenv("BLUESKY_APP_PASSWORD"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

SOCIAL_POST_SCHEMA = {
    "name": "social_post",
    "description": (
        "Create a new post on social media. Bluesky posts are limited to "
        "300 characters; posts exceeding this limit will be rejected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "text": {
                "type": "string",
                "description": "Text content of the post.",
            },
            "image_path": {
                "type": "string",
                "description": "Optional absolute path to an image file to attach.",
            },
            "image_alt": {
                "type": "string",
                "description": "Alt text for the image (for accessibility). Auto-generated from post text if omitted.",
            },
        },
        "required": ["text"],
    },
}

SOCIAL_REPLY_SCHEMA = {
    "name": "social_reply",
    "description": (
        "Reply to an existing post. The target account must be whitelisted — "
        "if not, the tool returns an error with the post link. Ask your "
        "handler for approval, then use social_whitelist to add the account."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "text": {
                "type": "string",
                "description": "Reply text.",
            },
            "uri": {
                "type": "string",
                "description": "AT URI of the post to reply to.",
            },
            "cid": {
                "type": "string",
                "description": "CID of the post to reply to.",
            },
        },
        "required": ["text", "uri", "cid"],
    },
}

SOCIAL_QUOTE_SCHEMA = {
    "name": "social_quote",
    "description": (
        "Quote an existing post with your own commentary. The target account "
        "must be whitelisted — if not, the tool returns an error with the post "
        "link. Ask your handler for approval, then use social_whitelist to add "
        "the account."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "text": {
                "type": "string",
                "description": "Your commentary text.",
            },
            "uri": {
                "type": "string",
                "description": "AT URI of the post to quote.",
            },
            "cid": {
                "type": "string",
                "description": "CID of the post to quote.",
            },
        },
        "required": ["text", "uri", "cid"],
    },
}

SOCIAL_THREAD_SCHEMA = {
    "name": "social_thread",
    "description": (
        "Post a thread (series of connected posts). Each post must be 300 "
        "characters or fewer. All posts are validated before any are published, "
        "so the thread either goes out in full or not at all. Use this when you "
        "want to say something that doesn't fit in a single post."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "posts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of post texts, in order. Each must be <= 300 characters.",
            },
        },
        "required": ["posts"],
    },
}

SOCIAL_READ_TIMELINE_SCHEMA = {
    "name": "social_read_timeline",
    "description": (
        "Read a social media feed. Supports 'following' (default home timeline), "
        "'for-you' (algorithmic feed), or a custom feed AT URI."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "feed": {
                "type": "string",
                "description": (
                    "Feed to read: 'following' (default), 'for-you', or a feed AT URI."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max posts to return (default 25, max 100).",
            },
        },
        "required": [],
    },
}

SOCIAL_READ_THREAD_SCHEMA = {
    "name": "social_read_thread",
    "description": "Read a post thread including parent chain and replies.",
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "uri": {
                "type": "string",
                "description": "AT URI of the post to read the thread for.",
            },
            "depth": {
                "type": "integer",
                "description": "How many levels of replies to fetch (default 6).",
            },
        },
        "required": ["uri"],
    },
}

SOCIAL_READ_PROFILE_SCHEMA = {
    "name": "social_read_profile",
    "description": "Read a user's profile info and recent posts.",
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "handle": {
                "type": "string",
                "description": "User handle (e.g. 'alice.bsky.social').",
            },
        },
        "required": ["handle"],
    },
}

SOCIAL_READ_NOTIFICATIONS_SCHEMA = {
    "name": "social_read_notifications",
    "description": (
        "Check recent notifications: likes, replies, follows, mentions, quotes. "
        "Reply and mention notifications include the full post text. "
        "By default only returns unread notifications and marks them as seen, "
        "preventing duplicate processing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "limit": {
                "type": "integer",
                "description": "Max notifications to return (default 25, max 100).",
            },
            "unread_only": {
                "type": "boolean",
                "description": "Only return unread notifications (default true). Set false to include already-seen ones.",
            },
        },
        "required": [],
    },
}

SOCIAL_LIKE_SCHEMA = {
    "name": "social_like",
    "description": "Like a post on social media.",
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "uri": {
                "type": "string",
                "description": "AT URI of the post to like.",
            },
            "cid": {
                "type": "string",
                "description": "CID of the post to like.",
            },
        },
        "required": ["uri", "cid"],
    },
}

SOCIAL_FOLLOW_SCHEMA = {
    "name": "social_follow",
    "description": "Follow a user on social media.",
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["bluesky"],
                "description": "Social media platform (default: bluesky).",
            },
            "handle": {
                "type": "string",
                "description": "Handle of the user to follow.",
            },
        },
        "required": ["handle"],
    },
}


SOCIAL_WHITELIST_SCHEMA = {
    "name": "social_whitelist",
    "description": (
        "Manage the social interaction whitelist. Accounts on this list "
        "can be replied to or quoted freely. Use 'add' after getting "
        "handler approval to interact with a new account."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "remove", "list", "check"],
                "description": "Action to perform.",
            },
            "handle": {
                "type": "string",
                "description": "Bluesky handle (e.g. user.bsky.social). Required for add/remove/check.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

_SOCIAL_TOOLS = [
    ("social_post",               SOCIAL_POST_SCHEMA,               social_post,               "📝", "Post to social media"),
    ("social_reply",              SOCIAL_REPLY_SCHEMA,              social_reply,              "💬", "Reply to a post"),
    ("social_quote",              SOCIAL_QUOTE_SCHEMA,              social_quote,              "🔁", "Quote a post"),
    ("social_thread",             SOCIAL_THREAD_SCHEMA,             social_thread,             "🧵", "Post a thread"),
    ("social_read_timeline",      SOCIAL_READ_TIMELINE_SCHEMA,      social_read_timeline,      "📰", "Read social media feed"),
    ("social_read_thread",        SOCIAL_READ_THREAD_SCHEMA,        social_read_thread,        "🧵", "Read a post thread"),
    ("social_read_profile",       SOCIAL_READ_PROFILE_SCHEMA,       social_read_profile,       "👤", "Read a user profile"),
    ("social_read_notifications", SOCIAL_READ_NOTIFICATIONS_SCHEMA, social_read_notifications, "🔔", "Check social notifications"),
    ("social_like",               SOCIAL_LIKE_SCHEMA,               social_like,               "❤️", "Like a post"),
    ("social_follow",             SOCIAL_FOLLOW_SCHEMA,             social_follow,             "➕", "Follow a user"),
    ("social_whitelist",          SOCIAL_WHITELIST_SCHEMA,          social_whitelist,          "✅", "Manage interaction whitelist"),
]

for _name, _schema, _handler, _emoji, _desc in _SOCIAL_TOOLS:
    registry.register(
        name=_name,
        toolset="social",
        schema=_schema,
        handler=_handler,
        check_fn=_check_social_requirements,
        requires_env=["BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD"],
        emoji=_emoji,
        description=_desc,
    )

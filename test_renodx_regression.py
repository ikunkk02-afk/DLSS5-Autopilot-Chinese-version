"""Regression coverage for RenoDX/Feeder identity and network fallbacks."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from core import installer, net, prefs, sources


def _pe(*markers: bytes, size: int = 300_000) -> bytes:
    data = b"MZ" + b"\0" * size
    return data + b"\0".join(markers)


FEEDER = _pe(
    b"DLSS 5 Feed 0.15.1",
    b"DLSS5_Feed.fx",
    b"Feeds DLSS 5 neural rendering",
    b"dlss5-feed.cfg",
    b"RenoDX.DLSS5",  # the feeder mentions the provider it configures
    b"DLSS",
)
RENODX_460 = _pe(
    b"RenoDX.DLSS5",
    b"DLSS 5 Neural Rendering",
    b"DLSS5 Generic",
    b"renodx-dlss5.addon64",
)


class RenoDXIdentityTests(unittest.TestCase):
    def test_find_renodx_rejects_feeder_under_original_name(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / "dlss5-feed.addon64"
            candidate.write_bytes(FEEDER)
            with mock.patch.object(prefs, "app_dir", return_value=root), \
                    mock.patch.object(prefs, "get", return_value=None):
                found, candidates = prefs.find_renodx()
            self.assertNotEqual(found, candidate)
            self.assertNotIn(candidate, candidates)

    def test_remembered_feeder_is_not_treated_as_non_sf_renodx(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / "dlss5-feed.addon64"
            candidate.write_bytes(FEEDER)
            with mock.patch.object(prefs, "app_dir", return_value=root), \
                    mock.patch.object(prefs, "get", return_value=str(candidate)):
                found, candidates = prefs.find_renodx(sf=False)
            self.assertIsNone(found)
            self.assertNotIn(candidate, candidates)

    def test_renamed_feeder_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            candidate = Path(td) / "renodx-dlss5.addon64"
            candidate.write_bytes(FEEDER)
            self.assertTrue(prefs.is_dlss5_feeder(candidate))
            self.assertFalse(prefs.is_renodx(candidate))

    def test_genuine_renodx_460_identity_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            candidate = Path(td) / "renodx-dlss5.addon64"
            candidate.write_bytes(RENODX_460)
            self.assertTrue(prefs.is_renodx(candidate))
            self.assertFalse(prefs.is_dlss5_feeder(candidate))

    def test_equal_feeder_and_renodx_files_block_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / installer.FEEDER_ADDON64).write_bytes(FEEDER)
            (root / installer.RENODX).write_bytes(FEEDER)
            with self.assertRaisesRegex(installer.InstallError, "same file"):
                installer._validate_neural_addons(
                    root, root, installer.Options(path="feeder"), True
                )


class NetworkRegressionTests(unittest.TestCase):
    def test_api_failure_uses_stale_cache(self):
        url = "https://api.github.com/repos/example/project/releases"
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(sources, "_API_CACHE", Path(td)), \
                mock.patch.object(sources, "_get", side_effect=urllib.error.URLError("offline")):
            path = sources._cache_path(url)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"cached": True}), encoding="utf8")
            old = time.time() - 8 * 3600
            os.utime(path, (old, old))
            sources.last_fallback = None
            self.assertEqual(sources._json(url), {"cached": True})
            self.assertEqual(
                sources.last_fallback,
                "Online version list update failed; using cache from 8 hours ago.",
            )

    def test_asset_failure_names_resource_and_url(self):
        url = "https://github.com/example/project/releases/download/v1/renodx.zip"
        with mock.patch.object(net, "download", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(installer.InstallError) as raised:
                installer._download_resource(url, "renodx-4.60.zip")
        message = str(raised.exception)
        self.assertIn("renodx-4.60.zip", message)
        self.assertIn(url, message)

    def test_github_api_403_and_429_are_distinct(self):
        url = "https://api.github.com/repos/example/project/releases"
        for code, text in ((403, "rejected"), (429, "rate limited")):
            error = urllib.error.HTTPError(url, code, "failure", {}, None)
            with self.subTest(code=code), \
                    mock.patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(sources.RateLimited, text):
                    sources._get(url, attempts=1)


if __name__ == "__main__":
    unittest.main()

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from main import parse_args, build_output_paths


def test_parse_args_basic():
    args = parse_args(["kubernetes"])
    assert args.topic == "kubernetes"
    assert args.lang is None
    assert args.text_only is False
    assert args.audio_only is False
    assert args.count is None


def test_parse_args_full():
    args = parse_args(["docker", "--lang", "en", "--text-only", "--count", "15"])
    assert args.topic == "docker"
    assert args.lang == "en"
    assert args.text_only is True
    assert args.count == 15


def test_build_output_paths():
    paths = build_output_paths("kubernetes")
    assert paths["base"] == "output/kubernetes"
    assert paths["data_json"] == "output/kubernetes/data.json"
    assert paths["interview_md"] == "output/kubernetes/interview.md"
    assert paths["audio_dir"] == "output/kubernetes/audio"
    assert paths["podcast"] == "output/kubernetes/podcast.mp3"

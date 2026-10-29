"""A 401 is reported to the user, not retried.

Run inside the container: bash docker/run.sh test

The rule these tests hold: when a source answers 401, the failure is named, the
source is named, and the user is asked to check availability or download the file
directly. Nothing here retries, changes credentials, or guesses another URL.
"""

import report
from conftest import REPO


def test_a_401_is_recognised_as_an_auth_failure_and_other_failures_are_not():
    assert report.is_auth_failure("error pulling image configuration: 401 Unauthorized")
    assert report.is_auth_failure("HEAD 401 authentication required")
    assert not report.is_auth_failure("manifest unknown")
    assert not report.is_auth_failure(
        "dial tcp: lookup registry-1.docker.io: no such host"
    )


def test_the_message_names_the_source_and_asks_for_support():
    text = report.message(
        "docker manifest inspect nvidia/cuda:13.1.0-devel-ubuntu24.04",
        "requested pull from the registry returned 401",
    )
    assert "download failed" in text
    assert "nvidia/cuda:13.1.0-devel-ubuntu24.04" in text
    assert "401" in text
    assert "check the availability" in text
    assert "download it directly" in text
    # The reason is kept, because it is what the user has to look at.
    assert "requested pull from the registry returned 401" in text


def test_a_non_auth_failure_is_reported_without_asking_for_support():
    text = report.message(
        "docker manifest inspect nvidia/cuda:12.4.1-devel-ubuntu24.04",
        "manifest unknown",
    )
    assert "download failed" in text
    assert "401" not in text
    assert "check the availability" not in text


def test_notify_prints_the_same_message_it_returns(capsys):
    returned = report.notify(
        "pip download into /workspace/out/wheelhouse",
        "401 Unauthorized",
    )
    captured = capsys.readouterr().out
    assert captured == returned + "\n"


def test_the_shell_helper_says_the_same_thing_as_the_python_module():
    # docker/run.sh and docker/build.sh source docker/pull.sh, and a shell script
    # cannot import this module, so the sentence is duplicated there. This test is
    # what stops the two copies from drifting apart.
    shell = (REPO / "docker" / "pull.sh").read_text().splitlines()
    start = next(i for i, line in enumerate(shell) if "grep -Eiq" in line)
    lines = [
        line.strip() for line in shell[start:] if line.strip().startswith('echo "')
    ]
    joined = " ".join(line.split('"')[1] for line in lines)
    assert joined == report.AUTH_SENTENCE

"""Report a download or registry failure to the user instead of retrying it.

A 401 from a registry or an index is not retried by this repo. It is an
availability problem on the source, which a build cannot fix by trying again, so
the message names the source and asks the user to check it or download the file
directly. Nothing here retries, changes credentials, or guesses an alternate URL.

The rule this module exists for:

  At errors 401, notify the user, verify the source, or download it directly.

Kept as a separate module because the same wording has to appear in three places
(the packaging download, the CUDA registry check, and the container run script),
and a message that is written three times is three messages.
"""

from __future__ import annotations

AUTH_MARKERS = (
    "401",
    "unauthorized",
    "unauthorised",
    "authentication required",
    "authorization required",
)

AUTH_SENTENCE = (
    "the source answered 401, so this repo does not retry it. Please check the "
    "availability of the source, or download it directly and put the file where "
    "the build expects it."
)


def is_auth_failure(detail: str) -> bool:
    """True when a failure looks like a 401 rather than a missing file or tag."""
    text = detail.lower()
    return any(marker in text for marker in AUTH_MARKERS)


def message(source: str, detail: str) -> str:
    """The notification: what failed, where, and what is asked of the user.

    `source` is the URL or image the build was reaching for, so the message is
    actionable on its own. `detail` is the tool output, kept because the reason
    for a refusal is usually in it.
    """
    lines = [f"download failed: {source}"]
    if is_auth_failure(detail):
        lines.append(AUTH_SENTENCE)
    if detail:
        lines.append(detail.strip())
    return "\n".join(lines)


def notify(source: str, detail: str) -> str:
    """Print the notification and return it, so a caller can also record it."""
    text = message(source, detail)
    print(text)
    return text

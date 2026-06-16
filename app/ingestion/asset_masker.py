import re

from app.assets.resolver import detect_asset_tokens

IMAGE_PLACEHOLDER_PATTERN = re.compile(
    r"<!--\s*image\s*-->|!\[[^\]]*\]\([^)]*\)",
    flags=re.IGNORECASE,
)


def append_asset_tokens(text: str, tokens: list[str]) -> str:
    existing = set(detect_asset_tokens(text))
    missing = [token for token in dict.fromkeys(tokens) if token not in existing]
    if not missing:
        return text
    suffix = "\n\n" + "\n".join(f"Image: {token}" for token in missing)
    return f"{text.rstrip()}{suffix}".strip()


def mask_asset_positions(text: str, tokens: list[str]) -> str:
    unique_tokens = list(dict.fromkeys(tokens))
    token_index = 0

    def replace_placeholder(match: re.Match[str]) -> str:
        nonlocal token_index
        if token_index >= len(unique_tokens):
            return match.group(0)
        token = unique_tokens[token_index]
        token_index += 1
        return token

    masked = IMAGE_PLACEHOLDER_PATTERN.sub(replace_placeholder, text)
    return append_asset_tokens(masked, unique_tokens[token_index:])


def first_asset_token(value: str) -> str | None:
    tokens = detect_asset_tokens(value)
    return tokens[0] if tokens else None

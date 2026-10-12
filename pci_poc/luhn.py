"""Length-aware Luhn validation helpers."""


MIN_PAN_LENGTH = 13
MAX_PAN_LENGTH = 19


def is_luhn_valid(candidate: str) -> bool:
    """Return whether an ASCII digit string has a valid PAN length and checksum."""
    if not candidate.isascii() or not candidate.isdigit():
        return False
    if not MIN_PAN_LENGTH <= len(candidate) <= MAX_PAN_LENGTH:
        return False

    checksum = 0
    parity = len(candidate) % 2
    for index, character in enumerate(candidate):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def append_luhn_check_digit(prefix: str) -> str:
    """Return a Luhn-valid number by appending a check digit to a digit prefix."""
    if not prefix.isascii() or not prefix.isdigit():
        raise ValueError("prefix must contain ASCII digits only")
    if not MIN_PAN_LENGTH - 1 <= len(prefix) < MAX_PAN_LENGTH:
        raise ValueError("prefix length cannot produce a supported PAN length")

    for check_digit in "0123456789":
        candidate = prefix + check_digit
        if is_luhn_valid(candidate):
            return candidate
    raise RuntimeError("could not calculate a Luhn check digit")
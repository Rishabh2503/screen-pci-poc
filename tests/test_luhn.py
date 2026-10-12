from pci_poc.luhn import append_luhn_check_digit, is_luhn_valid


def test_validates_length_and_luhn_checksum():
    valid = append_luhn_check_digit("400000123456789")
    assert valid == "4000001234567899"
    assert is_luhn_valid(valid)
    assert not is_luhn_valid(valid[:-1] + "0")
    assert not is_luhn_valid("123456789012")
    assert not is_luhn_valid("４０００００１２３４５６７８９９")


def test_check_digit_rejects_non_digit_prefix():
    try:
        append_luhn_check_digit("4000-123")
    except ValueError as error:
        assert "ASCII digits" in str(error)
    else:
        raise AssertionError("invalid prefix should be rejected")
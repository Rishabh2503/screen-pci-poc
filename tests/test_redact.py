import numpy as np
from PIL import Image

from pci_poc.detect import Detection
from pci_poc.redact import redact_image


def test_cover_redaction_changes_only_detected_region(tmp_path):
    pixels = np.zeros((60, 80, 3), dtype=np.uint8)
    pixels[:, :] = (240, 240, 240)
    pixels[15:40, 20:60] = (20, 80, 160)
    source = tmp_path / "synthetic.png"
    destination = tmp_path / "redacted.png"
    Image.fromarray(pixels).save(source)

    redact_image(source, [Detection((20, 15, 60, 40), "test", 1)], destination)
    result = np.asarray(Image.open(destination))

    assert destination.is_file()
    assert np.all(result[15:40, 20:60] == 0)
    assert np.array_equal(result[:15], pixels[:15])
    assert np.array_equal(result[:, :20], pixels[:, :20])


def test_empty_detection_list_saves_unchanged_copy(tmp_path):
    source = tmp_path / "synthetic.png"
    destination = tmp_path / "copy.png"
    Image.new("RGB", (30, 20), "white").save(source)

    redact_image(source, [], destination)

    assert np.array_equal(np.asarray(Image.open(source)), np.asarray(Image.open(destination)))


def test_blur_redaction_changes_detected_region(tmp_path):
    pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    pixels[:, :] = (240, 240, 240)
    pixels[20:60, 20:80] = np.indices((40, 60)).sum(axis=0)[:, :, None] % 2 * 255
    source = tmp_path / "synthetic.png"
    destination = tmp_path / "blurred.png"
    Image.fromarray(pixels).save(source)

    redact_image(
        source,
        [Detection((20, 20, 80, 60), "test", 1)],
        destination,
        method="blur",
    )
    result = np.asarray(Image.open(destination))

    assert destination.is_file()
    assert not np.array_equal(result[20:60, 20:80], pixels[20:60, 20:80])
    assert np.array_equal(result[:20], pixels[:20])
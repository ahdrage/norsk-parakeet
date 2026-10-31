import numpy as np

from norsk_parakeet.long_audio import SR, fix_number_spacing, split_at_pauses


def test_short_audio_is_one_chunk():
    wav = np.ones(10 * SR, np.float32)
    assert [len(c) for c in split_at_pauses(wav)] == [len(wav)]
    assert len(split_at_pauses(np.zeros(100, np.float32))) == 1


def test_chunks_cover_the_audio_and_respect_the_limit():
    wav = np.random.default_rng(0).normal(size=125 * SR).astype(np.float32)
    chunks = split_at_pauses(wav)
    assert sum(map(len, chunks)) == len(wav)
    assert np.array_equal(np.concatenate(chunks), wav)
    assert all(len(c) <= 30 * SR for c in chunks)


def test_cuts_fall_in_silence():
    wav = np.random.default_rng(0).normal(size=60 * SR).astype(np.float32)
    wav[25 * SR:26 * SR] = 0
    first = split_at_pauses(wav)[0]
    assert 25 * SR <= len(first) <= 26 * SR


def test_fix_number_spacing():
    assert fix_number_spacing("i september2017 og cirka400 kr") == "i september 2017 og cirka 400 kr"
    assert fix_number_spacing("CO2 og G20, 10,5 prosent, 1 000") == "CO2 og G20, 10,5 prosent, 1 000"

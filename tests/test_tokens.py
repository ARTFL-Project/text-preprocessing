"""
Tests for the Tokens container and PreprocessorToken classes.

Most Tokens methods that update metadata (slice, split_tokens, pop, popleft,
append, appendleft, extend, purge) access token.ext["start_byte"] /
token.ext["end_byte"].  Plain-text processing does not populate those fields,
so we use the make_token / make_tokens helpers from conftest to build Tokens
objects that carry proper byte-position metadata.
"""

import os
import pytest

from text_preprocessing import Tokens
from text_preprocessing import Token as PreprocessorToken

from conftest import make_token, make_tokens


# ===========================================================================
# PreprocessorToken
# ===========================================================================

class TestPreprocessorToken:

    def test_str_returns_text(self):
        t = make_token("hello", 0, 5)
        assert str(t) == "hello"

    def test_text_attribute(self):
        t = make_token("hello", 0, 5)
        assert t.text == "hello"

    def test_surface_form_from_ext(self):
        t = make_token("hello", 0, 5)
        assert t.surface_form == "hello"

    def test_ext_contains_byte_positions(self):
        t = make_token("hello", 10, 15)
        assert t.ext["start_byte"] == 10
        assert t.ext["end_byte"] == 15

    def test_equality_with_string(self):
        t = make_token("hello")
        assert t == "hello"
        assert t != "world"

    def test_equality_between_tokens(self):
        t1 = make_token("hello", 0, 5)
        t2 = make_token("hello", 99, 104)
        assert t1 == t2  # equality is on .text, not position

    def test_hashable(self):
        t = make_token("hello")
        d = {t: 1}
        assert d[t] == 1

    def test_add_concatenates_with_string(self):
        t = make_token("hello")
        assert t + " world" == "hello world"

    def test_repr_contains_text_and_surface(self):
        t = make_token("hello")
        r = repr(t)
        assert "hello" in r
        assert "surface_form" in r


# ===========================================================================
# Tokens — construction
# ===========================================================================

class TestTokensConstruction:

    def test_from_list_of_preprocessor_tokens(self):
        token_list = [make_token("hello"), make_token("world")]
        tokens = Tokens(token_list, metadata={"filename": "test"})
        assert len(tokens) == 2

    def test_metadata_stored(self):
        tokens = make_tokens(["hello", "world"])
        assert tokens.metadata["filename"] == "test.txt"

    def test_truthy_when_non_empty(self):
        tokens = make_tokens(["hello"])
        assert bool(tokens) is True

    def test_falsy_when_empty(self):
        tokens = Tokens([], metadata={"filename": "test"})
        assert bool(tokens) is False

    def test_len_reflects_token_count(self):
        tokens = make_tokens(["a", "b", "c", "d"])
        assert len(tokens) == 4


# ===========================================================================
# Tokens — iteration and indexing
# ===========================================================================

class TestTokensAccess:

    def test_iter_yields_preprocessor_tokens(self):
        tokens = make_tokens(["hello", "world"])
        for t in tokens:
            assert isinstance(t, PreprocessorToken)

    def test_iter_text_values(self):
        tokens = make_tokens(["hello", "world", "foo"])
        assert [t.text for t in tokens] == ["hello", "world", "foo"]

    def test_next_yields_all_tokens(self):
        tokens = make_tokens(["a", "b", "c"])
        result = []
        try:
            while True:
                result.append(next(tokens))
        except StopIteration:
            pass
        assert [t.text for t in result] == ["a", "b", "c"]

    def test_next_raises_stop_iteration(self):
        tokens = make_tokens(["a"])
        next(tokens)
        with pytest.raises(StopIteration):
            next(tokens)

    def test_next_resets_after_exhaustion(self):
        tokens = make_tokens(["a", "b"])
        list(tokens)  # exhaust via __iter__
        # __next__ should work again from the start
        assert next(tokens).text == "a"

    def test_getitem_int(self):
        tokens = make_tokens(["hello", "world"])
        assert tokens[0].text == "hello"
        assert tokens[1].text == "world"

    def test_getitem_negative_int(self):
        tokens = make_tokens(["hello", "world", "foo"])
        assert tokens[-1].text == "foo"

    def test_getitem_slice_returns_tokens(self):
        tokens = make_tokens(["hello", "world", "foo", "bar"])
        sliced = tokens[1:3]
        assert isinstance(sliced, Tokens)
        assert [t.text for t in sliced] == ["world", "foo"]

    def test_getitem_slice_updates_metadata_bytes(self):
        tokens = make_tokens(["hello", "world", "foo"])
        sliced = tokens[0:2]
        assert sliced.metadata["start_byte"] == tokens[0].ext["start_byte"]
        assert sliced.metadata["end_byte"] == tokens[1].ext["end_byte"]

    def test_getitem_slice_empty(self):
        tokens = make_tokens(["hello", "world"])
        sliced = tokens[5:10]
        assert len(sliced) == 0
        assert sliced.metadata["start_byte"] == 0
        assert sliced.metadata["end_byte"] == 0


# ===========================================================================
# Tokens — deque mutations
# ===========================================================================

class TestTokensMutations:

    def test_append_increases_length(self):
        tokens = make_tokens(["hello"])
        tokens.append(make_token("world", 10, 15))
        assert len(tokens) == 2
        assert tokens[-1].text == "world"

    def test_append_updates_end_byte(self):
        tokens = make_tokens(["hello"])
        tokens.append(make_token("world", 10, 15))
        assert tokens.metadata["end_byte"] == 15

    def test_appendleft_increases_length(self):
        tokens = make_tokens(["world"])
        tokens.appendleft(make_token("hello", 0, 5))
        assert len(tokens) == 2
        assert tokens[0].text == "hello"

    def test_appendleft_updates_start_byte(self):
        tokens = make_tokens(["world"])
        tokens.appendleft(make_token("hello", 0, 5))
        assert tokens.metadata["start_byte"] == 0

    def test_pop_removes_last_token(self):
        tokens = make_tokens(["hello", "world", "foo"])
        popped = tokens.pop()
        assert popped.text == "foo"
        assert len(tokens) == 2

    def test_popleft_removes_first_token(self):
        tokens = make_tokens(["hello", "world", "foo"])
        popped = tokens.popleft()
        assert popped.text == "hello"
        assert len(tokens) == 2

    def test_pop_empty_returns_none(self):
        tokens = Tokens([], metadata={"filename": "t", "start_byte": 0, "end_byte": 0})
        assert tokens.pop() is None

    def test_popleft_empty_returns_none(self):
        tokens = Tokens([], metadata={"filename": "t", "start_byte": 0, "end_byte": 0})
        assert tokens.popleft() is None

    def test_extend_combines_tokens(self):
        t1 = make_tokens(["hello", "world"])
        t2 = make_tokens(["foo", "bar"], metadata={"filename": "test.txt", "start_byte": 20, "end_byte": 30})
        t1.extend(t2)
        assert len(t1) == 4
        assert [t.text for t in t1] == ["hello", "world", "foo", "bar"]

    def test_extend_updates_end_byte(self):
        t1 = make_tokens(["hello"])
        t2 = make_tokens(["world"], metadata={"filename": "test.txt", "start_byte": 100, "end_byte": 200})
        t1.extend(t2)
        assert t1.metadata["end_byte"] == 200

    def test_purge_removes_empty_tokens(self):
        token_list = [
            make_token("hello", 0, 5),
            PreprocessorToken("", ext={"token": "", "start_byte": 6, "end_byte": 6}),
            make_token("world", 7, 12),
        ]
        tokens = Tokens(token_list, {"filename": "test", "start_byte": 0, "end_byte": 12})
        tokens.purge()
        assert len(tokens) == 2
        assert [t.text for t in tokens] == ["hello", "world"]

    def test_purge_removes_space_tokens(self):
        token_list = [
            make_token("hello", 0, 5),
            PreprocessorToken(" ", ext={"token": " ", "start_byte": 5, "end_byte": 6}),
            make_token("world", 6, 11),
        ]
        tokens = Tokens(token_list, {"filename": "test", "start_byte": 0, "end_byte": 11})
        tokens.purge()
        assert all(t.text != " " for t in tokens)

    def test_purge_empty_container_sets_bytes_to_zero(self):
        token_list = [
            PreprocessorToken("", ext={"token": "", "start_byte": 0, "end_byte": 0}),
        ]
        tokens = Tokens(token_list, {"filename": "test", "start_byte": 0, "end_byte": 0})
        tokens.purge()
        assert len(tokens) == 0
        assert tokens.metadata["start_byte"] == 0
        assert tokens.metadata["end_byte"] == 0


# ===========================================================================
# Tokens — split_tokens
# ===========================================================================

class TestSplitTokens:

    def test_split_yields_correct_number_of_chunks(self):
        tokens = make_tokens(["a", "b", "c", "d", "e", "f"])
        chunks = list(tokens.split_tokens(2))
        assert len(chunks) == 3

    def test_split_chunk_text(self):
        tokens = make_tokens(["a", "b", "c", "d"])
        chunks = list(tokens.split_tokens(2))
        assert [t.text for t in chunks[0]] == ["a", "b"]
        assert [t.text for t in chunks[1]] == ["c", "d"]

    def test_split_chunks_are_tokens_instances(self):
        tokens = make_tokens(["a", "b", "c"])
        for chunk in tokens.split_tokens(2):
            assert isinstance(chunk, Tokens)


# ===========================================================================
# Tokens — save / load
# ===========================================================================

class TestSaveLoad:

    def test_round_trip_preserves_text(self, tmp_path):
        tokens = make_tokens(["hello", "world", "foo"])
        path = str(tmp_path / "tokens.pkl")
        tokens.save(path)
        loaded = Tokens.load(path)
        assert [t.text for t in loaded] == ["hello", "world", "foo"]

    def test_round_trip_preserves_metadata(self, tmp_path):
        tokens = make_tokens(["hello", "world"])
        path = str(tmp_path / "tokens.pkl")
        tokens.save(path)
        loaded = Tokens.load(path)
        assert loaded.metadata["filename"] == "test.txt"

    def test_round_trip_preserves_byte_positions(self, tmp_path):
        tokens = make_tokens(["hello"])
        path = str(tmp_path / "tokens.pkl")
        tokens.save(path)
        loaded = Tokens.load(path)
        assert loaded[0].ext["start_byte"] == 0
        assert loaded[0].ext["end_byte"] == 5

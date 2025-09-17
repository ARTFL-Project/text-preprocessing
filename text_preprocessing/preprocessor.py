#!/usr/bin/env python3
"""Text Preprocessor"""

import gc
import multiprocessing as mp
import os
import sqlite3
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, DefaultDict, Deque, Iterable

import lz4.frame
import orjson
import regex as re
import spacy
import torch
from multiprocess.pool import Pool
from spacy.language import Language
from spacy.tokens import Doc, Token

from .modernizer import Modernizer
from .spacy_helpers import PreprocessorToken, Tokens, load_language_model

# Suppress all UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)
mp.set_start_method("spawn", force=True)

Doc.set_extension("metadata", default={})
Doc.set_extension("char_num", default=0)
Token.set_extension("ext", default={})

TAGS = re.compile(r"<[^>]+>")

PHILO_TEXT_OBJECT_TYPE: dict[str, int] = {
    "doc": 1,
    "div1": 2,
    "div2": 3,
    "div3": 4,
    "para": 5,
    "sent": 6,
    "word": 7,
}

PHILO_OBJECT_LEVEL: dict[int, str] = {1: "doc", 2: "div1", 3: "div2", 4: "div3", 5: "para", 6: "sent", 7: "word"}


def check_gpu_ram():
    """Returns the percentage of GPU memory being used."""
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    allocated_percent = (allocated / total) * 100

    if allocated_percent > 20:  # This is is only a subset of GPU RAM usage, but indicative of high usage
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        gc.collect()


def process_batch_texts(
    queue, text_fetcher_args, batch_texts, language_model, normalize_options, do_nlp, keep_all, progress_info
):
    nlp, using_gpu = load_language_model(language_model, normalize_options)
    text_fetcher = TextFetcher(nlp, **text_fetcher_args)  # Initialize text_fetcher with required params
    previous_philo_id = None
    for tokens, _ in text_fetcher(batch_texts, do_nlp=do_nlp, keep_all=keep_all, progress=False):
        if isinstance(tokens, PreparedDoc):
            spacy_doc = make_spacy_doc(nlp, tokens)
            if spacy_doc._.char_num > 10000 and using_gpu is True:
                split_doc = split_spacy_docs(nlp, spacy_doc)
                doc = Doc.from_docs(list(nlp.pipe(split_doc, batch_size=64)))
                doc._.metadata = spacy_doc._.metadata
                tokens = Tokens(doc, keep_all=keep_all)
            else:
                tokens = Tokens(nlp(spacy_doc), keep_all=keep_all)
        elif isinstance(tokens, Doc):
            tokens = Tokens(tokens, keep_all=keep_all)
        if using_gpu:
            check_gpu_ram()
        current_doc_id = tokens.metadata.get("philo_id").split()[0]
        if previous_philo_id != current_doc_id:
            progress_info["doc_count"] += 1
        if progress_info["progress"] is True:
            progress_info["count"] += 1
            if text_fetcher_args["text_object_type"] == "doc":
                print(
                    f"\r{progress_info['progress_prefix']} {progress_info['count']} texts processed...",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r{progress_info['progress_prefix']} {progress_info['count']} text chunks of {progress_info['doc_count']} documents processed...",
                    end="",
                    flush=True,
                )
        previous_philo_id = current_doc_id
        queue.put(tokens)
    queue.put(None)


def split_spacy_docs(nlp, doc: Doc) -> list[Doc]:
    """Split spacy doc into smaller docs of 10 sentences"""
    sentence_group: list[Doc] = []
    docs: list[Doc] = []
    for sent in doc.sents:
        if len(sentence_group) == 10:
            docs.append(Doc.from_docs(sentence_group))
            sentence_group = []
        else:
            sent_starts = []
            words = []
            for token in sent:
                sent_starts.append(token.is_sent_start)
                words.append(token.text)
            sent_doc = Doc(nlp.vocab, words, sent_starts=sent_starts)
            for pos, token in enumerate(sent):
                sent_doc[pos]._.ext = token._.ext
            sentence_group.append(sent_doc)
    if sentence_group:
        docs.append(Doc.from_docs(sentence_group))
    return docs


@dataclass(slots=True)
class PreparedDoc:
    """Prepared doc for conversion to Spacy Doc object"""

    words: list[str]
    sent_starts: list[bool]
    metadata: dict[str, Any]
    exts: list[dict[str, Any]]
    char_num: int


class PreProcessor:
    """Text Preprocessing class"""

    def __init__(
        self,
        word_regex: str = r"[\p{L}\p{M}\p{N}]+|'",
        sentence_boundaries: list[str] = [".", "!", "?"],
        language: str = "french",
        language_model: str | None = None,
        modernize: bool = False,
        strip_tags: bool = False,
        is_philo_db: bool = False,
        text_object_type: str = "doc",
        workers: int | None = None,
        stemmer: bool = False,
        lemmatizer: str | bool = False,
        ngrams: int | bool = False,
        ngram_gap: int = 0,
        ngram_word_order: bool = True,
        stopwords: str | bool = False,
        strip_punctuation: bool = True,
        strip_numbers: bool = True,
        lowercase: bool = True,
        min_word_length: int = 2,
        ascii: bool = False,
        convert_entities: bool = False,
        pos_to_keep: list[str] | bool = False,
        ents_to_keep: list[str] | bool = False,
        post_processing_function: Callable | None = None,
        **_,  # this is meant to make the constructor accept invalid keywords
    ):
        self.normalize_options = {
            "language": language,
            "stemmer": stemmer,
            "stopwords": stopwords or False,  # None is not allowable so set to False
            "strip_punctuation": strip_punctuation,
            "strip_numbers": strip_numbers,
            "lowercase": lowercase,
            "ascii": ascii,
            "convert_entities": convert_entities,
            "min_word_length": min_word_length,
            "lemmatizer": lemmatizer,
            "pos_to_keep": pos_to_keep or [],
            "ents_to_keep": ents_to_keep or [],
        }
        self.language = language
        self.language_model = language_model
        ngrams = ngrams or 0
        if ngrams:
            self.ngram_config = {
                "ngram_size": ngrams,
                "ngram_window": ngrams + ngram_gap,
                "ngram_word_order": ngram_word_order,
            }
        else:
            self.ngram_config = None
        self.post_func = post_processing_function
        if workers is None:
            cpu_count = os.cpu_count() or 2
            self.workers = cpu_count - 1
        else:
            self.workers = workers
        if self.normalize_options["pos_to_keep"] or self.normalize_options["ents_to_keep"] or lemmatizer == "spacy":
            self.do_nlp = True
        else:
            self.do_nlp = False
        if self.do_nlp is False:
            using_gpu = False
        else:
            using_gpu = spacy.prefer_gpu()
        if using_gpu is True:
            self.workers = 1
        self.text_fetcher_args = {
            "word_regex": word_regex,
            "sentence_boundaries": sentence_boundaries,
            "language": language,
            "modernize": modernize,
            "strip_tags": strip_tags,
            "is_philo_db": is_philo_db,
            "text_object_type": text_object_type,
            "workers": self.workers,
            "ngram_config": self.ngram_config,
        }

    def __process_batch(self, batch, keep_all, progress_info):
        queue = mp.Queue()
        process = mp.Process(
            target=process_batch_texts,
            args=(
                queue,
                self.text_fetcher_args,
                batch,
                self.language_model,
                self.normalize_options,
                self.do_nlp,
                keep_all,
                progress_info,
            ),
        )
        process.start()

        while True:
            tokens = queue.get()  # This blocks until data is available
            if tokens is None:  # End signal
                break
            if self.ngram_config is not None:
                tokens = generate_ngrams(**self.ngram_config, tokens=tokens)
            if self.post_func is not None:
                tokens = self.post_func(tokens)
            yield tokens

        process.join()

    def process_texts(
        self,
        texts: Iterable[str],
        keep_all=False,
        progress: bool = True,
        progress_prefix="Processing texts...",
    ) -> Iterable[Tokens]:
        """Process all documents. Returns an iterator of documents"""
        progress_info = {"count": 0, "doc_count": 0, "progress": progress, "progress_prefix": progress_prefix}
        current_batch = []
        if progress is True:
            if self.text_fetcher_args["text_object_type"] == "doc":
                print(f"\r{progress_prefix} 0 documents processed...", end="", flush=True)
            else:
                print(f"\r{progress_prefix} 0 text chunks of 0 documents processed...", end="", flush=True)
        for text in texts:
            current_batch.append(text)
            if len(current_batch) >= 100:
                yield from self.__process_batch(current_batch, keep_all, progress_info)
                progress_info["count"] += 1
                current_batch = []
                progress_info["doc_count"] += 100

        # Process the remaining texts
        if current_batch:
            yield from self.__process_batch(current_batch, keep_all, progress_info)

    def process_string(self, text: str, keep_all: bool = True) -> Tokens:
        """Take a string and return a list of preprocessed tokens"""
        mp.set_start_method("spawn")
        with mp.Pool(1) as pool:
            for tokens in pool.apply(
                process_batch_texts,
                (
                    self.text_fetcher_args,
                    [text],
                    self.language_model,
                    self.normalize_options,
                    self.do_nlp,
                    keep_all,
                    self.ngram_config,
                    self.post_func,
                ),
            ):
                output_tokens = Tokens(tokens, keep_all=keep_all)
                break
        return output_tokens


class TextFetcher:
    """Text fetcher"""

    word_regex: str = r"[\p{L}\p{M}\p{N}]+|'"
    sentence_boundaries: list[str] = [".", "!", "?"]
    language: str = "french"
    modernize: bool | Callable = lambda x: x
    text_object_type: str = "doc"
    workers: int | None = None
    token_regex: re.Pattern = re.compile(r"")
    strip_tags: bool = False
    model: Language | Callable = lambda x: x
    ngram_config: dict[str, Any] | None = None

    @classmethod
    def __init__(
        cls,
        model: Language,
        word_regex=r"[\p{L}\p{M}\p{N}]+|'",
        sentence_boundaries=[".", "!", "?"],
        language="french",
        modernize=False,
        strip_tags=False,
        is_philo_db=False,
        text_object_type="doc",
        ngram_config=None,
        workers=None,
        **_,  # this is meant to make the constructor accept invalid keywords
    ):
        cls.language = language
        cls.model = model
        if modernize is True:
            cls.modernize = Modernizer(language)
        else:
            cls.modernize = False
        cls.strip_tags = strip_tags
        cls.is_philo_db = is_philo_db
        cls.text_object_type = text_object_type
        cls.token_regex = re.compile(rf"({word_regex})|([{''.join(sentence_boundaries)}])")
        cls.sentence_boundaries = sentence_boundaries
        if workers is None:
            cpu_count = os.cpu_count() or 2
            cls.workers = cpu_count - 1
        else:
            cls.workers = workers
        cls.ngram_config = ngram_config

    @classmethod
    def __call__(
        cls,
        texts: Iterable[str],
        do_nlp: bool = False,
        keep_all=False,
        progress: bool = True,
        post_func: Callable | None = None,
    ) -> Iterable[tuple[PreparedDoc | Tokens, int]]:
        """Process all documents. Returns an iterator of documents"""
        doc_count: int = 0
        if progress is True:
            print("Extractings texts...", end="", flush=True)

        with Pool(cls.workers) as pool:
            for processed_docs in pool.imap_unordered(
                cls.__local_process, ((text, do_nlp, keep_all, post_func) for text in texts)
            ):
                doc_count += 1
                for doc in processed_docs:
                    yield doc, doc_count

    @classmethod
    def __local_process(cls, args) -> Iterable[PreparedDoc | Tokens]:
        text, do_nlp, keep_all, post_func = args
        if cls.is_philo_db is True:
            text_objects, sent_starts_list, metadata = cls.process_philo_text(text)
        else:
            text_objects, sent_starts_list, metadata = cls.process_text(text)
        docs = cls.__prepare_docs(text_objects, sent_starts_list, metadata)
        if do_nlp is True:
            return docs
        spacy_docs = (make_spacy_doc(cls.model, doc) for doc in docs)
        tokens_list: list[Tokens] = []
        for spacy_doc in spacy_docs:
            tokens = Tokens(cls.model(spacy_doc), keep_all=keep_all)
            if cls.ngram_config is not None:
                tokens = generate_ngrams(**cls.ngram_config, tokens=tokens)
            if post_func is not None:
                tokens = post_func(tokens)
            tokens_list.append(tokens)
        return tokens_list

    @classmethod
    def __prepare_docs(cls, text_objects, sent_starts_list, metadata) -> list[PreparedDoc]:
        """Prepare doc for creating Spacy doc"""
        list_doc_words: list[PreparedDoc] = []
        for processed_doc, sent_starts, local_metadata in zip(text_objects, sent_starts_list, metadata):
            words = []
            exts = []
            char_num = 0
            for word, ext in processed_doc:
                char_num += len(word)
                words.append(word)
                exts.append(ext)
            list_doc_words.append(PreparedDoc(words, sent_starts, local_metadata, exts, char_num))
        return list_doc_words

    @classmethod
    def process_text(cls, text: str):
        """Process one document. Return the transformed document"""
        with open(text, encoding="utf-8") as input_text:
            doc: str = input_text.read()
        tokens, sent_starts = cls.tokenize_text(doc)
        metadata: dict[str, Any] = {"filename": text}
        return tokens, sent_starts, metadata

    @classmethod
    def process_string(cls, text: str) -> Doc:
        """Process one string. Return the transformed document"""
        tokens, sent_starts = cls.tokenize_text(text)
        doc = Doc(cls.model.vocab, [word for word, _ in tokens], sent_starts=sent_starts)  # type: ignore
        for pos, (_, ext) in enumerate(tokens):
            doc[pos]._.ext = ext
        return doc

    @classmethod
    def tokenize_text(cls, doc: str) -> tuple[list[tuple[str, dict[str, str]]], list[bool]]:
        """Tokenize text"""
        if cls.strip_tags is True:
            doc = remove_tags(doc)
        tokens: list[tuple[str, dict[str, str]]] = []
        sent_starts: list[bool] = []
        new_sent: bool = False
        for match in cls.token_regex.finditer(doc):
            sent_starts.append(new_sent)
            new_sent = match[0] in cls.sentence_boundaries
            if cls.modernize is not False:
                tokens.append((cls.modernize(match[0]), {"token": match[0]}))  # type: ignore
            else:
                tokens.append((match[0], {"token": match[0]}))
        return tokens, sent_starts

    @classmethod
    def process_philo_text(cls, text: str, fetch_metadata: bool = True):
        """Process files produced by PhiloLogic parser"""
        docs: list = []
        current_object_id: str = ""
        current_text_object: list[tuple[str, dict[str, Any]]] = []
        text_path: str = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "TEXT"))
        db_path: str = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "toms.db"))
        if os.path.exists(db_path) is False:
            fetch_metadata = False
        else:
            db: sqlite3.Connection = sqlite3.connect(db_path)
            db.row_factory = sqlite3.Row
            cursor: sqlite3.Cursor = db.cursor()
        metadata_cache: DefaultDict[str, dict[str, Any]] = defaultdict(dict)
        metadata: list = []
        if text.endswith(".lz4"):
            open_file = lz4.frame.open
        else:
            open_file = open
        sent_id: str | None = None
        sent_starts_list: list[list[bool]] = []
        sent_starts: list[bool] = []
        current_sent_id: str = ""
        word_count = 0
        with open_file(text) as philo_db_text:
            for line in philo_db_text:
                word_obj: dict[str, Any] = orjson.loads(line.strip())
                if (
                    word_obj["philo_type"] == "punct" and current_sent_id
                ):  # workaround for bug in Philo4 parser where punctuation is assigned to wrong sentence
                    philo_id = current_sent_id.split()
                    word_obj["position"] = current_sent_id + " 0"
                else:
                    philo_id = word_obj["position"].split()
                object_id = " ".join(philo_id[: PHILO_TEXT_OBJECT_TYPE[cls.text_object_type]])
                current_sent_id = " ".join(philo_id[: PHILO_TEXT_OBJECT_TYPE["sent"]])
                if current_object_id == "":
                    current_object_id = object_id
                if object_id != current_object_id:
                    if current_text_object:
                        if fetch_metadata is True:
                            obj_metadata, metadata_cache = recursive_search(
                                cursor,
                                current_object_id,
                                cls.text_object_type,
                                metadata_cache,
                                text_path,
                                text,
                            )
                            if cls.text_object_type == "sent":
                                obj_metadata = {
                                    **obj_metadata,
                                    "philo_id": " ".join(current_text_object[0][1]["position"].split()[:6] + ["0"]),
                                    "philo_type": "sent",
                                    "start_byte": current_text_object[0][1]["start_byte"],
                                    "end_byte": current_text_object[-1][1]["end_byte"],
                                    "word_count": word_count,
                                }
                            metadata.append(obj_metadata)
                        else:
                            metadata.append(os.path.basename(text))
                        docs.append(current_text_object)
                        sent_starts_list.append(sent_starts)
                        sent_starts = []
                        current_text_object = []
                        word_count = 0
                    current_object_id = object_id
                if current_sent_id == sent_id:
                    sent_starts.append(False)
                else:
                    sent_starts.append(True)
                if cls.modernize is not False:
                    current_text_object.append((cls.modernize(word_obj["token"]), word_obj))  # type: ignore
                    word_count += 1
                else:
                    current_text_object.append((word_obj["token"], word_obj))
                    word_count += 1
                sent_id = " ".join(philo_id[: PHILO_TEXT_OBJECT_TYPE["sent"]])
        if current_text_object:
            if fetch_metadata is True:
                obj_metadata, _ = recursive_search(
                    cursor,
                    current_object_id,
                    cls.text_object_type,
                    metadata_cache,
                    text_path,
                    text,
                )
                if cls.text_object_type == "sent":
                    obj_metadata["philo_id"] = " ".join(current_text_object[0][1]["position"].split()[:6] + ["0"])
                    obj_metadata["philo_type"] = "sent"
                    obj_metadata["start_byte"] = current_text_object[0][1]["start_byte"]
                    obj_metadata["end_byte"] = current_text_object[-1][1]["end_byte"]
                metadata.append(obj_metadata)
            docs.append(current_text_object)
            sent_starts_list.append(sent_starts)
            sent_starts = []
        if fetch_metadata is False:
            metadata = [{"filename": os.path.basename(text)}]  # Return empty shell matching the number of docs returned
        return docs, sent_starts_list, metadata


def generate_ngrams(ngram_size: int, ngram_window: int, ngram_word_order: bool, tokens: Tokens) -> Tokens:
    """Generate ngrams from tokens"""
    ngrams: list[PreprocessorToken] = []
    ngram: Deque[PreprocessorToken] = deque()
    ngram_text: str
    tokens.purge()  # remove empty tokens
    for token in tokens:
        ngram.append(token)
        if len(ngram) == ngram_window:
            for local_ngram in combinations(ngram, ngram_size):
                ext: dict[str, Any] = local_ngram[0].ext
                if ngram_word_order is True:
                    ngram_text = "_".join(t.text for t in local_ngram)
                    ext["end_byte"] = local_ngram[-1].ext["end_byte"]
                else:
                    ngram_text = "_".join(t.text for t in sorted(local_ngram, key=lambda x: x.text))
                    ngram_sorted_position = sorted(local_ngram, key=lambda x: x.ext["start_byte"])
                    ext["start_byte"] = ngram_sorted_position[0].ext["start_byte"]
                    ext["end_byte"] = ngram_sorted_position[-1].ext["end_byte"]
                ngrams.append(PreprocessorToken(ngram_text, "", "", ext))
            ngram.popleft()
    return Tokens(ngrams, tokens.metadata)


def remove_tags(text: str) -> str:
    """Strip XML tags"""
    end_header_index: int = text.rfind("</teiHeader>")
    if end_header_index != -1:
        end_header_index += 12
        text = text[end_header_index:]
    text = TAGS.sub("", text)
    return text


def recursive_search(
    cursor: sqlite3.Cursor,
    position: str,
    object_type: str,
    metadata_cache: DefaultDict[str, dict[str, Any]],
    text_path: str,
    text: str,
) -> tuple[dict[str, Any], DefaultDict[str, dict[str, Any]]]:
    """Recursive look-up of PhiloLogic objects"""
    object_id = position.split()
    object_level = PHILO_TEXT_OBJECT_TYPE[object_type]
    obj_metadata: dict[str, Any] = {"parsed_filename": text}
    while object_id:
        current_id = f"{' '.join(object_id[:object_level])} {' '.join('0' for _ in range(7 - object_level))}"
        if current_id in metadata_cache:
            result = metadata_cache[current_id]
        else:
            cursor.execute("SELECT * from toms WHERE philo_id = ?", (current_id,))
            result = cursor.fetchone()
        if result is not None:
            for field in result.keys():
                if field not in obj_metadata:
                    if result[field] or object_level == 1:  # make sure we get all metadata stored at the last level
                        if field == "filename":
                            obj_metadata[field] = os.path.join(text_path, result[field])
                        else:
                            if field not in obj_metadata or obj_metadata[field]:
                                obj_metadata[field] = result[field]
                            if obj_metadata[field] is None:
                                obj_metadata[field] = ""
                        metadata_cache[current_id][field] = obj_metadata[field]
            current_philo_type = PHILO_OBJECT_LEVEL[object_level]
            philo_object_id = f"philo_{current_philo_type}_id"
            if philo_object_id not in obj_metadata or not obj_metadata[philo_object_id]:
                philo_object_id_value = " ".join(object_id[:object_level])
                obj_metadata[philo_object_id] = philo_object_id_value
                metadata_cache[current_id][philo_object_id] = philo_object_id_value
        object_id.pop()
        object_level -= 1
    return obj_metadata, metadata_cache


def make_spacy_doc(model: Language, prepared_doc: PreparedDoc) -> Doc:
    """Make Spacy doc"""
    doc = Doc(model.vocab, prepared_doc.words, sent_starts=prepared_doc.sent_starts)
    doc._.metadata = prepared_doc.metadata
    doc._.char_num = prepared_doc.char_num
    for pos, ext in enumerate(prepared_doc.exts):
        doc[pos]._.ext = ext
    return doc


def main():
    """Performance testing"""
    import timeit
    from math import floor

    word_num = 0
    for file in os.scandir(sys.argv[1]):
        word_num += len(open(file.path, encoding="utf-8").readlines())
    preproc = PreProcessor()
    start_time = timeit.default_timer()
    _ = [f for f in preproc.process_texts((i.path for i in os.scandir(sys.argv[1])))]
    elapsed = timeit.default_timer() - start_time
    print("\n", elapsed, floor(word_num / elapsed), "words processed per second")


if __name__ == "__main__":
    main()

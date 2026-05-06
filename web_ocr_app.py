from __future__ import annotations

import base64
import difflib
import importlib
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from infer_sentence import (
    load_model,
    preprocess_image,
    recognize_binary_text,
)


CHECKPOINT_PATH = Path("emnist_balanced_cnn.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Slownik EN+PL: korekta slow po OCR (preferowane z biblioteki wordfreq).
class MultilingualDictionaryCorrector:
    def __init__(
        self,
        languages: Tuple[str, ...] = ("en", "pl"),
        max_words_per_language: int = 50000,
    ) -> None:
        self.enabled = False
        self.languages: Tuple[str, ...] = tuple(
            dict.fromkeys(lang.strip().lower() for lang in languages if lang and lang.strip())
        ) or ("en",)
        self.active_languages: Tuple[str, ...] = ()
        self.words: List[str] = []
        self.words_by_len: Dict[int, List[str]] = {}
        self.word_set: set[str] = set()
        self.error: str | None = None
        self.source = "none"
        self._zipf_frequency = None
        self._ocr_char_options: Dict[str, List[str]] = {
            "0": ["o"],
            "1": ["l", "i"],
            "2": ["z"],
            "3": ["e"],
            "4": ["a"],
            "5": ["s"],
            "6": ["b", "g"],
            "7": ["t"],
            "8": ["b", "a"],
            "9": ["g", "q"],
        }

        self._initialize_words(max_words_per_language)

    @property
    def language_label(self) -> str:
        langs = self.active_languages or self.languages
        return "+".join(lang.upper() for lang in langs)

    def _initialize_words(self, max_words_per_language: int) -> None:
        wordfreq_errors: Dict[str, str] = {}

        try:
            words, active_languages, load_errors, zipf_frequency = self._load_words_from_wordfreq(
                max_words_per_language
            )
            if words:
                self._zipf_frequency = zipf_frequency
                self.source = "wordfreq"
                self._apply_loaded_words(words, active_languages)
                return
            wordfreq_errors = load_errors or {"wordfreq": "wordfreq nie zwrocil zadnych slow"}
        except Exception as exc:
            wordfreq_errors = {"wordfreq": str(exc)}

        if wordfreq_errors:
            details = "; ".join(
                f"{lang}: {details}" for lang, details in sorted(wordfreq_errors.items())
            )
            self.error = (
                "Korekta slownikowa jest niedostepna dla wybranych jezykow "
                f"({self.language_label}). Biblioteka wordfreq jest wymagana. Szczegoly: {details}"
            )
        else:
            self.error = (
                "Korekta slownikowa jest niedostepna. Brak poprawnych slow "
                f"dla jezykow: {self.language_label}."
            )

    def _apply_loaded_words(self, words: List[str], active_languages: Tuple[str, ...]) -> None:
        self.words = words
        self.words_by_len = {}
        for word in self.words:
            self.words_by_len.setdefault(len(word), []).append(word)
        self.word_set = set(self.words)
        self.active_languages = active_languages
        self.enabled = bool(self.words)

    def _load_words_from_wordfreq(
        self, max_words_per_language: int
    ) -> Tuple[List[str], Tuple[str, ...], Dict[str, str], object | None]:
        wordfreq_module = importlib.import_module("wordfreq")
        top_n_list = wordfreq_module.top_n_list
        zipf_frequency = getattr(wordfreq_module, "zipf_frequency", None)

        active_languages: List[str] = []
        merged_words: List[str] = []
        seen_words: set[str] = set()
        load_errors: Dict[str, str] = {}

        for language in self.languages:
            try:
                added_count = 0
                raw_words = top_n_list(language, max_words_per_language)
                for raw_word in raw_words:
                    if not isinstance(raw_word, str):
                        continue
                    word = raw_word.strip().lower()
                    if not word.isalpha() or len(word) < 2 or len(word) > 24:
                        continue
                    if word in seen_words:
                        continue
                    seen_words.add(word)
                    merged_words.append(word)
                    added_count += 1
                if added_count > 0:
                    active_languages.append(language)
                else:
                    load_errors[language] = "wordfreq zwrocil pusty slownik"
            except Exception as exc:
                load_errors[language] = str(exc)

        return merged_words, tuple(active_languages), load_errors, zipf_frequency

    @staticmethod
    def _levenshtein_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        if len(a) > len(b):
            a, b = b, a

        previous = list(range(len(b) + 1))
        for i, a_char in enumerate(a, start=1):
            current = [i]
            for j, b_char in enumerate(b, start=1):
                insertion = current[j - 1] + 1
                deletion = previous[j] + 1
                substitution = previous[j - 1] + (a_char != b_char)
                current.append(min(insertion, deletion, substitution))
            previous = current
        return previous[-1]

    def _candidate_score(self, observed_word: str, candidate: str) -> Tuple[float, int, int, float, float]:
        ratio = difflib.SequenceMatcher(None, observed_word, candidate).ratio()
        distance = self._levenshtein_distance(observed_word, candidate)

        # OCR czesto dobrze lapie poczatek/koniec slowa - lekki bonus.
        edge_bonus = 0.0
        edge_matches = 0
        if observed_word and candidate:
            if observed_word[0] == candidate[0]:
                edge_bonus += 0.05
                edge_matches += 1
            if observed_word[-1] == candidate[-1]:
                edge_bonus += 0.03
                edge_matches += 1

        length_penalty = abs(len(observed_word) - len(candidate)) * 0.02
        primary_score = ratio + edge_bonus - length_penalty

        zipf = 0.0
        if self._zipf_frequency is not None:
            for language in self.active_languages or self.languages:
                try:
                    zipf = max(zipf, float(self._zipf_frequency(candidate, language)))
                except Exception:
                    continue

        return primary_score, -distance, edge_matches, zipf, ratio

    @staticmethod
    def _looks_like_spaced_letters(text: str) -> bool:
        normalized = text.strip()
        # Przyklad: "h q L r" -> jedna sekwencja znakow alfanumerycznych rozdzielonych spacjami.
        parts = re.split(r"\s+", normalized)
        return len(parts) >= 3 and all(len(part) == 1 and part.isalnum() for part in parts)

    @staticmethod
    def _split_text_segments(text: str) -> List[str]:
        if not text:
            return []

        segments = [text[0]]
        for char in text[1:]:
            previous_char = segments[-1][-1]
            if char.isalnum() == previous_char.isalnum():
                segments[-1] += char
            else:
                segments.append(char)
        return segments

    def _generate_ocr_variants(self, token: str, max_variants: int = 12) -> List[str]:
        variants = [""]
        for char in token:
            options = self._ocr_char_options.get(char, [char])
            next_variants: List[str] = []
            for prefix in variants:
                for option in options:
                    next_variants.append(prefix + option)
                    if len(next_variants) >= max_variants:
                        break
                if len(next_variants) >= max_variants:
                    break
            variants = next_variants or variants

        unique_variants: List[str] = []
        seen: set[str] = set()
        for variant in variants:
            if variant in seen:
                continue
            seen.add(variant)
            unique_variants.append(variant)
        return unique_variants or [token]

    def _correct_word(
        self,
        word: str,
        forced_length: int | None = None,
        aggressive: bool = False,
    ) -> Tuple[str, float]:
        lowered = word.lower()
        if len(lowered) < 3:
            return lowered, 1.0

        if lowered in self.word_set:
            return lowered, 1.0

        candidate_pools: List[List[str]] = []
        if forced_length is not None:
            # Kluczowa regula: np. 5 wykrytych liter -> tylko kandydaci 5-literowi.
            same_length = self.words_by_len.get(forced_length, [])
            if same_length:
                candidate_pools.append(same_length)
            if not candidate_pools:
                # Przy wymuszonej dlugosci nie probujemy wszystkich dlugosci.
                return lowered, 0.0
        else:
            base_len = len(lowered)
            for maybe_len in (base_len - 2, base_len - 1, base_len, base_len + 1, base_len + 2):
                if maybe_len < 2:
                    continue
                words_for_len = self.words_by_len.get(maybe_len, [])
                if words_for_len:
                    candidate_pools.append(words_for_len)

        if not candidate_pools:
            candidate_pools.append(self.words)

        candidates: List[str] = []
        seen_candidates: set[str] = set()
        for pool in candidate_pools:
            pool_matches = difflib.get_close_matches(lowered, pool, n=30, cutoff=0.55)
            for candidate in pool_matches:
                if candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)
                candidates.append(candidate)

        if not candidates:
            return lowered, 0.0

        scored = []
        for candidate in candidates:
            score, neg_distance, edge_matches, zipf, ratio = self._candidate_score(lowered, candidate)
            scored.append((score, neg_distance, edge_matches, zipf, ratio, candidate))

        scored.sort(reverse=True)
        _, best_neg_distance, _, _, best_ratio, best_candidate = scored[0]
        best_distance = -best_neg_distance

        if aggressive and forced_length is not None:
            corrected = best_candidate
            return corrected, best_ratio

        # Conservative threshold to avoid over-correcting valid uncommon words.
        min_ratio = 0.72 if forced_length is None else 0.60
        max_distance = 2 if len(lowered) <= 5 else max(2, len(lowered) // 3)
        if best_ratio < min_ratio or best_distance > max_distance:
            return lowered, best_ratio

        corrected = best_candidate
        return corrected, best_ratio

    def correct_text(
        self,
        text: str,
        detected_char_count: int | None = None,
    ) -> Tuple[str, List[Dict[str, float | str]]]:
        if not self.enabled or not text:
            return text, []

        aggressive_mode = False
        if self._looks_like_spaced_letters(text):
            text = re.sub(r"\s+", "", text)
            aggressive_mode = True
            if detected_char_count is None:
                detected_char_count = len(text)

        # 1. Szybka sciezka: sprawdzenie wariantow OCR dla calego tokenu bez spacji.
        text_no_spaces = re.sub(r"\s+", "", text)
        no_space_variants = self._generate_ocr_variants(text_no_spaces)
        exact_variant = next(
            (variant for variant in no_space_variants if len(variant) >= 3 and variant.lower() in self.word_set),
            None,
        )
        if exact_variant is not None:
            corr = exact_variant.lower()
            if corr.lower() != text_no_spaces.lower():
                return corr, [{"from": text, "to": corr, "similarity": 1.0}]
            return corr, []

        # 2. Segmentacja uwzgledniajaca cyfry i znaki diakrytyczne.
        segments = self._split_text_segments(text)
        alnum_segments = [seg for seg in segments if seg.isalnum() and any(ch.isalpha() for ch in seg)]

        # Wymuszanie dlugosci ma sens tylko dla pojedynczego odczytanego slowa.
        forced_length: int | None = None
        if len(alnum_segments) == 1 and detected_char_count is not None and 2 <= detected_char_count <= 24:
            # Przekazujemy dokladna liczbe znakow do doboru slow slownikowych.
            forced_length = detected_char_count

        corrected_segments: List[str] = []
        corrections: List[Dict[str, float | str]] = []

        for segment in segments:
            # Korygujemy segmenty, ktore maja litery (samo isalpha lub mix liter i cyfr).
            if segment.isalnum() and any(ch.isalpha() for ch in segment):
                normalized_variants = self._generate_ocr_variants(segment)
                best_corrected = segment.lower()
                best_similarity = 1.0 if best_corrected in self.word_set else -1.0

                for normalized_segment in normalized_variants:
                    corrected, similarity = self._correct_word(
                        normalized_segment,
                        forced_length=forced_length,
                        aggressive=aggressive_mode,
                    )
                    if corrected not in self.word_set:
                        continue
                    if (
                        similarity > best_similarity
                        or (
                            abs(similarity - best_similarity) < 1e-6
                            and len(corrected) > len(best_corrected)
                        )
                    ):
                        best_corrected = corrected
                        best_similarity = similarity

                corrected_segments.append(best_corrected)
                if best_corrected.lower() != segment.lower():
                    corrections.append(
                        {
                            "from": segment,
                            "to": best_corrected,
                            "similarity": round(best_similarity, 3),
                        }
                    )
            else:
                corrected_segments.append(segment)

        return "".join(corrected_segments), corrections

    def contains_all_words(self, text: str) -> bool:
        if not self.enabled or not text:
            return False

        segments = self._split_text_segments(text.lower())
        word_segments = [
            segment for segment in segments if segment.isalnum() and any(ch.isalpha() for ch in segment)
        ]
        if not word_segments:
            return False

        return all(segment in self.word_set for segment in word_segments)

# Model OCR: ladowany raz przy starcie aplikacji.
MODEL = None
MODEL_ERROR = None
# Slowniki: ladowane raz przy starcie aplikacji, wybierane pozniej w UI.
DICTIONARY_CONFIGS: Dict[str, Tuple[str, ...]] = {
    "pl": ("pl",),
    "en": ("en",),
    "pl_en": ("pl", "en"),
}
DEFAULT_DICTIONARY_MODE = "pl_en"
DICTIONARIES = {
    mode: MultilingualDictionaryCorrector(languages=languages, max_words_per_language=50000)
    for mode, languages in DICTIONARY_CONFIGS.items()
}
DICTIONARY = DICTIONARIES[DEFAULT_DICTIONARY_MODE]
try:
    MODEL = load_model(CHECKPOINT_PATH, DEVICE)
except Exception as exc:  # pragma: no cover - runtime initialization safeguard
    MODEL_ERROR = str(exc)

app = Flask(__name__)


def get_dictionary(mode: str) -> MultilingualDictionaryCorrector:
    return DICTIONARIES.get(mode, DICTIONARY)


# Dekodowanie obrazu z canvas (base64 -> OpenCV BGR).
def decode_canvas_image(data_url: str) -> np.ndarray:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Niepoprawny format obrazu z canvas.")

    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Nie udalo sie odczytac obrazu z canvas.")
    return bgr


# Predykcja OCR dla pojedynczego obrazu.
def recognize_text_from_image(img_bgr: np.ndarray) -> Tuple[str, float, int]:
    if MODEL is None:
        raise RuntimeError(f"Model nie jest zaladowany: {MODEL_ERROR}")

    binary = preprocess_image(img_bgr)
    text, boxes, confidences = recognize_binary_text(binary, MODEL, DEVICE)
    if not boxes:
        return "", 0.0, 0

    mean_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
    return text, mean_confidence, len(boxes)


# Widok glowny aplikacji.
def index():
    model_ready = MODEL is not None
    dictionary_summary = " | ".join(
        f"{dictionary.language_label} {len(dictionary.words)} slow"
        for mode, dictionary in DICTIONARIES.items()
        if mode != "pl_en"
    )
    dictionaries_ready = all(dictionary.enabled for dictionary in DICTIONARIES.values())
    dictionary_errors = "; ".join(
        f"{dictionary.language_label}: {dictionary.error}"
        for dictionary in DICTIONARIES.values()
        if not dictionary.enabled and dictionary.error
    )
    return render_template(
        "index.html",
        model_ready=model_ready,
        model_error=MODEL_ERROR,
        checkpoint_path=str(CHECKPOINT_PATH),
        device=str(DEVICE),
        dictionary_enabled=DICTIONARY.enabled,
        dictionary_size=len(DICTIONARY.words),
        dictionary_languages=DICTIONARY.language_label,
        dictionary_source=DICTIONARY.source,
        dictionary_error=DICTIONARY.error,
        dictionaries_ready=dictionaries_ready,
        dictionary_summary=dictionary_summary,
        dictionary_errors=dictionary_errors,
    )


# Endpoint OCR dla obrazu z Paint.
def api_recognize():
    if MODEL is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"Model nie jest gotowy: {MODEL_ERROR}",
                }
            ),
            500,
        )

    payload: Dict[str, str] = request.get_json(silent=True) or {}
    canvas_data = payload.get("image", "")
    ocr_mode = payload.get("mode", "auto")
    dictionary_mode = payload.get("dictionary", DEFAULT_DICTIONARY_MODE)
    if dictionary_mode not in DICTIONARIES:
        dictionary_mode = DEFAULT_DICTIONARY_MODE
    dictionary = get_dictionary(dictionary_mode)

    try:
        img_bgr = decode_canvas_image(canvas_data)
        text, confidence, char_count = recognize_text_from_image(img_bgr)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except Exception:
        return jsonify({"ok": False, "error": "Wewnetrzny blad serwera podczas predykcji."}), 500

    if not text:
        return jsonify(
            {
                "ok": True,
                "text": "",
                "corrected_text": "",
                "char_count": 0,
                "confidence": 0.0,
                "dictionary_enabled": dictionary.enabled,
                "dictionary_applied": False,
                "dictionary_found": False,
                "dictionary_status": "empty",
                "dictionary_message": "Nie wykryto zadnych liter.",
                "dictionary_languages": dictionary.language_label,
                "dictionary_mode": dictionary_mode,
                "corrections": [],
                "message": "Nie wykryto zadnych liter. Napisz grubiej lub wiekszymi literami.",
            }
        )

    # Ignoruj spacje jesli wybrano tryb "1 Slowo"
    if ocr_mode == "word":
        text = text.replace(" ", "")

    # Slownik: korekta tekstu po OCR z uwzglednieniem dlugosci slowa.
    corrected_text, corrections = dictionary.correct_text(text, detected_char_count=char_count)
    dictionary_found = dictionary.contains_all_words(corrected_text)
    dictionary_status = "disabled"
    dictionary_message = "Slownik jest niedostepny."
    if dictionary.enabled:
        if corrections:
            dictionary_status = "corrected"
            dictionary_message = "Znaleziono podobne slowo i zastosowano korekte."
        elif dictionary_found:
            dictionary_status = "found"
            dictionary_message = "Wszystkie rozpoznane slowa znaleziono w wybranym slowniku."
        else:
            dictionary_status = "not_found"
            dictionary_message = (
                "Nie znaleziono tekstu w wybranym slowniku. "
                "Wynik pokazuje tylko odczyt modelu OCR."
            )

    return jsonify(
        {
            "ok": True,
            "text": text,
            "corrected_text": corrected_text,
            "char_count": char_count,
            "confidence": confidence,
            "dictionary_enabled": dictionary.enabled,
            "dictionary_applied": bool(corrections),
            "dictionary_found": dictionary_found,
            "dictionary_status": dictionary_status,
            "dictionary_message": dictionary_message,
            "dictionary_languages": dictionary.language_label,
            "dictionary_mode": dictionary_mode,
            "corrections": corrections,
        }
    )


app.add_url_rule("/", view_func=index, methods=["GET"])
app.add_url_rule("/api/recognize", view_func=api_recognize, methods=["POST"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)

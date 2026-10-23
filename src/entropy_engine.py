"""
Multi-Scale Narrative Entropy Engine
Analyzes lexical entropy compression and boundary deltas in serialized texts.
Requires: python -m spacy download en_core_web_sm
"""

import math
import re
import urllib.request
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import spacy


class NarrativeEntropyEngine:

    def __init__(self, spacy_model="en_core_web_sm"):
        print(f"Loading linguistic model: {spacy_model}...")
        self.nlp = spacy.load(spacy_model, disable=["ner"])

    def fetch_gutenberg_text(self, url: str) -> str:
        """Downloads raw text from Project Gutenberg and strips standard headers."""
        print(f"Fetching text from {url}...")
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Research Pipeline)"}
        )
        with urllib.request.urlopen(req) as response:
            raw_text = response.read().decode("utf-8-sig")

        # Strip Project Gutenberg boilerplate
        start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG", raw_text)
        end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG", raw_text)

        start_idx = start_match.end() if start_match else 0
        end_idx = end_match.start() if end_match else len(raw_text)

        cleaned_text = raw_text[start_idx:end_idx].strip()
        print(f"Extracted {len(cleaned_text):,} characters.")
        return cleaned_text

    def extract_chapters(self, raw_text: str, delimiter_regex: str):
        """Splits raw text into chronological chapters/installments using regex boundaries."""
        splits = re.split(delimiter_regex, raw_text)
        # Filter out empty blocks or short fragments
        chapters = [s.strip() for s in splits if len(s.strip().split()) > 400]
        print(f"Detected {len(chapters)} substantive structural units.")
        return chapters

    def tokenize_and_lemmatize(self, text: str):
        """Tokenizes text, strips digits and excess spaces, and normalizes lemmas."""
        doc = self.nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_punct and not token.is_space and not token.like_num
        ]
        return tokens

    @staticmethod
    def calculate_shannon_entropy(token_seq: list) -> float:
        """Calculates Shannon entropy in bits: H(X) = -sum(p * log2(p))."""
        n = len(token_seq)
        if n == 0:
            return 0.0
        counts = Counter(token_seq)
        entropy = -sum(
            (cnt / n) * math.log2(cnt / n) for cnt in counts.values() if cnt > 0
        )
        return float(entropy)

    def sliding_window_entropy(
        self, tokens: list, window_size: int = 500, step_size: int = 100
    ):
        """Generates continuous entropy trajectories across narrative token space."""
        trajectories = []
        indices = []
        for i in range(0, len(tokens) - window_size + 1, step_size):
            window = tokens[i : i + window_size]
            trajectories.append(self.calculate_shannon_entropy(window))
            indices.append(i + (window_size // 2))
        return np.array(indices), np.array(trajectories)

    def analyze_serial_boundaries(
        self, chapters: list, boundary_window: int = 400
    ):
        """Quantifies the Informational Bottleneck:

        Measures difference between the chapter core and the final installment
        ending.
        """
        results = []
        cumulative_token_count = 0
        boundary_coordinates = []

        for idx, chapter in enumerate(chapters, start=1):
            tokens = self.tokenize_and_lemmatize(chapter)
            if len(tokens) < boundary_window:
                continue

            cumulative_token_count += len(tokens)
            boundary_coordinates.append(cumulative_token_count)

            # Split into baseline interior and concluding boundary
            interior_tokens = tokens[:-boundary_window]
            boundary_tokens = tokens[-boundary_window:]

            h_interior = self.calculate_shannon_entropy(interior_tokens)
            h_boundary = self.calculate_shannon_entropy(boundary_tokens)
            delta_h = h_boundary - h_interior

            results.append({
                "chapter": idx,
                "token_length": len(tokens),
                "h_interior": h_interior,
                "h_boundary": h_boundary,
                "delta_h": delta_h,
            })

        return results, boundary_coordinates

    def plot_trajectory(
        self,
        full_tokens: list,
        boundary_coords: list,
        output_filename="entropy_plot.png",
    ):
        """Renders publication-grade trajectory visualization with boundary markers."""
        x_indices, h_values = self.sliding_window_entropy(
            full_tokens, window_size=600, step_size=150
        )

        plt.figure(figsize=(14, 6), dpi=300)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Plot dynamic trajectory
        plt.plot(
            x_indices,
            h_values,
            color="#1b4965",
            linewidth=1.8,
            label="Shannon Entropy H(W) [Window=600]",
        )

        # Plot rolling mean trendline
        if len(h_values) > 10:
            rolling_avg = np.convolve(
                h_values, np.ones(8) / 8, mode="valid"
            )
            rolling_x = x_indices[len(x_indices) - len(rolling_avg) :]
            plt.plot(
                rolling_x,
                rolling_avg,
                color="#e63946",
                linestyle="--",
                linewidth=1.5,
                label="Smoothed Macro Pacing",
            )

        # Plot installment boundaries
        for idx, coord in enumerate(boundary_coords):
            plt.axvline(
                x=coord,
                color="#6c757d",
                linestyle=":",
                alpha=0.7,
                label="Chapter/Installment Boundary" if idx == 0 else "",
            )

        plt.title(
            "Information-Theoretic Entropy Dynamics Across Serialized Narrative Space",
            fontsize=13,
            pad=15,
            fontweight="bold",
        )
        plt.xlabel("Narrative Progression (Sequential Token Index)", fontsize=11)
        plt.ylabel("Lexical Shannon Entropy (Bits / Token)", fontsize=11)
        plt.legend(loc="lower left", frameon=True)
        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"Figure saved successfully as: {output_filename}")


if __name__ == "__main__":
    # Canonical Case Study: The Hound of the Baskervilles (Serialized in Strand Magazine, 1901)
    GUTENBERG_URL = "https://www.gutenberg.org/files/2852/2852-0.txt"
    CHAPTER_REGEX = r"Chapter\s+[IVXLCDM\d]+\.?"

    engine = NarrativeEntropyEngine()

    raw_novel = engine.fetch_gutenberg_text(GUTENBERG_URL)
    chapters = engine.extract_chapters(raw_novel, CHAPTER_REGEX)

    print("\n--- Running Boundary Entropy Delta Analysis ---")
    metrics, boundaries = engine.analyze_serial_boundaries(
        chapters, boundary_window=350
    )

    print(
        f"{'Chapter':<8}{'Tokens':<10}{'H_Core (bits)':<15}{'H_Boundary (bits)':<18}{'Delta H (bits)':<12}"
    )
    print("-" * 65)
    for m in metrics:
        print(
            f"{m['chapter']:<8}{m['token_length']:<10}{m['h_interior']:<15.3f}{m['h_boundary']:<18.3f}{m['delta_h']:<12.3f}"
        )

    print("\nProcessing complete narrative tokens for visualization...")
    all_tokens = engine.tokenize_and_lemmatize(raw_novel)
    engine.plot_trajectory(all_tokens, boundaries)

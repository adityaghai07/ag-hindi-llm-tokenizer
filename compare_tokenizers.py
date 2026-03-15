"""
Tokenizer fertility comparison across models on a Hindi paragraph.
Saves a bar chart to tokenizer_comparison.png.

Requirements:
    pip install transformers sentencepiece tiktoken matplotlib huggingface_hub
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast
import sentencepiece as spm
from huggingface_hub import hf_hub_download
import tiktoken
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Test paragraph (~200 words, Hindi)
# ---------------------------------------------------------------------------
HINDI_PARA = """
भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ, धर्म और संस्कृतियाँ एक साथ फलती-फूलती हैं।
हिमालय की ऊँची चोटियों से लेकर हिंद महासागर के तटों तक, इस देश की भूगोल अद्वितीय है।
यहाँ की मिट्टी में सदियों का इतिहास समाया हुआ है — सिंधु घाटी सभ्यता से लेकर मुगल साम्राज्य
और ब्रिटिश शासन तक। स्वतंत्रता के बाद भारत ने लोकतंत्र को अपनाया और आज यह विश्व का सबसे
बड़ा लोकतांत्रिक देश है। विज्ञान और प्रौद्योगिकी के क्षेत्र में भारत ने उल्लेखनीय प्रगति की है।
इसरो ने चंद्रयान और मंगलयान जैसे मिशनों से पूरी दुनिया को चौंका दिया है। शिक्षा के क्षेत्र में
आईआईटी और आईआईएम जैसे संस्थान विश्वस्तरीय प्रतिभाएँ तैयार कर रहे हैं। कृषि अभी भी
करोड़ों लोगों की आजीविका का आधार है, जबकि सूचना प्रौद्योगिकी उद्योग ने देश को वैश्विक मंच पर
एक नई पहचान दिलाई है। भारतीय सिनेमा, संगीत और साहित्य ने विश्व भर में अपनी छाप छोड़ी है।
योग और आयुर्वेद आज अंतरराष्ट्रीय स्तर पर मान्यता प्राप्त कर चुके हैं। भारत की युवा पीढ़ी
नवाचार और उद्यमिता में आगे बढ़ रही है और देश को एक नई दिशा दे रही है।
""".strip()


def count_tokens(tokenizer_fn, text):
    """Call tokenizer_fn(text) and return token count."""
    return len(tokenizer_fn(text))


# ---------------------------------------------------------------------------
# Tokenizer loaders
# ---------------------------------------------------------------------------

def load_our_bpe():
    tok = PreTrainedTokenizerFast.from_pretrained("adityaghai07/ag_hindi_bpe_tokenizer_32k")
    return lambda text: tok.encode(text, add_special_tokens=False)


def load_our_unigram():
    model_path = hf_hub_download(
        repo_id="adityaghai07/ag_hindi_uni_tokenizer_32k",
        filename="hindi_unigram.model"
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return lambda text: sp.EncodeAsIds(text)


def load_hf_tokenizer(repo_id):
    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    return lambda text: tok.encode(text, add_special_tokens=False)


def load_tiktoken_gpt4():
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: enc.encode(text)


# ---------------------------------------------------------------------------
# Model list: (display_name, loader_fn)
# ---------------------------------------------------------------------------
MODELS = [
    ("Our BPE (32K)",        load_our_bpe),
    ("Our Unigram (32K)",    load_our_unigram),
    ("Sarvam-1",             lambda: load_hf_tokenizer("sarvamai/sarvam-1")),
    ("Gemma-3 4B",           lambda: load_hf_tokenizer("unsloth/gemma-3-4b-it")),
    ("LLaMA-4 Scout 17B",    lambda: load_hf_tokenizer("unsloth/Llama-4-Scout-17B-16E-Instruct")),
    ("DeepSeek-V3",          lambda: load_hf_tokenizer("unsloth/DeepSeek-V3")),
    ("GPT-4 (cl100k)",       load_tiktoken_gpt4),
    ("SmolDocling 256M",     lambda: load_hf_tokenizer("ds4sd/SmolDocling-256M-preview")),
]


def main():
    names = []
    token_counts = []
    errors = []

    for name, loader in MODELS:
        print(f"Loading {name}...", end=" ", flush=True)
        try:
            tokenizer_fn = loader()
            count = count_tokens(tokenizer_fn, HINDI_PARA)
            names.append(name)
            token_counts.append(count)
            print(f"{count} tokens")
        except Exception as e:
            print(f"FAILED ({e})")
            errors.append(name)

    if not names:
        print("No tokenizers loaded successfully.")
        return

    # Word count for fertility
    words = len(HINDI_PARA.split())
    fertilities = [c / words for c in token_counts]

    # Sort by token count ascending
    paired = sorted(zip(token_counts, fertilities, names), key=lambda x: x[0])
    token_counts_s, fertilities_s, names_s = zip(*paired)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # Color palette: highlight our models in indigo, gradient blues for others
    our_color   = "#4f46e5"   # indigo
    other_palette = sns.color_palette("Blues_r", n_colors=len(names_s))

    colors = []
    other_idx = 0
    for n in names_s:
        if n.startswith("Our"):
            colors.append(our_color)
        else:
            colors.append(other_palette[other_idx])
            other_idx += 1

    bars = ax.barh(
        names_s, token_counts_s,
        color=colors, edgecolor="#ffffff",
        linewidth=1.2, height=0.55
    )

    # Annotate bars
    max_val = max(token_counts_s)
    for bar, count, fert in zip(bars, token_counts_s, fertilities_s):
        ax.text(
            bar.get_width() + max_val * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{count} tokens  ·  {fert:.2f} t/w",
            va="center", ha="left", fontsize=9.5,
            color="#1e293b", fontweight="medium"
        )

    ax.set_xlabel("Token count  (lower = more efficient for Hindi)", fontsize=11, color="#475569")
    ax.set_title(
        f"Tokenizer Fertility Comparison — Hindi Paragraph ({words} words)",
        fontsize=14, fontweight="bold", color="#0f172a", pad=16
    )
    ax.set_xlim(0, max_val * 1.28)
    ax.tick_params(axis="y", labelsize=10.5, colors="#1e293b")
    ax.tick_params(axis="x", colors="#64748b")

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=our_color,        label="Our tokenizers (adityaghai07)"),
        Patch(facecolor=other_palette[2], label="Other models"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=9.5, framealpha=0.85,
        edgecolor="#e2e8f0", facecolor="#ffffff"
    )

    plt.tight_layout()
    out_path = "tokenizer_comparison.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nChart saved to {out_path}")

    if errors:
        print(f"\nFailed to load: {', '.join(errors)}")


if __name__ == "__main__":
    main()

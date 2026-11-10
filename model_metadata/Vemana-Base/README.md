---
license: apache-2.0
language: [te]
library_name: transformers
pipeline_tag: text-generation
tags: [udgamlabs, small-language-model, trained-from-scratch, base-model, telugu, gemma4]
datasets: [ai4bharat/sangraha, HuggingFaceFW/fineweb-2]
---

![Vemana · Udgam Labs](./banner.png)

**UDGAM LABS** · Language model · 278M parameters, Telugu, base

# Vemana-Base

Vemana is a 278M-parameter Telugu language model, trained from scratch on 5.1 billion tokens of public Telugu web text,
read about 2.4 times (12.0 billion token positions), with its own 32,000-token Telugu tokenizer. On Telugu Wikipedia
pages written after the compared models were released, it predicts text better than every other open base model under
1 billion parameters that we tested, including nanochat-telugu-560M, twice its size (0.361 against 0.383 bits per
byte). It is a base model: it continues Telugu text and does not follow instructions. It runs on a laptop-class CPU.

**Try it:** [on your device, in the browser](https://udgamlabs.com/models/vemana#try) · [on Hugging Face Spaces](https://huggingface.co/spaces/udgamlabs/Vemana) · [GGUF for llama.cpp](https://huggingface.co/udgamlabs/Vemana-Base-GGUF) · [every checkpoint](https://huggingface.co/udgamlabs/Vemana-Checkpoints) · [code on GitHub](https://github.com/udgamlabs/vemana) · [udgamlabs.com/models/vemana](https://udgamlabs.com/models/vemana) · [technical report](https://udgamlabs.com/research/vemana)

## Quick start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("udgamlabs/Vemana-Base")
model = AutoModelForCausalLM.from_pretrained("udgamlabs/Vemana-Base")
inputs = tok("తెలుగు భాష", return_tensors="pt")   # the tokenizer adds <s>
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```

Needs transformers 5.19 or newer (Gemma 4 text layout). Keep prompt and output within 1,024 tokens, the length it was
trained at (about 570 Telugu words). On a Mac mini's CPU it writes about 65 tokens per second.

**What it writes** (sampling as above; English translations are ours):

> **వర్షాకాలంలో** సీజనల్ వ్యాధులు ఎక్కువగా వస్తాయి. అందువల్ల ఆరోగ్యంగా ఉండటం చాలా ముఖ్యం. మీరు వర్షాకాలంలో డెంగ్యూ,
> మలేరియా, టైఫాయిడ్, వైరల్ ఇన్ఫెక్షన్ల వంటి సీజనల్ వ్యాధుల బారిన పడే ప్రమాదం ఉంది. …
>
> *In the rainy season, seasonal illnesses are more common, so staying healthy matters a lot. In the rainy season you
> are at risk of seasonal illnesses such as dengue, malaria, typhoid and viral infections. …*

> **కంప్యూటర్ అనేది** మనకు అత్యంత అవసరమైన వస్తువులలో ఒకటి. కంప్యూటర్లు మనం చేసే చాలా పనులను చేస్తాయి, … ఈ సాధనాలలో
> ఒకటి CPU. …
>
> *A computer is one of the things we need most. Computers do many of the tasks we do, … One of these tools is the
> CPU. …*

## Results

**Fresh Telugu text.** Bits per byte (lower is better) on Telugu Wikipedia pages created after the compared models
were released. One scoring script, the same pages and float32 for every model; window 1,024 tokens (Goldfish 512, its
limit); measured by us on 27 October 2026.

| Model | Parameters | Sep-Oct 2026 (416 pages) | Jul-Aug 2026 (477 pages) | 23-26 Oct 2026 (29 pages) |
|---|---|---|---|---|
| **Vemana-Base** | 278M | **0.361** | **0.385** | **0.346** |
| nanochat-telugu-560M | 561M | 0.383 | 0.408 | 0.364 |
| Dhee-Pravega-OG Atom | 646M | 0.393 | 0.414 | 0.371 |
| Pothana Base v2 | 222M (370M with its shared blocks) | 0.440 (0.426 on the words its tokenizer covers) | 0.456 (0.444) | 0.425 (0.410) |
| Goldfish Telugu 1000mb | 125M | 0.458 | 0.480 | 0.435 |
| Gemma-3-1B | 1.0B | 0.484 | 0.502 | 0.441 |
| Qwen3.5-0.8B-Base | 0.87B | 0.584 | | |
| Gemma-3-270m | 268M | 0.585 | | |
| Qwen3-0.6B-Base | 0.6B | 0.692 | | |

- Vemana minus nanochat-telugu-560M (paired page bootstrap, 95% interval): −0.022 [−0.024, −0.021] on the Sep-Oct
  pages, −0.023 [−0.024, −0.022] on Jul-Aug, −0.018 [−0.021, −0.015] on 23-26 Oct. It holds at a 512-token window
  (−0.017) and without the editor who wrote a third of the Sep-Oct pages (−0.023).
- We fixed the test, the win margin (0.02 bits per byte) and the robustness checks before training, and chose the final
  checkpoint in advance.

**Multiple-choice tests** (lm-eval 0.4.13, 0-shot): Belebele-te 25.1%, MILU-te 25.3% (cloze form 30.2%), IndicXNLI-te
35.0%, IndicCOPA-te 58.4%. Chance is 25, 25, 33 and 50% (about 55% on IndicCOPA from answer length alone).

## Where it falls short

- **Larger models are better.** Sarvam-1 (2.5B parameters, about 9× larger) is level with Vemana on the Sep-Oct pages
  (0.356 against 0.361) and better on the Jul-Aug pages (0.365 against 0.385).
- **Knowledge and reasoning tests are at chance,** as for most open models under 1B we tested; Qwen3.5-0.8B does
  better on the letter forms of Belebele (35.6%) and MILU (31.1%). They need a bigger model or more data.
- **It makes things up.** Asked to continue "తెలుగు భాష" (the Telugu language), it wrote fluently that Telugu Language
  Day is celebrated on 21 September; it is 29 August.
- **It loses the thread.** A short story prompt ("ఒక రోజు ఒక రైతు", one day a farmer) turns into the same sentence
  about a farmer and a cow, repeated with small changes.
- **Base model only.** It continues text; it does not answer questions or follow requests, and it has no safety tuning.
- **1,024-token context.**

## How it was trained

- **Data:** public Telugu web text from three open datasets: Sangraha (AI4Bharat, its human-verified Telugu part),
  FineWeb-2 (Hugging Face) and HPLT 3.0. Filtered for Telugu script and length, near-duplicates removed across the
  three, and the Sep-Oct test pages and the multiple-choice test sets removed from the training text. The Jul-Aug and
  late-October test pages were written in 2026, after the training text was collected (web crawls up to 2024). 5.1B
  unique tokens.
- **Run:** 12.0B token positions on one H100 in about 16 hours, about $69 of GPU time, live in public at
  [udgamlabs.com/live/vemana](https://udgamlabs.com/live/vemana).
- **Architecture:** Gemma 4 text layout, 18 layers, width 1,024, 8 query and 2 key-value heads, full attention, tied
  embeddings, our own 32,000-token Telugu tokenizer (byte fallback, so any text can be written).
- **Checkpoints:** every 10% of the run and the final model, in
  [Vemana-Checkpoints](https://huggingface.co/udgamlabs/Vemana-Checkpoints).
- **Technical report:** the data, tokenizer, training run, the test and the rules fixed before it, and every result,
  at [udgamlabs.com/research/vemana](https://udgamlabs.com/research/vemana) (with a PDF).

## Intended use and limitations

A base for research and for building Telugu tools: fine-tuning (chat, summarisation, classification), continued
pretraining, and studies of how a small model learns a language. It can produce false, offensive or harmful text and
should not be used for decisions about people.

## Licence

Apache-2.0. Copyright 2026 Udgam Labs LLP. Training data: Sangraha by AI4Bharat (CC-BY-4.0), FineWeb-2 by Hugging Face
(ODC-By 1.0; use also subject to Common Crawl's terms of use), HPLT 3.0 by the HPLT project (CC0 packaging). Details in
`NOTICE`.

## Citation

```bibtex
@misc{udgamlabs2026vemana,
  title  = {Vemana: a 278M Telugu language model trained from scratch},
  author = {Prasad, Jitesh K},
  year   = {2026},
  howpublished = {Udgam Labs},
  url    = {https://huggingface.co/udgamlabs/Vemana-Base}
}
```

---

**About Udgam Labs.** Udgam Labs is an independent AI research lab. We work from first principles: before we build, we understand why things work, whether we are training a model, fine-tuning one or applying AI to a new problem. Udgam (उद्गम) is Sanskrit for origin. [udgamlabs.com](https://udgamlabs.com)

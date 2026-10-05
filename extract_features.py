"""Feature extraction for the SHL grammar-scoring notebook.

Writes every feature the notebook uses into features/. The GPU steps need two GPUs (we used Kaggle, 2x T4,
internet on, competition data attached, about 8 GPU-hours in total). Install the extra packages first:

    pip install faster-whisper==1.1.1 "nemo_toolkit[asr]" "bitsandbytes>=0.46.1" "mistral-common[audio]" spacy
    python -m spacy download en_core_web_sm
    python extract_features.py                         # all steps
    python extract_features.py --steps text_llm gec    # some steps (they can run in separate sessions)

Steps, in dependency order: waveform, asr, encoders, decoder, audio_llm, text_llm, gec, hand.
decoder, text_llm, gec and hand read the ASR outputs; hand also reads the waveform and gec outputs.
"""
import argparse
import difflib
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

# set in main(); the forked GPU jobs inherit them
DATA = FEAT = KEYS = CLIPS = None
N_CLIPS = None


def reorder(keys):
    pos = {k: i for i, k in enumerate(keys)}
    return np.array([pos[k] for k in KEYS])


def load_audio(path):
    x, sr = sf.read(path, dtype='float32')
    return (x.mean(1) if x.ndim > 1 else x), sr


def run_on_gpus(*jobs):
    """Run each GPU job in its own forked process, in parallel (one per GPU). Models load independently and
    all GPU memory is released when a job ends. The main process itself never initialises CUDA."""
    import multiprocessing as mp
    procs = [mp.get_context('fork').Process(target=fn, args=args) for fn, *args in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [jobs[i][0].__name__ for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f'failed: {failed}')


# waveform statistics, gate statistic and prosody

def audio_stats():
    rows = []
    for key, path in CLIPS:
        x, sr = load_audio(path)
        frames = x[: len(x) // 400 * 400].reshape(-1, 400)
        db = 20 * np.log10(np.sqrt((frames ** 2).mean(1) + 1e-12) + 1e-9)
        split, filename = key.split('/')
        rows.append(dict(split=split, filename=filename, sr=sr, duration=len(x) / sr, rms=float(np.sqrt((x ** 2).mean())),
                         peak=float(np.abs(x).max()), db_max=float(db.max()), db_p50=float(np.median(db)),
                         db_p90=float(np.percentile(db, 90)), voiced_frac=float((db > max(db.max() - 35, -60)).mean()),
                         frac_zero=float((x == 0).mean()), dc=float(x.mean())))
    pd.DataFrame(rows).to_csv(FEAT / 'asr/audio_stats.csv', index=False)


def speech_ratio(path):
    x, _ = load_audio(path)
    frames = x[: len(x) // 400 * 400].reshape(-1, 400)
    return float((np.sqrt((frames ** 2).mean(1)) > 0.02).mean())


def run_lengths(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def prosody(path):
    import librosa
    x, sr = load_audio(path)
    db = 20 * np.log10(librosa.feature.rms(y=x, frame_length=400, hop_length=160)[0] + 1e-6)     # 10 ms hop
    active = db > np.percentile(db, 95) - 35
    f = {'pr_active_ratio': float(active.mean()), 'pr_peak': float(np.abs(x).max()),
         'pr_rms_db': float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)), 'pr_clip_frac': float((np.abs(x) > 0.99).mean()),
         'pr_db_p95': float(np.percentile(db, 95)), 'pr_db_p10': float(np.percentile(db, 10)),
         'pr_snr_proxy': float(np.percentile(db, 95) - np.percentile(db, 10))}
    if active.sum() < 20:
        return f
    a, b = np.flatnonzero(active)[[0, -1]]
    pauses = run_lengths(~active[a:b + 1])
    pauses = pauses[pauses >= 30]
    span_min = (b - a + 1) * 0.01 / 60
    f.update(pr_pauses_per_min=len(pauses) / span_min, pr_long_pauses_per_min=float((pauses >= 100).sum()) / span_min,
             pr_mean_pause=float(pauses.mean() * 0.01) if len(pauses) else 0.0,
             pr_loud_mean=float(db[active].mean()), pr_loud_std=float(db[active].std()),
             pr_voiced_runs_mean=float(run_lengths(active[a:b + 1]).mean() * 0.01))
    f0 = librosa.yin(x, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    n = min(len(f0), len(active))
    f0 = f0[:n][active[:n]]
    semitones = 12 * np.log2(f0 / np.median(f0))
    f.update(pr_f0_median=float(np.median(f0)), pr_pitch_std=float(semitones.std()),
             pr_pitch_range=float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
             pr_pitch_delta=float(np.abs(np.diff(semitones)).mean()))
    f['pr_centroid'] = float(librosa.feature.spectral_centroid(y=x, sr=sr, hop_length=320)[0].mean())
    f['pr_bandwidth'] = float(librosa.feature.spectral_bandwidth(y=x, sr=sr, hop_length=320)[0].mean())
    return f


# speech recognition

WHISPER_PROMPTS = {
    'plain': None,
    'verbatim': "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. So, uh, I I go to the the office and, um, he don't know.",
}
FILLER_PROMPT = 'Um, uh, so, like, I I mean, uh, you know, hmm.'


def preload_cuda_libs():
    """CTranslate2 (faster-whisper) looks for the cuBLAS/cuDNN libraries that pip installs under site-packages/nvidia."""
    import ctypes, site
    libs = [p for sp in site.getsitepackages() for d in ('cublas', 'cudnn') for p in sorted(glob.glob(f'{sp}/nvidia/{d}/lib/*.so*'))]
    for _ in range(2):                      # second pass picks up libraries whose dependencies loaded later
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def segment_dict(s):
    return dict(start=s.start, end=s.end, text=s.text, avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
                compression_ratio=s.compression_ratio,
                words=[(w.word, round(w.start, 3), round(w.end, 3), round(w.probability, 4)) for w in (s.words or [])])


def whisper_v3(variant, gpu):
    preload_cuda_libs()
    from faster_whisper import WhisperModel
    model = WhisperModel('large-v3', device='cuda', device_index=gpu, compute_type='float16')
    out = {}
    for key, path in CLIPS:
        x, _ = load_audio(path)
        segments, info = model.transcribe(x, language='en', task='transcribe', beam_size=5, temperature=0.0,
                                          condition_on_previous_text=False, word_timestamps=True, vad_filter=False,
                                          initial_prompt=WHISPER_PROMPTS[variant], without_timestamps=False)
        segs = [segment_dict(s) for s in segments]
        out[key] = dict(text=''.join(s['text'] for s in segs).strip(), segments=segs, language_prob=info.language_probability)
    json.dump(out, open(FEAT / f'asr/asr_{variant}.json', 'w'))


def whisper_v2(gpu):
    preload_cuda_libs()
    from faster_whisper import WhisperModel
    model = WhisperModel('large-v2', device='cuda', device_index=gpu, compute_type='float16')

    def decode(x, prompt):
        segments, _ = model.transcribe(x, language='en', beam_size=5, temperature=[0.0, 0.2, 0.4, 0.6],
                                       condition_on_previous_text=False, word_timestamps=True, vad_filter=True,
                                       vad_parameters=dict(min_silence_duration_ms=500), initial_prompt=prompt,
                                       hallucination_silence_threshold=2.0)
        return [segment_dict(s) for s in segments]

    out = {}
    for key, path in CLIPS:
        x, _ = load_audio(path)
        segs = decode(x, FILLER_PROMPT)
        text = ''.join(s['text'] for s in segs).strip()
        bad = any(s['compression_ratio'] > 2.4 for s in segs) or 'you know, hmm' in text.lower()
        if bad:
            segs = decode(x, None)
            text = ''.join(s['text'] for s in segs).strip()
        out[key] = dict(text=text, segments=segs, redecoded=bool(bad))
    json.dump(out, open(FEAT / 'asr/asr_wv2.json', 'w'))


def parakeet(gpu):
    import torch
    import nemo.collections.asr as nemo_asr
    torch.cuda.set_device(gpu)
    model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2').cuda().eval()
    keys, paths = zip(*CLIPS)
    out = {}
    for i in range(0, len(paths), 16):
        results = model.transcribe([str(p) for p in paths[i:i + 16]], batch_size=16, timestamps=True, verbose=False)
        for key, r in zip(keys[i:i + 16], results):
            words = r.timestamp.get('word', []) if getattr(r, 'timestamp', None) else []
            out[key] = dict(text=r.text, words=[(w.get('word'), round(float(w.get('start', 0)), 3), round(float(w.get('end', 0)), 3))
                                                for w in words])
    json.dump(out, open(FEAT / 'asr/asr_parakeet.json', 'w'))


# speech encoders

def wavlm(gpu):
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel
    dev = torch.device(f'cuda:{gpu}')
    fe = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-large')
    model = WavLMModel.from_pretrained('microsoft/wavlm-large').to(dev).half().eval()
    means, stds = [], []
    for _, path in CLIPS:
        inputs = fe(load_audio(path)[0], sampling_rate=16000, return_tensors='pt').input_values.to(dev).half()
        with torch.inference_mode():
            H = torch.stack(model(inputs, output_hidden_states=True).hidden_states, 0)[:, 0].float()   # layers x frames x dim
        means.append(H.mean(1).cpu().numpy().astype(np.float16))
        stds.append(H.std(1).cpu().numpy().astype(np.float16))
    np.save(FEAT / 'speech/wavlm_mean.npy', np.stack(means))
    np.save(FEAT / 'speech/wavlm_std.npy', np.stack(stds))


def whisper_encoder(gpu):
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
    dev = torch.device(f'cuda:{gpu}')
    fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
    encoder = WhisperModel.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).encoder.to(dev).eval()
    window = 30 * 16000
    means, stds = [], []
    for _, path in CLIPS:
        x = load_audio(path)[0]
        chunks = [x[s:s + window] for s in range(0, len(x), window)]
        if len(chunks) > 1 and len(chunks[-1]) < 16000:
            chunks = chunks[:-1]
        total = total_sq = None
        n = 0
        for c in chunks:
            feats = fe(c, sampling_rate=16000, return_tensors='pt').input_features.to(dev).half()
            valid = max(1, min(1500, int(np.ceil(len(c) / 320))))       # encoder frames are 20 ms; the rest is padding
            with torch.inference_mode():
                H = torch.stack(encoder(feats, output_hidden_states=True).hidden_states, 0)[:, 0, :valid].float()
            total = H.sum(1) if total is None else total + H.sum(1)
            total_sq = (H ** 2).sum(1) if total_sq is None else total_sq + (H ** 2).sum(1)
            n += valid
        mean = total / n
        means.append(mean.cpu().numpy().astype(np.float16))
        stds.append((total_sq / n - mean ** 2).clamp_min(0).sqrt().cpu().numpy().astype(np.float16))
    np.save(FEAT / 'speech/whisperenc_mean.npy', np.stack(means))
    np.save(FEAT / 'speech/whisperenc_std.npy', np.stack(stds))


# Whisper decoder with teacher forcing

def text_windows(rec, variant):
    windows = {}
    if variant == 'parakeet':
        for word, start, _ in rec['words']:
            windows.setdefault(int(start // 30), []).append(word)
    else:
        for seg in rec['segments']:
            windows.setdefault(int(seg['start'] // 30), []).append(seg['text'].strip())
    return windows


def whisper_decoder(variant, gpu):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    dev = f'cuda:{gpu}'
    proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', dtype=torch.float16).to(dev).eval()
    asr = json.load(open(FEAT / f'asr/asr_{variant}.json'))
    prefix = proc.tokenizer.convert_tokens_to_ids(['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|notimestamps|>'])
    eot = proc.tokenizer.convert_tokens_to_ids('<|endoftext|>')
    states, stats = [], []
    for key, path in CLIPS:
        x = load_audio(path)[0]
        total, logprobs, n = 0, [], 0
        for w, words in sorted(text_windows(asr[key], variant).items()):
            chunk = x[w * 480000:(w + 1) * 480000]
            if len(chunk) < 1600 or not words:
                continue
            feats = proc.feature_extractor(chunk, sampling_rate=16000, return_tensors='pt').input_features.to(dev).half()
            ids = torch.tensor([prefix + proc.tokenizer.encode(' ' + ' '.join(words), add_special_tokens=False)[:440] + [eot]], device=dev)
            with torch.inference_mode():
                out = model(input_features=feats, decoder_input_ids=ids[:, :-1], output_hidden_states=True)
            H = torch.stack(out.decoder_hidden_states, 0)[:, 0, 3:].float()           # drop the first three prompt positions
            lp = torch.log_softmax(out.logits[0, 3:].float(), -1).gather(-1, ids[0, 4:, None])[:, 0]
            total = total + H.sum(1)
            n += H.shape[1]
            logprobs.append(lp.cpu().numpy())
        if n == 0:
            states.append(np.zeros((33, 1280), np.float16))
            stats.append(np.zeros(6, np.float32))
            continue
        lp = np.concatenate(logprobs)
        states.append((total / n).cpu().numpy().astype(np.float16))
        stats.append(np.array([lp.mean(), np.median(lp), np.percentile(lp, 10), (lp < -2).mean(), (lp < -5).mean(), lp.min()], np.float32))
    np.save(FEAT / f'decoder/wdec_{variant}_mean.npy', np.stack(states))
    np.save(FEAT / f'decoder/wdec_{variant}_lp.npy', np.stack(stats))


# audio LLMs

QWEN2_AUDIO_RUBRIC = (
    "Score the speaker's GRAMMAR from 1 to 5. 1: struggles with sentence structure and syntax, limited control of simple structures. "
    "2: limited understanding of sentence structure; consistent basic mistakes; may leave sentences incomplete. "
    "3: decent grasp of sentence structure but makes grammatical errors. 4: strong control of grammar; only occasional minor errors. "
    "5: high grammatical accuracy and adept control of complex grammar. Answer with a single digit from 1 to 5.")
VOXTRAL_RUBRIC = (
    "Score the speaker's GRAMMAR from 1 to 5. 1: struggles with sentence structure and syntax. 2: limited control; consistent basic mistakes; "
    "incomplete sentences. 3: decent grasp but makes grammatical errors. 4: strong control; occasional minor errors. 5: high accuracy and adept "
    "control of complex grammar. Answer with a single digit.")


def qwen2_audio():
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    name = 'Qwen/Qwen2-Audio-7B-Instruct'
    proc = AutoProcessor.from_pretrained(name)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(name, dtype=torch.float16, device_map='auto').eval()
    layers = [8, 14, 18, 22, 26, model.config.text_config.num_hidden_layers]
    audio_id = getattr(model.config, 'audio_token_index', None) or getattr(model.config, 'audio_token_id', None)
    digits = [proc.tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]
    conversation = [{'role': 'user', 'content': [{'type': 'audio', 'audio_url': 'clip.wav'}, {'type': 'text', 'text': QWEN2_AUDIO_RUBRIC}]}]
    prompt = proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    probs, audio_states, last_states = [], [], []
    for _, path in CLIPS:
        x = load_audio(path)[0]
        chunks = [x[k:k + 30 * 16000] for k in range(0, len(x), 30 * 16000)]
        if len(chunks) > 1 and len(chunks[-1]) < 5 * 16000:
            chunks = chunks[:-1]
        weights = np.array([len(c) for c in chunks], float)
        weights /= weights.sum()
        p = a = l = 0
        for chunk, w in zip(chunks, weights):
            inputs = proc(text=prompt, audio=[chunk], sampling_rate=16000, return_tensors='pt', padding=True).to(model.device)
            inputs['input_features'] = inputs['input_features'].half()
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True)
            logits = out.logits[0, -1].float()
            is_audio = inputs['input_ids'][0] == audio_id
            hs = out.hidden_states
            p = p + w * np.r_[torch.softmax(logits[digits], -1).cpu().numpy(), torch.softmax(logits, -1)[digits].sum().item()]
            a = a + w * np.stack([hs[k][0][is_audio.to(hs[k].device)].float().mean(0).cpu().numpy() for k in layers])
            l = l + w * np.stack([hs[k][0, -1].float().cpu().numpy() for k in layers])
            del out
        probs.append(p)
        audio_states.append(a.astype(np.float16))
        last_states.append(l.astype(np.float16))
    np.save(FEAT / 'audio_llm/qaudio_probs.npy', np.stack(probs))
    np.save(FEAT / 'audio_llm/qaudio_audiomean.npy', np.stack(audio_states))
    np.save(FEAT / 'audio_llm/qaudio_last.npy', np.stack(last_states))
    json.dump(dict(keys=[k for k, _ in CLIPS], layers=layers), open(FEAT / 'audio_llm/qaudio_meta.json', 'w'))


def voxtral():
    import torch
    import transformers.processing_utils
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    transformers.processing_utils.ProcessorMixin.__repr__ = lambda self: 'Processor'   # Mistral tokenizer repr raises in transformers 5.0
    name = 'mistralai/Voxtral-Mini-3B-2507'
    proc = AutoProcessor.from_pretrained(name)
    model = VoxtralForConditionalGeneration.from_pretrained(name, dtype=torch.float16, device_map={'': 0}).eval()
    cfg = getattr(model.config, 'text_config', model.config)
    layers = sorted({int(round(cfg.num_hidden_layers * f)) for f in (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)})
    audio_id = getattr(model.config, 'audio_token_id', None) or getattr(model.config, 'audio_token_index', None)
    digits = [proc.tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]
    probs, audio_states, last_states = [], [], []
    for _, path in CLIPS:
        conversation = [{'role': 'user', 'content': [{'type': 'audio', 'path': str(path)}, {'type': 'text', 'text': VOXTRAL_RUBRIC}]}]
        inputs = proc.apply_chat_template(conversation, return_tensors='pt')
        inputs = {k: (v.to('cuda:0').half() if v.dtype == torch.float32 else v.to('cuda:0')) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        is_audio = inputs['input_ids'][0] == audio_id
        hs = out.hidden_states
        probs.append(torch.softmax(out.logits[0, -1].float()[digits], -1).cpu().numpy())
        audio_states.append(np.stack([hs[k][0][is_audio].float().mean(0).cpu().numpy() for k in layers]).astype(np.float16))
        last_states.append(np.stack([hs[k][0, -1].float().cpu().numpy() for k in layers]).astype(np.float16))
        del out
    np.save(FEAT / 'audio_llm/vox_probs.npy', np.stack(probs))
    np.save(FEAT / 'audio_llm/vox_amean.npy', np.stack(audio_states))
    np.save(FEAT / 'audio_llm/vox_last.npy', np.stack(last_states))
    json.dump(dict(keys=[k for k, _ in CLIPS], layers=layers), open(FEAT / 'audio_llm/vox_meta.json', 'w'))


# text LLMs on the transcripts

RUBRIC = """1: The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.
2: The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.
3: The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.
4: The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.
5: Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."""
HEAD = ("Below is an automatic speech-recognition transcript of a person speaking English for up to one minute in response to an "
        "open-ended question. The transcript may contain recognition errors, fillers and repetitions.\n\nTranscript:\n\"\"\"\n{t}\n\"\"\"\n\n")
LLM_PROMPTS = {    # name -> (prompt, answer digits)
    'grammar': (HEAD + "Rate the speaker's GRAMMAR using this rubric:\n" + RUBRIC + "\n\nAnswer with a single digit from 1 to 5.", '12345'),
    'vocab': (HEAD + "Rate the speaker's VOCABULARY range and accuracy from 1 (very limited, basic words, frequent misuse) to 5 (wide, precise, sophisticated vocabulary).\n\nAnswer with a single digit from 1 to 5.", '12345'),
    'coherence': (HEAD + "Rate the FLUENCY and COHERENCE of the speaker's response from 1 (fragmented, hard to follow, many breakdowns) to 5 (smooth, well-organized, easy to follow).\n\nAnswer with a single digit from 1 to 5.", '12345'),
    'cefr': (HEAD + "Estimate the speaker's overall English proficiency on the CEFR scale. Use 1 = A1, 2 = A2, 3 = B1, 4 = B2, 5 = C1, 6 = C2.\n\nAnswer with a single digit from 1 to 6.", '123456'),
    'errors': (HEAD + "How many grammatical errors does the speaker make relative to how much they say? Use 1 = almost none, 2 = few, 3 = some, 4 = many, 5 = very many / almost every sentence.\n\nAnswer with a single digit from 1 to 5.", '12345'),
}


def load_llm(name, load_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(device_map='auto', dtype=torch.float16)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    return tok, AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()


def llm_score(tok, model, texts, prompt, layers, keep_states, batch=4):
    import torch
    digit_ids = [tok.encode(d, add_special_tokens=False)[-1] for d in LLM_PROMPTS[prompt][1]]
    probs, last, mean = [None] * len(texts), [None] * len(texts), [None] * len(texts)
    order = np.argsort([len(t) for t in texts])                 # length-sorted batches
    for b in range(0, len(texts), batch):
        idx = order[b:b + batch]
        strings = []
        for i in idx:
            body = LLM_PROMPTS[prompt][0].format(t=texts[i] if texts[i].strip() else '(no speech detected)')
            messages = [{'role': 'system', 'content': 'You are an expert examiner of spoken English proficiency.'}, {'role': 'user', 'content': body}]
            strings.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        enc = tok(strings, return_tensors='pt', padding=True, return_offsets_mapping=True)
        offsets = enc.pop('offset_mapping')
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=keep_states)
        logits = out.logits[:, -1].float()
        p = torch.softmax(logits[:, digit_ids], -1).cpu().numpy()
        mass = torch.log_softmax(logits, -1)[:, digit_ids].exp().sum(-1).cpu().numpy()
        for j, i in enumerate(idx):
            probs[i] = np.r_[p[j], mass[j]]
            if not keep_states:
                continue
            text = texts[i] if texts[i].strip() else '(no speech detected)'
            c0 = strings[j].find(text)
            o = offsets[j].numpy()
            attn = enc['attention_mask'][j].cpu().numpy().astype(bool)
            sel = attn & (o[:, 0] >= c0) & (o[:, 1] <= c0 + len(text)) & (o[:, 1] > o[:, 0])   # tokens of the transcript
            if sel.sum() == 0:
                sel = attn
            sel = torch.tensor(sel)
            hs = out.hidden_states
            last[i] = np.stack([hs[k][j, -1].float().cpu().numpy() for k in layers]).astype(np.float16)
            mean[i] = np.stack([hs[k][j][sel.to(hs[k].device)].float().mean(0).cpu().numpy() for k in layers]).astype(np.float16)
        del out
    return np.stack(probs), ((np.stack(last), np.stack(mean)) if keep_states else None)


def text_llm(name, variants, load_4bit=False):
    tok, model = load_llm(name, load_4bit)
    n_layers = model.config.num_hidden_layers
    layers = sorted({int(round(n_layers * f)) for f in (0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)})
    tag = name.split('/')[-1]
    for variant in variants:
        asr = json.load(open(FEAT / f'asr/asr_{variant}.json'))
        keys = list(asr)[:N_CLIPS]
        texts = [asr[k]['text'] for k in keys]
        json.dump(keys, open(FEAT / f'text_llm/keys_{variant}.json', 'w'))
        for prompt in LLM_PROMPTS:
            probs, states = llm_score(tok, model, texts, prompt, layers, keep_states=(prompt == 'grammar'))
            np.save(FEAT / f'text_llm/{tag}__{variant}__{prompt}__probs.npy', probs)
            if states is not None:
                np.save(FEAT / f'text_llm/{tag}__{variant}__{prompt}__last.npy', states[0])
                np.save(FEAT / f'text_llm/{tag}__{variant}__{prompt}__mean.npy', states[1])


# grammatical-error correction and LM surprisal

FILLER_RE = re.compile(r"\b(u+h+|u+m+|e+r+m*|h+m+|m+h*m+|a+h+|mm+)\b[,.]?\s*", re.I)
GEC_SYSTEM = 'You are a meticulous English grammar corrector.'
GEC_USER = ("Correct the grammatical errors in the following transcript of spontaneous speech. Make MINIMAL edits: fix only grammar "
            "(verb forms, tense, agreement, articles, prepositions, plurals, word order, missing words, sentence structure). Do not change "
            "word choice or style, do not add content, do not paraphrase. Output only the corrected text.\n\nTranscript:\n{t}")


def clean_disfluencies(text):
    text = FILLER_RE.sub('', text)
    text = re.sub(r"\b(\w+)(\s*,?\s+\1\b)+", r"\1", text, flags=re.I)      # "I I went" -> "I went"
    return re.sub(r"\s+", " ", text).strip()


def surprisal(tok, model, texts):
    import torch
    rows = []
    for t in texts:
        enc = tok(clean_disfluencies(t) or '(silence)', return_tensors='pt', truncation=True, max_length=1024).to(model.device)
        with torch.inference_mode():
            logits = model(**enc).logits[0, :-1]
        target = enc['input_ids'][0, 1:].to(logits.device)
        nll = torch.nn.functional.cross_entropy(logits.float(), target, reduction='none').cpu().numpy()
        rows.append([nll.mean(), np.median(nll), np.percentile(nll, 90), (nll > 6).mean(), (nll > 10).mean()] if len(nll) else [0] * 5)
    return np.array(rows, dtype=np.float32)


def correct_grammar(tok, model, texts, batch=12):
    import torch
    corrected = [None] * len(texts)
    order = np.argsort([len(t) for t in texts])
    for b in range(0, len(texts), batch):
        idx = order[b:b + batch]
        prompts = [tok.apply_chat_template([{'role': 'system', 'content': GEC_SYSTEM},
                                            {'role': 'user', 'content': GEC_USER.format(t=clean_disfluencies(texts[i]) or '(silence)')}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in idx]
        enc = tok(prompts, return_tensors='pt', padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            corrected[i] = tok.decode(generated[j, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return corrected


def gec_and_surprisal():
    tok, model = load_llm('Qwen/Qwen2.5-7B-Instruct')
    for variant in ['verbatim', 'wv2']:
        asr = json.load(open(FEAT / f'asr/asr_{variant}.json'))
        keys = list(asr)[:N_CLIPS]
        texts = [asr[k]['text'] for k in keys]
        np.save(FEAT / f'gec/Qwen2.5-7B-Instruct__{variant}__surprisal.npy', surprisal(tok, model, texts))
        json.dump(dict(keys=keys, clean=[clean_disfluencies(t) for t in texts], corrected=correct_grammar(tok, model, texts)),
                  open(FEAT / f'gec/Qwen2.5-7B-Instruct__{variant}__gec.json', 'w'))


# interpretable transcript features

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
FILLER_WORDS = {'um', 'umm', 'uh', 'uhh', 'er', 'erm', 'ah', 'hmm', 'mm', 'mhm', 'eh'}
SUBORDINATORS = {'because', 'although', 'which', 'who', 'whom', 'whose', 'when', 'while', 'if', 'unless', 'since', 'that',
                 'though', 'whereas', 'whether'}
FUNCTION_WORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did',
                  'to', 'of', 'in', 'on', 'at', 'for', 'with', 'from', 'by', 'about'}


def mattr(words, w=25):
    if len(words) < w:
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i:i + w])) / w for i in range(len(words) - w + 1)]))


def whisper_features(rec, duration):
    segs = rec['segments']
    words = [tuple(w) for s in segs for w in s['words']]                     # (text, start, end, probability)
    flat = [t for w in words for t in WORD.findall(w[0].lower())]
    n = len(flat)
    f = {'n_words': n, 'wpm': n / duration * 60}
    if words:
        t_first, t_last = words[0][1], words[-1][2]
        span = max(t_last - t_first, 1e-3)
        gaps = np.array([words[i + 1][1] - words[i][2] for i in range(len(words) - 1)]) if len(words) > 1 else np.zeros(0)
        probs = np.array([w[3] for w in words])
        speech_time = np.array([w[2] - w[1] for w in words]).sum()
        f.update(span_frac=span / duration, lead_silence=t_first, trail_silence=max(duration - t_last, 0),
                 artic_rate=n / max(speech_time, 1e-3), span_rate=n / span * 60, p_mean=probs.mean(),
                 p_min10=np.percentile(probs, 10), p_low_frac=(probs < 0.5).mean(), p_vlow_frac=(probs < 0.2).mean())
        for thr in (0.3, 0.6, 1.0, 2.0):
            f[f'pause{thr}_per_min'] = (gaps > thr).sum() / span * 60
        long_gaps = gaps[gaps > 0.3]
        f['pause_mean'] = long_gaps.mean() if len(long_gaps) else 0.0
        f['pause_total_frac'] = long_gaps.sum() / span
        runs = np.diff(np.r_[0, np.flatnonzero(gaps > 0.3) + 1, len(words)])
        f['mlr'] = runs.mean()
        f['mlr_max'] = runs.max()
    else:
        for k in ['span_frac', 'lead_silence', 'trail_silence', 'artic_rate', 'span_rate', 'p_mean', 'p_min10', 'p_low_frac',
                  'p_vlow_frac', 'pause0.3_per_min', 'pause0.6_per_min', 'pause1.0_per_min', 'pause2.0_per_min', 'pause_mean',
                  'pause_total_frac', 'mlr', 'mlr_max']:
            f[k] = 0.0
        f['lead_silence'] = duration
        f['trail_silence'] = duration
    f['filler_rate'] = sum(t in FILLER_WORDS for t in flat) / max(n, 1)
    f['rep_rate'] = sum(flat[i] == flat[i + 1] for i in range(n - 1)) / max(n, 1)
    bigrams = list(zip(flat, flat[1:]))
    f['bigram_rep_rate'] = sum(bigrams[i] == bigrams[i + 2] for i in range(len(bigrams) - 2)) / max(n, 1)
    content = [t for t in flat if t not in FILLER_WORDS]
    f['ttr_mattr'] = mattr(content)
    f['mean_wlen'] = np.mean([len(t) for t in content]) if content else 0.0
    f['long_word_frac'] = np.mean([len(t) >= 7 for t in content]) if content else 0.0
    sents = [s for s in re.split(r'[.!?]+', rec['text']) if WORD.search(s.lower())]
    f['n_sents_rate'] = len(sents) / max(n, 1)
    f['words_per_sent'] = n / max(len(sents), 1)
    f['comma_rate'] = rec['text'].count(',') / max(n, 1)
    lp = [s['avg_logprob'] for s in segs]
    ns = [s['no_speech_prob'] for s in segs]
    cr = [s['compression_ratio'] for s in segs]
    f['avg_logprob'] = float(np.mean(lp)) if lp else -3.0
    f['min_logprob'] = float(np.min(lp)) if lp else -3.0
    f['no_speech_max'] = float(np.max(ns)) if ns else 1.0
    f['no_speech_mean'] = float(np.mean(ns)) if ns else 1.0
    f['compress_max'] = float(np.max(cr)) if cr else 0.0
    f['subord_rate'] = sum(t in SUBORDINATORS for t in flat) / max(n, 1)
    f['i_rate'] = sum(t == 'i' for t in flat) / max(n, 1)
    return f


def content_words(text):
    return [w for w in WORD.findall(text.lower()) if w not in FILLER_WORDS]


def wer(ref, hyp):
    if not ref:
        return 1.0 if hyp else 0.0
    d = np.arange(len(hyp) + 1)
    for i, x in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, z in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (x != z))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def cross_asr_features(key, duration, asr):
    rec = asr['parakeet'][key]
    pk, v3p, v2, v3v = (content_words(asr[name][key]['text']) for name in ['parakeet', 'plain', 'wv2', 'verbatim'])
    n = len(pk)
    f = {'pk_wpm': n / duration * 60, 'pk_ttr': mattr(pk), 'pk_wlen': np.mean([len(t) for t in pk]) if n else 0,
         'pk_rep': sum(pk[i] == pk[i + 1] for i in range(n - 1)) / max(n, 1)}
    if len(rec['words']) > 1:
        s, e = np.array([w[1] for w in rec['words']]), np.array([w[2] for w in rec['words']])
        gaps, span = s[1:] - e[:-1], max(e[-1] - s[0], 1e-3)
        f.update(pk_span_rate=len(rec['words']) / span * 60, pk_p03=(gaps > 0.3).sum() / span * 60, pk_p1=(gaps > 1).sum() / span * 60,
                 pk_mlr=np.diff(np.r_[0, np.flatnonzero(gaps > 0.3) + 1, len(rec['words'])]).mean())
    f['pk_words_per_sent'] = n / max(len([s for s in re.split(r'[.!?]+', rec['text']) if WORD.search(s.lower())]), 1)
    f.update(wer_pk_v3p=wer(pk, v3p), wer_pk_v2=wer(pk, v2), wer_v3p_v2=wer(v3p, v2), wer_v3v_pk=wer(v3v, pk),
             len_ratio_pk_v3=len(pk) / max(len(v3p), 1), len_ratio_v2_v3=len(v2) / max(len(v3p), 1))
    keep = ('p_mean', 'p_min10', 'p_low_frac', 'avg_logprob', 'filler_rate', 'rep_rate', 'ttr_mattr', 'no_speech_mean')
    f.update({f'v2_{k}': v for k, v in whisper_features(asr['wv2'][key], duration).items() if k in keep})
    return f


def tree_depth(tok):
    d = 0
    while tok.head.i != tok.i:
        tok, d = tok.head, d + 1
    return d


def syntax_features(text, nlp):
    doc = nlp(FILLER_RE.sub('', text))
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    n = max(len(toks), 1)
    sents = [s for s in doc.sents if any(not t.is_punct for t in s)]
    dep = pd.Series([t.dep_ for t in toks]).value_counts() if toks else pd.Series(dtype=float)
    pos = pd.Series([t.tag_ for t in toks]).value_counts() if toks else pd.Series(dtype=float)
    f = {f'dep_{d}': dep.get(d, 0) / n for d in ['advcl', 'ccomp', 'xcomp', 'relcl', 'acl', 'mark', 'aux', 'auxpass', 'nsubjpass',
                                                 'conj', 'cc', 'det', 'prep', 'amod', 'advmod', 'neg', 'dep', 'intj']}
    f.update({f'pos_{p}': pos.get(p, 0) / n for p in ['VBD', 'VBZ', 'VBP', 'VBG', 'VBN', 'VB', 'MD', 'NN', 'NNS', 'NNP', 'PRP',
                                                       'JJ', 'RB', 'IN', 'DT', 'WDT', 'WP', 'WRB', 'UH', 'CC']})
    f['clauses_per_sent'] = (f['dep_advcl'] + f['dep_ccomp'] + f['dep_xcomp'] + f['dep_relcl'] + f['dep_acl']) * n / max(len(sents), 1)
    f['mean_depth'] = np.mean([tree_depth(t) for t in toks]) if toks else 0
    f['max_depth_mean'] = np.mean([max(tree_depth(t) for t in s) for s in sents]) if sents else 0
    f['sent_len_mean'] = n / max(len(sents), 1)
    f['sent_len_std'] = np.std([len(s) for s in sents]) if sents else 0
    bad = 0
    for t in toks:                                     # crude subject-verb agreement check
        if t.dep_ == 'nsubj' and t.head.tag_ in ('VBZ', 'VBP'):
            singular = t.tag_ in ('NN', 'NNP') or t.lower_ in ('he', 'she', 'it', 'this', 'that')
            plural = t.tag_ in ('NNS', 'NNPS') or t.lower_ in ('we', 'they', 'these', 'those')
            bad += (singular and t.head.tag_ == 'VBP') or (plural and t.head.tag_ == 'VBZ')
    f['agree_err_rate'] = bad / n
    f['verbless_sent_frac'] = np.mean([not any(t.pos_ in ('VERB', 'AUX') for t in s) for s in sents]) if sents else 0
    f['past_present_mix'] = min(f['pos_VBD'], f['pos_VBZ'] + f['pos_VBP'])
    return f


def gec_features(original, corrected):
    a, b = WORD.findall(original.lower()), WORD.findall(corrected.lower())
    n = max(len(a), 1)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    sub = ins = dele = func = edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        edits += 1
        if tag == 'replace':
            sub += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            ins += j2 - j1
        elif tag == 'delete':
            dele += i2 - i1
        func += sum(w in FUNCTION_WORDS for w in a[i1:i2] + b[j1:j2])
    return dict(gec_edit_rate=edits / n * 100, gec_sub=sub / n * 100, gec_ins=ins / n * 100, gec_del=dele / n * 100,
                gec_func=func / n * 100, gec_sim=sm.ratio(), gec_len_ratio=len(b) / n)


def gec_table(variant):
    d = json.load(open(FEAT / f'gec/Qwen2.5-7B-Instruct__{variant}__gec.json'))
    idx = reorder(d['keys'])
    df = pd.DataFrame([gec_features(d['clean'][i], d['corrected'][i] or '') for i in idx]).add_prefix(f'{variant}_')
    nll = np.load(FEAT / f'gec/Qwen2.5-7B-Instruct__{variant}__surprisal.npy')[idx]
    df[[f'{variant}_nll_{s}' for s in ['mean', 'med', 'p90', 'gt6', 'gt10']]] = nll
    return df


def hand_features():
    import spacy
    nlp = spacy.load('en_core_web_sm')
    durations = pd.read_csv(FEAT / 'asr/audio_stats.csv').duration.values
    asr = {v: json.load(open(FEAT / f'asr/asr_{v}.json')) for v in ['verbatim', 'plain', 'wv2', 'parakeet']}
    whisper = pd.DataFrame([whisper_features(asr['verbatim'][k], d) for k, d in zip(KEYS, durations)]).drop(columns='n_words')
    prosody_table = pd.read_csv(FEAT / 'audio/prosody.csv').drop(columns='key')
    cross = pd.DataFrame([cross_asr_features(k, d, asr) for k, d in zip(KEYS, durations)])
    syntax = [pd.DataFrame([syntax_features(asr[v][k]['text'], nlp) for k in KEYS]).add_prefix(p)
              for v, p in [('parakeet', 'pk_'), ('verbatim', 'v3v_')]]
    table = pd.concat([whisper, prosody_table, cross, *syntax, gec_table('verbatim'), gec_table('wv2')], axis=1).fillna(0)
    table.insert(0, 'key', KEYS)
    return table


# steps

def step_waveform():
    """Duration/loudness stats, the speech ratio used for the grade-0 gate, and prosody (CPU)."""
    audio_stats()
    np.save(FEAT / 'audio/speech_ratio.npy', np.array([speech_ratio(p) for _, p in CLIPS]))
    table = pd.DataFrame(Parallel(n_jobs=4)(delayed(prosody)(p) for _, p in CLIPS))
    table.insert(0, 'key', [k for k, _ in CLIPS])
    table.to_csv(FEAT / 'audio/prosody.csv', index=False)


def step_asr():
    """Whisper large-v3 (plain and verbatim prompt), then Parakeet and Whisper large-v2, two models at a time."""
    run_on_gpus((whisper_v3, 'plain', 0), (whisper_v3, 'verbatim', 1))
    run_on_gpus((parakeet, 0), (whisper_v2, 1))


def step_encoders():
    """WavLM-Large and Whisper large-v3 encoder embeddings."""
    run_on_gpus((wavlm, 0), (whisper_encoder, 1))


def step_decoder():
    """Whisper decoder states over the verbatim Whisper and the Parakeet transcripts."""
    run_on_gpus((whisper_decoder, 'verbatim', 0), (whisper_decoder, 'parakeet', 1))


def step_audio_llm():
    """Qwen2-Audio-7B and Voxtral-Mini-3B probes."""
    run_on_gpus((qwen2_audio,))
    run_on_gpus((voxtral,))


def step_text_llm():
    """Qwen2.5-7B (fp16) and Qwen2.5-14B (4-bit) probes on the transcripts."""
    run_on_gpus((text_llm, 'Qwen/Qwen2.5-7B-Instruct', ['verbatim', 'plain']))
    run_on_gpus((text_llm, 'Qwen/Qwen2.5-14B-Instruct', ['wv2', 'parakeet'], True))


def step_gec():
    """Grammatical-error correction and LM surprisal with Qwen2.5-7B."""
    run_on_gpus((gec_and_surprisal,))


def step_hand():
    """Interpretable transcript features (CPU), written as one table."""
    hand_features().to_csv(FEAT / 'hand_features.csv', index=False)


STEPS = {name.removeprefix('step_'): fn for name, fn in globals().items() if name.startswith('step_')}


def main():
    global DATA, FEAT, KEYS, CLIPS, N_CLIPS
    parser = argparse.ArgumentParser(description='Extract the features used by shl_grammar_scoring.ipynb.')
    parser.add_argument('--data', type=Path, help='competition folder (train.csv, test.csv, train/, test/); found automatically on Kaggle')
    parser.add_argument('--out', type=Path, default=Path('features'))
    parser.add_argument('--steps', nargs='+', choices=list(STEPS), default=list(STEPS))
    parser.add_argument('--n-clips', type=int, help='only the first N clips, for a quick test')
    args = parser.parse_args()

    if args.data is None:
        found = [Path(p).parent for p in sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True)) if (Path(p).parent / 'test.csv').exists()]
        args.data = found[0] if found else Path('data')
    DATA, FEAT, N_CLIPS = args.data, args.out, args.n_clips
    for sub in ['asr', 'audio', 'speech', 'decoder', 'audio_llm', 'text_llm', 'gec']:
        (FEAT / sub).mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(DATA / 'train.csv'), pd.read_csv(DATA / 'test.csv')
    KEYS = [f'train/{f}' for f in train.filename] + [f'test/{f}' for f in test.filename]   # filenames repeat across splits
    CLIPS = [(k, DATA / k) for k in KEYS][:N_CLIPS]

    for name in args.steps:
        print(f'[{name}]', flush=True)
        STEPS[name]()


if __name__ == '__main__':
    main()

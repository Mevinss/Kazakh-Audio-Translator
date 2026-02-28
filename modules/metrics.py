try:
    from jiwer import wer, cer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis strings."""
    if not _JIWER_AVAILABLE:
        raise ImportError("jiwer is required for WER calculation. Install it with: pip install jiwer")
    if not reference or not hypothesis:
        return None
    return round(wer(reference, hypothesis), 4)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate between reference and hypothesis strings."""
    if not _JIWER_AVAILABLE:
        raise ImportError("jiwer is required for CER calculation. Install it with: pip install jiwer")
    if not reference or not hypothesis:
        return None
    return round(cer(reference, hypothesis), 4)


def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score between reference and hypothesis strings."""
    if not reference or not hypothesis:
        return None
    try:
        import evaluate
        bleu = evaluate.load("bleu")
        score = bleu.compute(
            predictions=[hypothesis],
            references=[[reference]],
        )
        return round(score.get("bleu", 0.0), 4)
    except Exception:
        return None

from typing import List
from ckip_transformers.nlp import CkipWordSegmenter


class CKIPTokenizer:
    """
    Wrap CKIP word segmenter as a reusable tokenizer.
    """

    def __init__(self, ws_model: str = "bert-base-chinese", device: int = -1, model_dir: str = "./models/ckip-ws") -> None:
        """
        Args:
            ws_model (str):
                CKIP word segmenter model name.
            device (int):
                -1: CPU, 0/1/...: GPU id.
            model_dir (str):
                Local directory of CKIP model weights.
        """
        self.ws = CkipWordSegmenter(
            model=ws_model,
            device=device,
            model_name=model_dir
        )

    def tokenize(self, text: str) -> List[str]:
        return self.ws([text])[0]

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        return self.ws(texts)

from huggingface_hub import snapshot_download
import os


def _download(repo_id: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


def download_bge_m3():
    _download("BAAI/bge-m3", "./models/bge-m3")


def download_bge_reranker():
    _download("BAAI/bge-reranker-v2-m3", "./models/bge-reranker-v2-m3")


def download_ckip_ws():
    _download("ckiplab/bert-base-chinese-ws", "./models/ckip-ws")


if __name__ == "__main__":
    download_bge_m3()
    download_bge_reranker()
    download_ckip_ws()
    print("All models downloaded into ./models/")
import os, pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.database import initQdrant
from utils.denseVec import DenseVector


def main():

    # Initialize Qdrant database.
    df = pd.read_excel("./knowledgeDoc/知識文件蒐集_20251031.xlsx")

    embedder = DenseVector()
    embeddingDocs = embedder.encode(df["relevance"].values.tolist(), encode_type="document")

    payloads = {
        "collection": "FAQ",
        "dimension": embedder.dimension,
        "vecList": embeddingDocs,
        "payloadList": df.to_dict(orient="records")
    }

    client = initQdrant(**payloads)
    print(f"Qdrant initialized with collection: {payloads['collection']}, total points: {len(payloads['vecList'])}")

    # vecSearch
    query = "申請AI PRO手機憑證時，有哪些接收OTP驗證碼的選項？"
    search_result = client.query_points(
        collection_name="FAQ",
        query=embedder.encode(query, encode_type="query"),
        with_payload=["topic", "subtype", "relevance"],
        limit=3,
    )

    for point in search_result.points:
        print("---" * 5)
        print(f"ID: {point.id}")
        print(f"Score: {point.score}")
        print(f"Payload: {point.payload}")


if __name__ == "__main__":
    main()

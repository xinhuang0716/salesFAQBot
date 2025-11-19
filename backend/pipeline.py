import os, pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.database import initQdrant, Qdrant
from utils.denseVec import DenseVector


def main():

    collection = "FAQ"
    df = pd.read_excel("./knowledgeDoc/知識文件蒐集_20251031.xlsx")
    embedder = DenseVector()

    # initialize Qdrant database.
    client = Qdrant(collection=collection)
    if client is None:
        print(f"Collection {collection} does not exist. Initializing Qdrant database...")

        payloads = {
            "collection": collection,
            "dimension": embedder.dimension,
            "vecList": embedder.encode(df["relevance"].values.tolist(), encode_type="document"),
            "payloadList": df.to_dict(orient="records")
        }
        
        client = initQdrant(**payloads)

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

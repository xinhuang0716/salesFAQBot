# import necessary modules
from qdrant_client import models
from utils.database import initQdrant


# main worlkflow
def main():

    # initialize Qdrant database
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    collectionName = "FAQ"
    client = initQdrant(model_name, collectionName)

    if client is None:
        return print("Failed to initialize Qdrant client")

    # vector search test
    search_result = client.query_points(
        collection_name=collectionName,
        query=models.Document(text="若顯示「所選擇分公司未留存電子信箱」，請客戶先 至「線上變更基本資料」新增Email，或選擇其他分公司進行辦理。", model=model_name),
        with_payload=["topic", "subtype", "relevance"],
        limit=5,
    )

    for point in search_result.points:
        print("---" * 5)
        print(f"ID: {point.id}")
        print(f"Score: {point.score}")
        print(f"Payload: {point.payload}")


if __name__ == "__main__":
    main()

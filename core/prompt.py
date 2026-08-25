import textwrap


class PromptTemplate:
    """Prompt template for RAG response generation."""

    @staticmethod
    def system_prompt() -> str:
        """Return the system prompt for RAG response generation.

        Returns:
            str: The system prompt string.

        """
        prompt = """
        你是一位專業的證券公司數位內部助理，負責回答公司知識庫涵蓋之內容，包括：

        - 內部規章
        - 商品與服務
        - 數位平台操作
        - 帳務與交易流程

        回答對象為公司同仁或內部使用者，請以專業、友善且清楚的語氣回覆。

        ---

        # Core Principles

        你只能根據提供的 Knowledge Base（參考資料）回答問題。

        所有回答內容皆須能從參考資料中找到明確依據，不得：

        - 使用外部知識補充
        - 根據常識推論
        - 自行猜測流程、條件、費率、日期或規則
        - 捏造參考資料中不存在的資訊

        若參考資料不足以回答問題，請直接回覆：

        根據目前的資料庫內容，我沒有找到完整或足夠的相關資訊。建議您聯繫人工客服進一步確認。

        不得補充推測內容或提供不具依據的答案。

        ---

        # Response Guidelines

        回答時請：

        1. 優先直接回答問題重點。
        2. 僅使用參考資料中明確記載的資訊。
        3. 若多份資料提到相同議題，請整合為一致的回答。
        4. 重要資訊請以 **粗體** 標示，例如：適用對象、費率、限制條件、注意事項、申請資格、使用期限等

        ---

        # Output Format

        ### 回答

        （以 1～2 句話直接回答問題）

        ### 補充說明

        - 條件或限制
        - 注意事項
        - 其他相關資訊

        ---

        若參考資料不足：

        ### 回答

        根據目前的資料庫內容，我沒有找到完整或足夠的相關資訊。建議您聯繫人工客服進一步確認。
        """

        return textwrap.dedent(prompt).strip()

    @staticmethod
    def construct(query: str, top_k_docs: list[str]) -> str:
        """Construct the full prompt for RAG response generation.

        Args:
            query (str): User's question or query.
            top_k_docs (list[str]): List of top-k retrieved documents.

        Returns:
            str: The constructed full prompt.

        """
        # Format the retrieved documents as context.
        if not top_k_docs:
            context = "（無相關參考文檔）"
        else:
            context = "\n\n".join([f"參考文檔 [{idx}]\n{doc}\n---" for idx, doc in enumerate(top_k_docs, 1)])

        # Construct the full prompt.
        full_prompt = f"""
        ## Knowledge Base:
        {context}

        ## 用戶問題:
        {query}

        請根據上述 Knowledge Base 回答用戶問題。
        """

        return textwrap.dedent(full_prompt).strip()
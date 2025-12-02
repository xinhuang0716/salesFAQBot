def build_text(row: dict) -> str:    
    """
    Build a normalized text representation from a single row in the knowledge Excel file.
    The resulting text will be fed into the embedding model.

    Args:
        row (list[dict]): A list of dictinoary from the DataFrame records that contains at least 'topic', 'subtype', and 'relevance'.

    Returns:
        str: A single concatenated string, for example: "[主題]{topic}\n[子題]{subtype}\n[內容]\n{relevance}"
    """

    return "[主題]{}\n[子題]{}\n[內容]{}".format(str(row["topic"]), str(row["subtype"]), str(row["relevance"]))
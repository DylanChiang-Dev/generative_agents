"""
RAG 模組 - 文本分塊器 (Chunker)

將長文本切分為適合 Embedding 處理的小段落。
支援多種分塊策略：固定大小、按句子、按段落。
"""

import re
from typing import List, Dict, Any


class TextChunker:
    """文本分塊器類別"""

    def __init__(self,
                 chunk_size: int = 512,
                 overlap: int = 50,
                 strategy: str = "fixed"):
        """
        初始化分塊器

        Args:
            chunk_size: 每個分塊的最大字元數
            overlap: 相鄰分塊的重疊字元數，確保語義連續性
            strategy: 分塊策略 - "fixed" (固定大小), "sentence" (按句子), "paragraph" (按段落)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        將文本分塊

        Args:
            text: 原始文本

        Returns:
            分塊列表，每個分塊包含 text, start_idx, end_idx, chunk_idx
        """
        if self.strategy == "fixed":
            return self._chunk_fixed(text)
        elif self.strategy == "sentence":
            return self._chunk_by_sentence(text)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraph(text)
        else:
            raise ValueError(f"未知的分塊策略: {self.strategy}")

    def _chunk_fixed(self, text: str) -> List[Dict[str, Any]]:
        """
        固定大小分塊

        使用滑動窗口方式，確保分塊之間有重疊以保持語義連續性
        """
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # 嘗試在詞邊界處截斷（避免切斷詞語）
            if end < len(text):
                # 尋找最近的空格或標點符號
                for i in range(end, max(start, end - 50), -1):
                    if text[i] in ' \n\t。，！？；：、':
                        end = i + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:  # 只添加非空分塊
                chunks.append({
                    "text": chunk_text,
                    "start_idx": start,
                    "end_idx": end,
                    "chunk_idx": chunk_idx
                })
                chunk_idx += 1

            # 下一個分塊的起始位置（考慮重疊）
            start = end - self.overlap if end < len(text) else len(text)

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[Dict[str, Any]]:
        """
        按句子分塊

        將多個句子組合成一個分塊，直到達到 chunk_size
        """
        # 中英文句子分割
        sentence_pattern = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_idx = 0
        char_pos = 0

        for sentence in sentences:
            # 如果當前分塊加上新句子超過限制
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "start_idx": current_start,
                    "end_idx": char_pos,
                    "chunk_idx": chunk_idx
                })
                chunk_idx += 1
                current_chunk = ""
                current_start = char_pos

            current_chunk += sentence + " "
            char_pos += len(sentence) + 1

        # 處理最後一個分塊
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "start_idx": current_start,
                "end_idx": len(text),
                "chunk_idx": chunk_idx
            })

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[Dict[str, Any]]:
        """
        按段落分塊

        以空行分割段落，如果段落過長則進一步分割
        """
        # 按空行分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        chunk_idx = 0
        char_pos = 0

        for para in paragraphs:
            if len(para) <= self.chunk_size:
                # 段落大小合適，直接作為一個分塊
                chunks.append({
                    "text": para,
                    "start_idx": char_pos,
                    "end_idx": char_pos + len(para),
                    "chunk_idx": chunk_idx
                })
                chunk_idx += 1
            else:
                # 段落過長，使用固定大小分塊
                sub_chunker = TextChunker(
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                    strategy="fixed"
                )
                sub_chunks = sub_chunker.chunk(para)
                for sub in sub_chunks:
                    sub["start_idx"] += char_pos
                    sub["end_idx"] += char_pos
                    sub["chunk_idx"] = chunk_idx
                    chunks.append(sub)
                    chunk_idx += 1

            char_pos += len(para) + 2  # +2 for paragraph separator

        return chunks


def chunk_text(text: str,
               chunk_size: int = 512,
               overlap: int = 50,
               strategy: str = "fixed") -> List[str]:
    """
    便捷函數：將文本分塊，只返回文本列表

    Args:
        text: 原始文本
        chunk_size: 每個分塊的最大字元數
        overlap: 相鄰分塊的重疊字元數
        strategy: 分塊策略

    Returns:
        分塊文本列表
    """
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap, strategy=strategy)
    chunks = chunker.chunk(text)
    return [c["text"] for c in chunks]


def chunk_file(file_path: str,
               chunk_size: int = 512,
               overlap: int = 50,
               strategy: str = "fixed",
               encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """
    從文件讀取並分塊

    Args:
        file_path: 文件路徑
        chunk_size: 每個分塊的最大字元數
        overlap: 重疊字元數
        strategy: 分塊策略
        encoding: 文件編碼

    Returns:
        分塊列表，包含來源文件資訊
    """
    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()

    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap, strategy=strategy)
    chunks = chunker.chunk(text)

    # 添加來源文件資訊
    for chunk in chunks:
        chunk["source"] = file_path

    return chunks


if __name__ == "__main__":
    # 測試範例
    sample_text = """
    人工智慧（Artificial Intelligence, AI）是計算機科學的一個分支，旨在創建能夠執行通常需要人類智慧的任務的系統。

    這些任務包括語音識別、視覺感知、決策制定和語言翻譯等。機器學習是人工智慧的一個子領域，專注於開發能夠從數據中學習的算法。

    深度學習是機器學習的一個子集，使用類似於人腦結構的神經網絡。這些網絡可以自動發現數據中的模式和特徵，無需人工設計。

    近年來，大型語言模型（LLM）如 GPT 系列在自然語言處理領域取得了重大突破，能夠生成高質量的文本、回答問題並進行對話。
    """

    print("=== 固定大小分塊 ===")
    chunker = TextChunker(chunk_size=200, overlap=20, strategy="fixed")
    for chunk in chunker.chunk(sample_text):
        print(f"\n[Chunk {chunk['chunk_idx']}] ({chunk['start_idx']}-{chunk['end_idx']})")
        print(chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"])

    print("\n=== 按段落分塊 ===")
    chunker = TextChunker(chunk_size=500, strategy="paragraph")
    for chunk in chunker.chunk(sample_text):
        print(f"\n[Chunk {chunk['chunk_idx']}]")
        print(chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"])

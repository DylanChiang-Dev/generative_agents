#!/usr/bin/env python3
"""
RAG 系統測試腳本
用於驗證 RAG 檢索功能是否正常運作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from reverie.backend_server.rag.rag_interface import RAGSystem


def test_rag_query():
    """測試 RAG 查詢功能"""
    print("=" * 60)
    print("RAG 系統測試")
    print("=" * 60)

    test_queries = [
        "离婚时财产如何分割？",
        "结婚年龄是多少？",
        "什么情况下婚姻无效？",
        "夫妻共同财产包括哪些？",
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)

        results = RAGSystem.query(query, k=2)

        if results:
            for i, r in enumerate(results):
                print(f"  [{i+1}] Score: {r['score']:.4f}")
                # 顯示前 80 個字符
                text_preview = r['text'][:80].replace('\n', ' ')
                print(f"      {text_preview}...")
        else:
            print("  ❌ 無結果")

    print("\n" + "=" * 60)


def test_persona_integration():
    """模擬 Persona 關鍵詞觸發"""
    print("\n📌 Persona 關鍵詞觸發測試")
    print("=" * 60)

    # 模擬 check_legal_context 邏輯
    keywords = ["婚姻", "离婚", "财产", "抚养", "收养", "夫妻", "子女"]

    test_thoughts = [
        "今天天氣真好，我要去公園散步",           # 無關鍵詞
        "我在想关于离婚财产分割的问题",           # 有關鍵詞
        "夫妻之间应该如何相处",                   # 有關鍵詞
        "我需要了解子女抚养权的规定",             # 有關鍵詞
    ]

    for thought in test_thoughts:
        print(f"\n💭 Thought: {thought}")

        triggered = False
        for kw in keywords:
            if kw in thought:
                print(f"   ✅ 觸發關鍵詞: {kw}")
                results = RAGSystem.query(thought, k=1)
                if results:
                    text_preview = results[0]['text'][:60].replace('\n', ' ')
                    print(f"   📚 檢索結果: {text_preview}...")
                triggered = True
                break

        if not triggered:
            print("   ⏭️  未觸發 RAG（無相關關鍵詞）")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n🚀 開始 RAG 系統測試...\n")

    # 測試 1: RAG 查詢
    test_rag_query()

    # 測試 2: Persona 整合
    test_persona_integration()

    print("\n✅ 測試完成！")

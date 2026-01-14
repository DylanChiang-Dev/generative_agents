# rag_test_002 模擬記錄

## 概述

這是一個三人 Generative Agents 模擬實驗，測試 RAG（檢索增強生成）功能的整合效果。

## 角色

| 角色 | 身份描述 |
|------|---------|
| **Isabella Rodriguez** | Hobbs Cafe 咖啡店老闆 |
| **Sam Moore** | 市長競選候選人，前海軍軍人 |
| **Tao Chiang** | 法律顧問 |

## 模擬統計

- **總對話記錄**：724 條
- **RAG 調用次數**：1 次
- **模擬開始時間**：February 14, 2025, 07:00:00

---

## RAG 調用記錄

### 調用 #1

**時間戳**：`2026-01-14 17:59:37`

**觸發查詢**（Isabella Rodriguez 的對話開場白）：

> "Hey you two, so glad you're here bright and early! I just got the coffee machines all set up—want me to whip up your usual drinks while we start our meeting? First off, let's chat through Sam's mayoral campaign updates, then **I have a quick legal question for Tao about a friend's divorce and custody situation**. Oh, and I can't forget to mention: we're hosting a cozy Valentine's Day party here at 5pm tonight, and I'd love if both of you could help spread the word around the community to get more folks to join in on the fun!"

**RAG 檢索結果**：

從婚姻法文檔 `marriage_law.txt` 中檢索到 2 個相關片段：

| Chunk | 相關性分數 | 內容摘要 |
|-------|-----------|---------|
| 1 | 0.170 | 夫妻共同財產處理權、離婚協議、離婚訴訟程序（第1076條、第1079條） |
| 0 | 0.156 | 婚姻家庭法一般規定、婚姻自由原則、共同財產定義（第1040-1062條） |

**知識來源**：`/Users/dc/Documents/generative_agents/reverie/backend_server/rag/data/marriage_law.txt`

---

## 主要對話主題

1. **Sam 的市長競選策略**
   - 人行道安全交叉口試點計劃（預計減少 38% 行人事故）
   - 小企業許可費分級結構改革
   - 社區圓桌會議策劃

2. **法律諮詢**
   - 離婚與撫養權問題（觸發 RAG 檢索）

3. **社區活動**
   - 情人節派對籌備（當日下午 5 點於 Hobbs Cafe）

---

## 文件結構

```
rag_test_002/
├── README.md           # 本文件
├── rag_log.jsonl       # RAG 調用日誌
├── environment/        # 環境狀態快照
├── movement/          # 角色移動與對話記錄 (442 個 JSON 文件)
├── personas/          # 角色記憶數據
│   ├── Isabella Rodriguez/
│   ├── Sam Moore/
│   └── Tao Chiang/
└── reverie/           # 其他模擬數據
```

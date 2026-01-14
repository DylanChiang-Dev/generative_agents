# 生成式代理人模擬專案 - 作業展示

本專案為課程作業展示用途，基於史丹佛大學的 Generative Agents 研究專案進行部署與測試。

## 專案說明

本專案展示了如何使用大型語言模型（LLM）驅動的生成式代理人，模擬具有可信人類行為的虛擬角色。代理人能夠進行日常活動規劃、社交互動、記憶形成與反思等認知行為。

## 技術架構

- **前端環境伺服器**：Django 網頁應用程式，負責視覺化呈現與環境狀態管理
- **後端模擬伺服器**：Python 程式，驅動代理人的認知模組與行為邏輯
- **LLM 整合**：支援 OpenAI API 相容的語言模型服務（目前配置為 Doubao/火山引擎）

---

## 🔧 API 遷移記錄：OpenAI → 火山引擎 (Volcengine)

本專案已完成從 OpenAI API 到火山引擎 (Volcengine) Doubao 模型的完整遷移。

### 遷移內容

| 項目 | 原設定 | 新設定 |
|------|--------|--------|
| API Base URL | `api.openai.com` | `ark.cn-beijing.volces.com/api/v3` |
| Chat 模型 | `gpt-3.5-turbo` / `gpt-4` | `doubao-seed-1-8-251228` |
| Embedding 模型 | `text-embedding-ada-002` | `ep-xxxxxxxx` (Doubao-embedding-vision) |
| Embedding 端點 | `/v1/embeddings` | `/api/v3/embeddings/multimodal` |

### 🧪 測試場景更新

#### 新增 3 人互動測試 (`base_three_person_setup`)

已建立一個專注於 3 位特定角色互動的測試場景，用於驗證多方對話與社交行為。

**包含角色：**
1. **Tao Chiang**：婚姻家庭律師，住在 Tao Chiang's house，正在撰寫「AI 協助律師」書籍，並支持 Sam Moore 競選市長。
2. **Sam Moore**：市長候選人，住在 Moore family's house，積極進行競選活動。
3. **Isabella Rodriguez**：Hobbs Cafe 老闆，住在 Isabella Rodriguez's apartment，提供聚會場所。

**預設劇情：**
- 三人設定於早上 **9:00 AM** 在 **Hobbs Cafe** 集合。
- 討論主題包含：市長選舉輔選、Tao 的新書發表。

**啟動方式：**
在 `reverie.py` 啟動時：
1. `Enter the name of the forked simulation`: **`base_three_person_setup`**
2. `Enter the name of the new simulation`: [您的自訂名稱]

---

### 🏠 地圖修改：Tao Chiang's house

為了讓 Tao Chiang 擁有獨立的住所，已將地圖中原本的「Yuriko Yamamoto's house」重新命名為「Tao Chiang's house」。

#### 修改的地圖檔案

| 檔案 | 修改內容 |
|------|----------|
| `the_ville/matrix/special_blocks/sector_blocks.csv` | `32196, the Ville, Yuriko Yamamoto's house` → `32196, the Ville, Tao Chiang's house` |
| `the_ville/matrix/special_blocks/arena_blocks.csv` | `32174, the Ville, Yuriko Yamamoto's house, main room` → `32174, the Ville, Tao Chiang's house, main room` |
| | `32184, the Ville, Yuriko Yamamoto's house, bathroom` → `32184, the Ville, Tao Chiang's house, bathroom` |
| `the_ville/matrix/special_blocks/spawning_location_blocks.csv` | `32309, the Ville, Yuriko Yamamoto's house, main room, sp-A` → `32309, the Ville, Tao Chiang's house, main room, sp-A` |
| | `32319, the Ville, Yuriko Yamamoto's house, main room, sp-B` → `32319, the Ville, Tao Chiang's house, main room, sp-B` |

#### 修改的角色檔案 (`base_three_person_setup/personas/Tao Chiang/`)

| 檔案 | 修改內容 |
|------|----------|
| `bootstrap_memory/scratch.json` | `living_area`: `"the Ville:Adam Smith's house:main room"` → `"the Ville:Tao Chiang's house:main room"` |
| `bootstrap_memory/spatial_memory.json` | 將 `"Adam Smith's house"` 區塊重新命名為 `"Tao Chiang's house"` |

#### 修改的環境檔案

| 檔案 | 修改內容 |
|------|----------|
| `base_three_person_setup/environment/0.json` | Tao Chiang 初始座標: `(20, 65)` → `(28, 65)` (對應 Tao Chiang's house 的 spawn point) |

#### 房屋結構

Tao Chiang's house 包含以下區域：
- **main room**: closet, bed, desk, cooking area, kitchen sink, refrigerator
- **bathroom**: bathroom sink, shower, toilet

---

### 修改的檔案

1. **`reverie/backend_server/utils.py`** - API 設定檔
   - `openai_api_key`: 火山引擎 API Key
   - `openai_api_base`: 火山引擎 API Base URL
   - `model_id`: Chat 模型 ID
   - `embedding_model_id`: Embedding 模型 Endpoint ID

2. **`reverie/backend_server/persona/prompt_template/gpt_structure.py`** - 核心 API 呼叫
   - `ChatGPT_request()`, `GPT4_request()`, `ChatGPT_single_request()`: 使用 `model_id` 變數
   - `GPT_request()`: 使用 `model_id` 變數，忽略舊的 `engine` 參數
   - `get_embedding()`: 重寫為使用 `requests` 直接呼叫火山引擎 multimodal embedding API

3. **`reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`** - 移除所有硬編碼 `engine` 參數

4. **`reverie/backend_server/test.py`** - 測試腳本更新

### 已刪除的檔案

- `reverie/backend_server/persona/prompt_template/defunct_run_gpt_prompt.py` - 廢棄的舊版程式碼

### 設定您自己的 API

編輯 `reverie/backend_server/utils.py`：

```python
# API Configuration for Volcengine (Doubao)
openai_api_key = "your-volcengine-api-key"
openai_api_base = "https://ark.cn-beijing.volces.com/api/v3"
model_id = "doubao-seed-1-8-251228"  # 或您的模型 ID
embedding_model_id = "ep-xxxxxxxx"   # 您的 Embedding Endpoint ID
```

### 測試 API 連線

```bash
python3 reverie/backend_server/test.py
```

---

## 🧠 RAG 系統實現 (Retrieval-Augmented Generation)

本專案已實現一個完整的 RAG 系統，展示檢索增強生成的核心技術能力。

### RAG 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG 系統架構                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【索引階段 - 離線處理】                                          │
│                                                                 │
│   知識文檔 ──→ 文本分塊 ──→ Embedding 模型 ──→ 向量存儲           │
│   (.txt)      (Chunking)   (火山引擎)      (JSON/NumPy)         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【查詢階段 - 實時處理】                                          │
│                                                                 │
│   用戶問題 ──→ Embedding ──→ 向量相似度 ──→ Top-K 文檔           │
│      │           │          (Cosine)        │                   │
│      └───────────┴──────────────────────────┘                   │
│                         ↓                                       │
│   Prompt = Context(檢索結果) + Query(用戶問題)                   │
│                         ↓                                       │
│                    LLM 生成回答                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 實現模組

| 模組 | 檔案位置 | 功能 | 展示的技術能力 |
|------|----------|------|----------------|
| **Chunker** | `rag/chunker.py` | 將文本切分成段落 | 文本預處理、分塊策略 (固定大小/句子/段落) |
| **Indexer** | `rag/indexer.py` | 調用 Embedding API 建立索引 | Embedding 模型整合、批量處理 |
| **Vector Store** | `rag/vector_store.py` | 存儲向量與原文映射 | 向量索引設計與持久化 (JSON) |
| **Retriever** | `rag/retriever.py` | 餘弦相似度檢索 Top-K | 向量檢索算法實現 (NumPy) |
| **Interface** | `rag/rag_interface.py` | 統一調用接口 | 單例模式、系統整合 |
| **Integration** | `persona/persona.py` | Agent 關鍵詞觸發 | 認知模組整合 |

### 系統驗證

已通過自動化測試腳本驗證系統功能：

**1. 法律知識檢索**
```text
Query: 离婚时财产如何分割？
[1] Score: 0.6190
    內容: ...夫妻共同财产...继承或者受赠的财产...
[2] Score: 0.4279
    內容: ...婚姻家庭一般规定...
```

**2. Agent 關鍵詞觸發**
- 當 Agent 思考內容包含「婚姻」、「离婚」、「财产」、「抚养」等關鍵詞時，自動觸發 RAG。
- 檢索到的法律條文會注入到 Agent 的 Context 中，輔助決策。

---

## 📊 模擬實驗記錄：rag_test_002

### 劇本設計

本實驗設計了一個三人會面場景，用於測試 RAG 系統在多人對話中的觸發與整合能力。

#### 角色設定

| 角色 | 年齡 | 身份 | 性格特質 |
|------|------|------|----------|
| **Tao Chiang** | 35 | 婚姻家庭律師，十年執業經驗 | 善解人意、耐心、值得信賴、分析力強 |
| **Sam Moore** | 65 | 退役海軍軍官，市長候選人 | 智慧、足智多謀、幽默 |
| **Isabella Rodriguez** | 34 | Hobbs Cafe 咖啡店老闆 | 友善、外向、好客 |

#### 劇情大綱

**時間**：2025 年 2 月 14 日（情人節）早上 8:00

**地點**：Hobbs Cafe

**會議目的**：
1. **Sam 的市長競選策略**：討論如何贏得選民支持
2. **法律諮詢**：Isabella 想為一位朋友諮詢離婚和撫養權問題 ← **RAG 觸發點**
3. **社區活動**：邀請大家幫忙宣傳當日下午 5 點的情人節派對

#### 各角色每日計劃

**Tao Chiang**：
> 早上 8 點去 Hobbs Cafe 與 Sam 和 Isabella 會面討論競選策略。10 點回家處理客戶諮詢至中午，下午 2 點到 6 點處理法律案件。

**Sam Moore**：
> 早上 5 點起床，與妻子 Jennifer 共進早餐後步行到 Hobbs Cafe 開會。會後在 Johnson Park 散步，並與鄰居交流競選理念。

**Isabella Rodriguez**：
> 早上 7:30 開店準備，8 點主持三人會議，同時詢問 Tao 關於朋友離婚和撫養權的法律意見。全天經營咖啡店至晚上 8 點。

### RAG 調用實例

在模擬過程中，當 Isabella 提到「離婚和撫養權問題」時，系統自動觸發 RAG 檢索：

**觸發對話**：
> "Hey you two, so glad you're here bright and early! ... I have a quick legal question for Tao about **a friend's divorce and custody situation**..."

**檢索結果**：

| 來源 | 相關性分數 | 法律條文 |
|------|-----------|----------|
| `marriage_law.txt` Chunk 1 | 0.170 | 第 1076 條（離婚協議）、第 1079 條（離婚訴訟程序） |
| `marriage_law.txt` Chunk 0 | 0.156 | 第 1040-1062 條（婚姻家庭一般規定、共同財產） |

這些法律條文被注入到 Tao Chiang 的上下文中，使他能夠提供專業的法律建議。

### 模擬統計

| 指標 | 數值 |
|------|------|
| 總對話記錄 | 724 條 |
| RAG 調用次數 | 1 次 |
| 模擬時長 | 約 7 小時（模擬時間） |
| 主要討論主題 | 市長競選策略、小企業許可費改革、人行道安全試點計劃 |

### 存儲位置

對話記錄和模擬數據存放於以下位置：

```
generative_agents/                              # 專案根目錄
├── rag_dialogue.md                            # 📝 已提取的完整對話記錄 (724 條)
│
└── environment/frontend_server/storage/rag_test_002/
    ├── README.md                               # 模擬說明文件
    ├── rag_log.jsonl                           # RAG 調用日誌
    ├── movement/                               # 442 個時間步的狀態與對話 JSON
    │   ├── 1.json ... 442.json                 # 每個時間步的完整狀態
    │   └── (chat 字段記錄對話內容)
    ├── personas/                               # 角色記憶數據
    │   ├── Tao Chiang/bootstrap_memory/
    │   ├── Sam Moore/bootstrap_memory/
    │   └── Isabella Rodriguez/bootstrap_memory/
    └── environment/                            # 環境狀態快照
```

**文件說明**：
- `rag_dialogue.md`：從 movement JSON 中提取的所有對話，格式化為 Markdown 方便閱讀
- `movement/*.json`：原始對話數據，`chat` 字段包含對話列表，`null` 表示該時間步無對話
- `rag_log.jsonl`：記錄 RAG 系統被觸發的時間、查詢內容和檢索結果

### 如何測試

本專案包含一個端到端的測試腳本，可用於驗證 RAG 系統狀態：

```bash
# 運行 RAG 測試腳本
python3 reverie/backend_server/rag/test_rag_demo.py
```

### 核心技術實現細節

#### 1. 文本分塊 (Chunking)
```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    將長文本切分為可處理的小段落
    - chunk_size: 每個分塊的最大字元數
    - overlap: 相鄰分塊的重疊字元數，確保語義連續性
    """
```

#### 2. 向量化 (Embedding)
```python
def get_embedding(text: str) -> List[float]:
    """
    調用火山引擎 Embedding API 將文本轉換為向量
    返回: 高維向量
    """
```

#### 3. 向量存儲 (Vector Store)
使用輕量級 JSON 實現，無需額外數據庫依賴，方便部署與教學展示。

#### 4. 相似度檢索 (Retrieval)
```python
def retrieve(query: str, k: int = 3):
    # 1. Query 向量化
    # 2. 計算 Cosine Similarity
    # 3. 排序並返回 Top-K
```

### 與現有專案的整合

本專案的 Generative Agents 已使用 Embedding 進行記憶檢索。RAG 系統復用了 `gpt_structure.py` 中的 `get_embedding` 函數，確保資源利用效率。

在 `Persona` 類中新增了 `check_legal_context` 方法，使 Agent 具備主動查詢法律知識的能力。

### 自建 RAG vs 使用現成庫

| 方面 | 自建實現 (本專案) | 使用 ChromaDB/LangChain |
|------|------------------|-------------------------|
| Embedding 調用 | ✅ 自己調用 | ✅ 自己調用 |
| 向量存儲 | ✅ 自己實現 (JSON) | ❌ 庫封裝 |
| 相似度計算 | ✅ 自己實現 (NumPy) | ❌ 庫封裝 |
| 展示底層原理 | ✅ 完整展示 | ⚠️ 部分隱藏 |
| 依賴複雜度 | ✅ 低 (僅 NumPy) | ⚠️ 高 |

**選擇自建實現的原因**：更能展示對 RAG 技術的深入理解，包括向量索引設計、相似度算法、檢索策略等核心概念，且易於集成到現有的 Agent 模擬循環中。

---

## 快速開始

### 環境設置
```bash
pip install -r requirements.txt
```

### 啟動伺服器
需同時執行兩個伺服器：

1. 環境伺服器：
```bash
cd environment/frontend_server
python3 manage.py runserver
```

2. 模擬伺服器：
```bash
cd reverie/backend_server
python3 reverie.py
```

### API 設定
編輯 `reverie/backend_server/utils.py` 設定您的 API 金鑰與端點。

---

# 以下為原專案說明文件

---

# Generative Agents: Interactive Simulacra of Human Behavior

<p align="center" width="100%">
<img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

This repository accompanies our research paper titled "[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)." It contains our core simulation module for  generative agents—computational agents that simulate believable human behaviors—and their game environment. Below, we document the steps for setting up the simulation environment on your local machine and for replaying the simulation as a demo animation.

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Isabella_Rodriguez.png" alt="Generative Isabella">   Setting Up the Environment 
To set up your environment, you will need to generate a `utils.py` file that contains your OpenAI API key and download the necessary packages.

### Step 1. Generate Utils File
In the `reverie/backend_server` folder (where `reverie.py` is located), create a new file titled `utils.py` and copy and paste the content below into the file:
```
# Copy and paste your OpenAI API Key
openai_api_key = "<Your OpenAI API>"
# Put your name
key_owner = "<Name>"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose 
debug = True
```
Replace `<Your OpenAI API>` with your OpenAI API key, and `<name>` with your name.
 
### Step 2. Install requirements.txt
Install everything listed in the `requirements.txt` file (I strongly recommend first setting up a virtualenv as usual). A note on Python version: we tested our environment on Python 3.9.12. 

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Klaus_Mueller.png" alt="Generative Klaus">   Running a Simulation 
To run a new simulation, you will need to concurrently start two servers: the environment server and the agent simulation server.

### Step 1. Starting the Environment Server
Again, the environment is implemented as a Django project, and as such, you will need to start the Django server. To do this, first navigate to `environment/frontend_server` (this is where `manage.py` is located) in your command line. Then run the following command:

    python manage.py runserver

Then, on your favorite browser, go to [http://localhost:8000/](http://localhost:8000/). If you see a message that says, "Your environment server is up and running," your server is running properly. Ensure that the environment server continues to run while you are running the simulation, so keep this command-line tab open! (Note: I recommend using either Chrome or Safari. Firefox might produce some frontend glitches, although it should not interfere with the actual simulation.)

### Step 2. Starting the Simulation Server
Open up another command line (the one you used in Step 1 should still be running the environment server, so leave that as it is). Navigate to `reverie/backend_server` and run `reverie.py`.

    python reverie.py
This will start the simulation server. A command-line prompt will appear, asking the following: "Enter the name of the forked simulation: ". To start a 3-agent simulation with Isabella Rodriguez, Maria Lopez, and Klaus Mueller, type the following:
    
    base_the_ville_isabella_maria_klaus
The prompt will then ask, "Enter the name of the new simulation: ". Type any name to denote your current simulation (e.g., just "test-simulation" will do for now).

    test-simulation
Keep the simulator server running. At this stage, it will display the following prompt: "Enter option: "

### Step 3. Running and Saving the Simulation
On your browser, navigate to [http://localhost:8000/simulator_home](http://localhost:8000/simulator_home). You should see the map of Smallville, along with a list of active agents on the map. You can move around the map using your keyboard arrows. Please keep this tab open. To run the simulation, type the following command in your simulation server in response to the prompt, "Enter option":

    run <step-count>
Note that you will want to replace `<step-count>` above with an integer indicating the number of game steps you want to simulate. For instance, if you want to simulate 100 game steps, you should input `run 100`. One game step represents 10 seconds in the game.


Your simulation should be running, and you will see the agents moving on the map in your browser. Once the simulation finishes running, the "Enter option" prompt will re-appear. At this point, you can simulate more steps by re-entering the run command with your desired game steps, exit the simulation without saving by typing `exit`, or save and exit by typing `fin`.

The saved simulation can be accessed the next time you run the simulation server by providing the name of your simulation as the forked simulation. This will allow you to restart your simulation from the point where you left off.

### Step 4. Replaying a Simulation
You can replay a simulation that you have already run simply by having your environment server running and navigating to the following address in your browser: `http://localhost:8000/replay/<simulation-name>/<starting-time-step>`. Please make sure to replace `<simulation-name>` with the name of the simulation you want to replay, and `<starting-time-step>` with the integer time-step from which you wish to start the replay.

For instance, by visiting the following link, you will initiate a pre-simulated example, starting at time-step 1:  
[http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/](http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/)

### Step 5. Demoing a Simulation
You may have noticed that all character sprites in the replay look identical. We would like to clarify that the replay function is primarily intended for debugging purposes and does not prioritize optimizing the size of the simulation folder or the visuals. To properly demonstrate a simulation with appropriate character sprites, you will need to compress the simulation first. To do this, open the `compress_sim_storage.py` file located in the `reverie` directory using a text editor. Then, execute the `compress` function with the name of the target simulation as its input. By doing so, the simulation file will be compressed, making it ready for demonstration.

To start the demo, go to the following address on your browser: `http://localhost:8000/demo/<simulation-name>/<starting-time-step>/<simulation-speed>`. Note that `<simulation-name>` and `<starting-time-step>` denote the same things as mentioned above. `<simulation-speed>` can be set to control the demo speed, where 1 is the slowest, and 5 is the fastest. For instance, visiting the following link will start a pre-simulated example, beginning at time-step 1, with a medium demo speed:  
[http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/](http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/)

### Tips
We've noticed that OpenAI's API can hang when it reaches the hourly rate limit. When this happens, you may need to restart your simulation. For now, we recommend saving your simulation often as you progress to ensure that you lose as little of the simulation as possible when you do need to stop and rerun it. Running these simulations, at least as of early 2023, could be somewhat costly, especially when there are many agents in the environment.

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Maria_Lopez.png" alt="Generative Maria">   Simulation Storage Location
All simulations that you save will be located in `environment/frontend_server/storage`, and all compressed demos will be located in `environment/frontend_server/compressed_storage`. 

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Sam_Moore.png" alt="Generative Sam">   Customization

There are two ways to optionally customize your simulations. 

### Author and Load Agent History
First is to initialize agents with unique history at the start of the simulation. To do this, you would want to 1) start your simulation using one of the base simulations, and 2) author and load agent history. More specifically, here are the steps:

#### Step 1. Starting Up a Base Simulation 
There are two base simulations included in the repository: `base_the_ville_n25` with 25 agents, and `base_the_ville_isabella_maria_klaus` with 3 agents. Load one of the base simulations by following the steps until step 2 above. 

#### Step 2. Loading a History File 
Then, when prompted with "Enter option: ", you should load the agent history by responding with the following command:

    call -- load history the_ville/<history_file_name>.csv
Note that you will need to replace `<history_file_name>` with the name of an existing history file. There are two history files included in the repo as examples: `agent_history_init_n25.csv` for `base_the_ville_n25` and `agent_history_init_n3.csv` for `base_the_ville_isabella_maria_klaus`. These files include semicolon-separated lists of memory records for each of the agents—loading them will insert the memory records into the agents' memory stream.

#### Step 3. Further Customization 
To customize the initialization by authoring your own history file, place your file in the following folder: `environment/frontend_server/static_dirs/assets/the_ville`. The column format for your custom history file will have to match the example history files included. Therefore, we recommend starting the process by copying and pasting the ones that are already in the repository.

### Create New Base Simulations
For a more involved customization, you will need to author your own base simulation files. The most straightforward approach would be to copy and paste an existing base simulation folder, renaming and editing it according to your requirements. This process will be simpler if you decide to keep the agent names unchanged. However, if you wish to change their names or increase the number of agents that the Smallville map can accommodate, you might need to directly edit the map using the [Tiled](https://www.mapeditor.org/) map editor.


## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Eddy_Lin.png" alt="Generative Eddy">   Authors and Citation 

**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

Please cite our paper if you use the code or data in this repository. 
```
@inproceedings{Park2023GenerativeAgents,  
author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},  
title = {Generative Agents: Interactive Simulacra of Human Behavior},  
year = {2023},  
publisher = {Association for Computing Machinery},  
address = {New York, NY, USA},  
booktitle = {In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},  
keywords = {Human-AI interaction, agents, generative AI, large language models},  
location = {San Francisco, CA, USA},  
series = {UIST '23}
}
```

## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Wolfgang_Schulz.png" alt="Generative Wolfgang">   Acknowledgements

We encourage you to support the following three amazing artists who have designed the game assets for this project, especially if you are planning to use the assets included here for your own project: 
* Background art: [PixyMoon (@_PixyMoon\_)](https://twitter.com/_PixyMoon_)
* Furniture/interior design: [LimeZu (@lime_px)](https://twitter.com/lime_px)
* Character design: [ぴぽ (@pipohi)](https://twitter.com/pipohi)

In addition, we thank Lindsay Popowski, Philip Guo, Michael Terry, and the Center for Advanced Study in the Behavioral Sciences (CASBS) community for their insights, discussions, and support. Lastly, all locations featured in Smallville are inspired by real-world locations that Joon has frequented as an undergraduate and graduate student---he thanks everyone there for feeding and supporting him all these years.



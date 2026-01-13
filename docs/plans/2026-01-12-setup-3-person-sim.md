# Setup 3-Person Simulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Create a new simulation environment named `base_three_person_setup` containing 3 specific agents: Tao Chiang, Isabella Rodriguez (Coffee Shop Owner), and Sam Moore (Mayoral Candidate).

**Architecture:**
We will create a new storage directory `environment/frontend_server/storage/base_three_person_setup`. We will copy the structural files (`reverie/`, `environment/`) from an existing 3-person base (`base_the_ville_isabella_maria_klaus`) and copy the specific persona data from the comprehensive 25-person base (`base_the_ville_n25`). Finally, we will configure the `meta.json` to bootstrap this new world.

**Tech Stack:** Python, JSON configuration, File System operations.

---

### Task 1: Create Simulation Directory Structure

**Files:**
- Create Directory: `environment/frontend_server/storage/base_three_person_setup`
- Create Directory: `environment/frontend_server/storage/base_three_person_setup/reverie`
- Create Directory: `environment/frontend_server/storage/base_three_person_setup/environment`
- Create Directory: `environment/frontend_server/storage/base_three_person_setup/personas`

**Step 1: Check if directory exists (Safety)**
Run: `ls -d environment/frontend_server/storage/base_three_person_setup`
Expected: Error (No such file or directory) - if it exists, we should abort or use a different name.

**Step 2: Create directories**
Run: `mkdir -p environment/frontend_server/storage/base_three_person_setup/{reverie,environment,personas}`

**Step 3: Verify creation**
Run: `ls -F environment/frontend_server/storage/base_three_person_setup/`
Expected: `environment/`, `personas/`, `reverie/`

---

### Task 2: Copy Environment and Reverie Config

**Files:**
- Copy Source: `environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/environment/0.json`
- Copy Source: `environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/reverie/meta.json`
- Target: `environment/frontend_server/storage/base_three_person_setup/`

**Step 1: Copy environment state**
Run: `cp environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/environment/0.json environment/frontend_server/storage/base_three_person_setup/environment/`

**Step 2: Copy reverie meta config**
Run: `cp environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/reverie/meta.json environment/frontend_server/storage/base_three_person_setup/reverie/`

**Step 3: Verify files**
Run: `ls -l environment/frontend_server/storage/base_three_person_setup/environment/0.json environment/frontend_server/storage/base_three_person_setup/reverie/meta.json`

---

### Task 3: Import Personas

**Files:**
- Source Base: `environment/frontend_server/storage/base_the_ville_n25/personas/`
- Target: `environment/frontend_server/storage/base_three_person_setup/personas/`

**Step 1: Copy Tao Chiang**
Run: `cp -r "environment/frontend_server/storage/base_the_ville_n25/personas/Tao Chiang" environment/frontend_server/storage/base_three_person_setup/personas/`

**Step 2: Copy Isabella Rodriguez**
Run: `cp -r "environment/frontend_server/storage/base_the_ville_n25/personas/Isabella Rodriguez" environment/frontend_server/storage/base_three_person_setup/personas/`

**Step 3: Copy Sam Moore**
Run: `cp -r "environment/frontend_server/storage/base_the_ville_n25/personas/Sam Moore" environment/frontend_server/storage/base_three_person_setup/personas/`

**Step 4: Verify Personas**
Run: `ls -F environment/frontend_server/storage/base_three_person_setup/personas/`
Expected: `Isabella Rodriguez/`, `Sam Moore/`, `Tao Chiang/`

---

### Task 4: Configure Simulation Metadata

**Files:**
- Modify: `environment/frontend_server/storage/base_three_person_setup/reverie/meta.json`

**Step 1: Read current meta.json**
Run: `cat environment/frontend_server/storage/base_three_person_setup/reverie/meta.json`

**Step 2: Update meta.json**
Action: Edit the file to:
1. Change `fork_sim_code` to `base_three_person_setup`
2. Update `persona_names` list to exactly: `["Tao Chiang", "Isabella Rodriguez", "Sam Moore"]`

JSON to write:
```json
{
  "fork_sim_code": "base_three_person_setup",
  "start_date": "February 13, 2023",
  "curr_time": "February 13, 2023, 00:00:00",
  "sec_per_step": 10,
  "maze_name": "the_ville",
  "persona_names": [
    "Tao Chiang",
    "Isabella Rodriguez",
    "Sam Moore"
  ],
  "step": 0
}
```

**Step 3: Verify content**
Run: `cat environment/frontend_server/storage/base_three_person_setup/reverie/meta.json`

---

### Task 5: Launch Instructions (User Guidance)

**Step 1: Verify the simulation is recognized**
This is a manual step for the user, but we will provide the command.
Run: `echo "Verification complete. Ready to guide user on startup."`

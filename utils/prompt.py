import re

query_system_prompt = r'''You are an expert Medical AI Assistant. Your task is to analyze medical images and answer clinical questions.
You have access to an external Knowledge Base.
Use the knowledge as a reference, but prioritize the visual evidence in the image.

**CRITICAL OUTPUT FORMAT:**
You MUST structure your response strictly using the following XML tags:

1. `<Visual_Findings>`: detailed description of observable features in the image (e.g., density, shape, location, texture, borders). DO NOT make a final diagnosis here, only describe what you see.
2. `<Knowledge_Check>`: Briefly mention if the retrieved knowledge helps or conflicts with your visual findings.
3. `<Reasoning>`: Synthesize the visual findings and knowledge to deduce the answer. Explain step-by-step.
4. `<Answer>`: The final concise diagnosis or answer to the question.
'''

def query_prompt(problem, knowledges):
    if knowledges.lower() == 'None':
        return rf"""No Relevant Medical Knowledge.

Please answer the following question strictly based on the provided medical image:
{problem}
"""
    return rf"""**Relevant Medical Knowledge:**
{knowledges}

Please answer the following question strictly based on the provided medical image:
{problem}
"""


summarize_system_prompt = r'''You are a Medical Trajectory Summarizer.
Your goal is to distill a complex reasoning trace into a structured summary for the Knowledge Base Controller.

**Input:**
- Question
- Ground Truth
- VLM's Reasoning Trace
- Critic's Evaluation

**Task:**
Analyze the interaction. If the status was "wrong" or "partial_correct", identify specific medical facts or logic rules that were missing or incorrect.
**Ignore visual perception errors** (as they cannot be fixed by text knowledge). Focus ONLY on what logical rule needs to be learned.

**Output Format:**
Summary: [A concise sentence describing the scenario]
Gap Analysis: [What knowledge was missing? e.g., "The model failed to link feature X to disease Y."]
Suggested Rule: [A draft text of the medical knowledge that should be added/modified]'''



def summarize_prompt(problem, trajectory, evaluation, groundtruth):
    return rf'''**Question:** {problem}
**Ground Truth:** {groundtruth}
**VLM Output:** {trajectory}
**Evaluation:** {evaluation}'''


controller_system_prompt = r'''You are the Guardian of the Medical Knowledge Base.
You receive a summary of a diagnostic failure. Your job is to generate a JSON instruction to update the Knowledge Base (KB).

**SAFEGUARDS (CRITICAL):**
1. **Visual Filter:** If the summary indicates a "Visual Perception Error" (e.g., model didn't see the lesion), output an EMPTY action. You cannot fix the model's eyes by adding text.
2. **Conflict Check:** Do not add knowledge that contradicts established medical facts.
3. **Specificity:** The "content" you add must be generalized enough to be useful, but specific enough to fix the error.

**Operations:**
- "add": Insert a new knowledge entry.
- "modify": Update an existing entry (only if the retrieved knowledge was explicitly wrong).
- "delete": Remove an entry (rarely used, prefer modify).
- "none": No action taken.

**Output Format (JSON only):**
```json
[
    {{  "action": "modify",
        "knowledge": "the modified knowledge content",
        "modified_from": "the ID of the knowledge that is modified",
        "reason": "Why this update is necessary"
    }},
    {{  "action": "add",
        "knowledge": "the added knowledge content",
        "reason": "Why this update is necessary"
    }},
    {{  "action": "delete",
        "delete_id": "the deleted knowledge ID",
        "reason": "Why this update is necessary"
    }},
    {{  "action": "none" }},
    ...    
]
```
'''


def controller_prompt(problem, trajectories, ground_truth, knowledges):
    return rf'''**Question:** {problem}

**Ground Truth:** {ground_truth}

**Summarizer Report:**
{trajectories}

**Existing Knowledge Retrieved:** {knowledges}

Based on the above, generate the database update instruction.'''


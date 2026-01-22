import re

query_system_prompt = rf'''You are an expert Medical AI Assistant. Your task is to analyze medical images and answer clinical questions.
You have access to an external Knowledge Base.

**INPUT**
1. <Medical Image>
2. <Relevant Medical Knowledge> (if any)
3. <Visual_Findings> (from the medical image)
4. <Clinical Question>

**CRITICAL OUTPUT FORMAT:**
You MUST structure your response strictly using the following XML tags:
1. `<Knowledge_Check>`: Briefly mention if the retrieved knowledge helps or conflicts with your visual findings.
2. `<Reasoning>`: Synthesize the visual findings and knowledge to deduce the answer. Explain step-by-step.
3. `<Answer>`: The final concise diagnosis or answer to the question.

**IMPORTANT NOTES:**
Use the knowledge as a reference, but prioritize the visual evidence in the medical image.
'''


def query_prompt(problem, knowledges, visual_findings):
    if knowledges.lower() == '':
        return rf"""No Relevant Medical Knowledge.

**Visual Findings:**
{visual_findings}

Please answer the following question strictly based on the provided medical image:
{problem}
"""
    return rf"""**Relevant Medical Knowledge:**
{knowledges}

**Visual Findings:**
{visual_findings}

Please answer the following question strictly based on the provided medical image:
{problem}
"""


summarize_system_prompt = r'''You are a Medical Trajectory Summarizer.

**Context:**
An agent system is provided with medical knowledge and generates a diagnostic reasoning trajectory for a clinical problem. Your task is to summarize this trajectory step-by-step in a structured format.

**Input:**
- Question: The clinical problem being addressed
- Ground Truth: The correct answer
- VLM's Reasoning Trace: The model's diagnostic process
- Critic's Evaluation: Assessment of correctness and error type

**Task:**
Analyze the reasoning trajectory and create a structured summary.

**For each step, identify:**
1. **Action**: What was done (e.g., visual observation, knowledge application, logical inference, hypothesis formation, conclusion)
2. **Knowledge/Principle Used**: What medical knowledge or reasoning principle was applied
3. **Outcome**: The key intermediate result or conclusion reached

**Error Analysis:**
Using the evaluation results and ground-truth answer, identify steps that reflect:
- Detours or backtracking
- Errors or suboptimal choices
- Missing or misapplied knowledge

For each problematic step, explain:
- **Likely Cause**: (e.g., ambiguous evidence, missing knowledge, incorrect knowledge, overgeneralization, logical fallacy)
- **Impact**: How it affected diagnostic progress or efficiency

**Important:** Preserve each step's original outcome, even if it was wrong or later corrected. Do NOT retroactively fix errors.

**Note:** Ignore visual perception errors (as they cannot be fixed by text knowledge). Focus ONLY on reasoning and knowledge gaps.

**Output Format:**
Summary: [One concise sentence describing the overall diagnostic scenario]

Trajectory Steps:

1. [Action]: [Description of action] → [Outcome]
    Knowledge Used: [Specific knowledge or principle applied]
    Issue (if any): [Error type and likely cause]
2. [Action]: [Description of action] → [Outcome]
    Knowledge Used: [Specific knowledge or principle applied]
    Issue (if any): [Error type and likely cause]
...

Gap Analysis: [What specific medical knowledge or logical rule was missing or incorrectly applied? e.g., "The model failed to recognize that feature X is a hallmark of disease Y, not disease Z."]
'''
# Suggested Rule: [A draft text of the medical knowledge that should be added/modified]


def summarize_prompt(problem, trajectory, evaluation, groundtruth):
    return rf'''**Question:** {problem}
**Ground Truth:** {groundtruth}
**VLM Output:** {trajectory}
**Evaluation:** {evaluation}
'''



controller_system_prompt = r'''You are the Guardian of the Medical Knowledge Base.
Your goal is to EVOLVE the knowledge base by learning from failures.

**INPUTS:**
- Diagnostic Failure Summary
- Ground Truth
- **Currently Retrieved Knowledge** (Important: Check this to avoid duplicates!)

**DECISION LOGIC:**
1. **Visual Errors:** If error_type is "visual_perception_error", output "action": "none". (Text cannot fix vision).
2. **Add Knowledge:** Use ONLY when the model identified the visual features correctly but failed the diagnosis due to missing rules.
   - *Duplicate Check:* If the "Currently Retrieved Knowledge" already contains this rule, do NOT add it again. Output "none" or "modify" to clarify it.
3. **Modify Knowledge:** Use when existing knowledge explicitly misled the model.
4. **Delete Knowledge:** Use ONLY when a knowledge entry is factually incorrect (e.g., outdated medical fact). Do not delete just because it wasn't used.

**Output Format (JSON List):**
```json
[
    {
        "action": "add",
        "knowledge": "Standardized medical fact (e.g., 'Kerley B lines are indicative of pulmonary edema.')",
        "reason": "Model saw lines but missed edema diagnosis."
    },
    {
        "action": "modify",
        "knowledge": "Updated text...",
        "modified_id": "the modified knowledge ID, such as '17'",
        "reason": "Previous text was ambiguous."
    }
    {  "action": "delete",
        "delete_id": "the deleted knowledge ID",
        "reason": "Factually incorrect information."
    },
    {  "action": "none" },
    ...
]
```
'''


def controller_prompt(problem, trajectories, ground_truth, knowledges):
    if knowledges.lower() == '':
        knowledges = "No Existing Retrieval Knowledge."
    return rf'''**Question:**
{problem}

**Ground Truth:**
{ground_truth}

**Summarizer Report:**
{trajectories}

**Existing Retrieval Knowledge:**
{knowledges}

Based on the above, generate the database update instruction.'''


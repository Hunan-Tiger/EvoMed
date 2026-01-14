import re

def query_prompt(problem, knowledges):
    return rf'''Please answer the following problem based on the given medical image:
{problem}

Before answering, you carefully read the following helpful medical knowledge as reference:
{knowledges}
'''


def query_prompt_noKnowledge(problem):
    return rf'''Please answer the following problem based on the given medical image:
{problem}  
'''


def summarize_prompt(problem, trajectory, evaluation, groundtruth):
    return rf'''An agent system is given medical knowledge and generates a diagnostic reasoning trajectory for a clinical problem. Your task is to summarize this trajectory step by step in a structured way.
For each step:
1. Describe the action taken (clinical observation, reasoning, decision, revision, etc.).
2. Specify what knowledge type or reasoning principle was used.
3. State the key intermediate outcome reached at that step.

Using the evaluation results and the ground-truth answer, identify any steps that reflect detours, errors, suboptimal choices, or backtracking. For each, explain the likely cause (e.g., ambiguous evidence, missing/incorrect knowledge, overgeneralization) and the impact on diagnostic progress or efficiency. Keep each step’s original outcome even if it was wrong or later corrected.

Return a numbered summary of the diagnostic trajectory steps, in the following format:

1. Step 1: description of the action and its core outcome.
2. Step 2: description of the action and its core outcome.
3. ...

<problem>
{problem}
</problem>

<trajectory>
{trajectory}
</trajectory>

<evaluation>
{evaluation}
</evaluation>

<ground_truth>
{groundtruth}
</ground_truth>
'''



def update_knowledge_prompt(problem, trajectories, answer, knowledges):
    return rf'''An agent system is provided with a set of medical knowledge items and has tried to solve the clinical problem multiple times with both successful and wrong solutions. Review these problem-solving attempts and extract generalizable insights.
Follow these steps:

1. Trajectory Analysis:
− For successful steps: Identify key correct decisions and insights
− For errors: Pinpoint where and why the reasoning went wrong
− Note any important patterns or strategies used/missed
− Review why some trajectories fail? Is there any existing knowledge that is missed, or does the knowledge not provide enough guidance?

2. Update Existing Medical Knowledge
- Some reasoning trajectories may be clinically correct, while others may be incorrect, incomplete, or unsafe.
  You should ensure that the retained or updated knowledge supports accurate, evidence-aligned, and clinically robust reasoning.
- For each knowledge item, choose one action: [modify, add, delete]
  * modify: Revise existing medical knowledge to improve correctness, clarity, or clinical usefulness
  * add: Introduce missing medical knowledge that improves future diagnostic or decision performance
  * delete: Remove misleading, non-generalizable, or clinically inappropriate knowledge
- Similar knowledge points can be merged through 'delete' and 'modify' operations.
- You may update at most 10 generalizable medical knowledge items for this case.

3. Requirements for Each Knowledge Item (Modified / Added / Deleted)
- Prioritize clinical reasoning and diagnostic patterns over isolated medical facts.
- Present knowledge as actionable clinical insights and diagnostic guidance. For example: "On a normal chest X-ray, the pleural edge must appear continuous and sharply defined. Conversely, sharp, radiolucent (dark) lines along the lung periphery indicate pneumothorax."
- Emphasize knowledge quality and relevance over quantity.
- Maintain concise and focused content:
  * Target length: ≤100 words per item
  * Eliminate redundancy and verbose explanations

Please provide reasoning in details under the guidance of the above 3 steps. After the step-by-step reasoning, you will finish by returning in this JSON format as follows:
```json
[
    {{  "option": "modify",
        "knowledge": "the modified knowledge content",
        "modified_from": "the ID of the knowledge that is modified"
    }},
    {{  "option": "add",
        "knowledge": "the added knowledge content",
    }},
    {{  "option": "delete",
        "delete_id": "the deleted knowledge ID"
    }}, ...
]
```

Note that your updated knowledge entries may not need to cover all the options. You can only use one type of updates or choose to keep all knowledge unchanged.

<problem>
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<groundtruth>
{answer}
</groundtruth>

<knowledge>
{knowledges}
</knowledge>
'''
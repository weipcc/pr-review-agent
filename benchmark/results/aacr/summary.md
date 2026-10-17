## Table 1: overall

| version | PRs | recall (all) | recall (human-written) | recall (AI-written) | precision | findings | mean input tokens | mean time (s) | PRs with no findings | missing files in output | hit 10-step cap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 10 | 10/34 (29%) | 4/17 (24%) | 6/17 (35%) | 10/22 (45%) | 22 | 4,547 | 23 | 2 | 0 | – |
| V2 | 10 | 9/34 (26%) | 3/17 (18%) | 6/17 (35%) | 9/16 (56%) | 16 | 9,883 | 38 | 3 | 0 | – |
| V3 | 10 | 3/34 (9%) | 1/17 (6%) | 2/17 (12%) | 3/5 (60%) | 5 | 87,758 | 60 | 8 | 3 | 8 |

## Table 2: per PR (matched reference comments / findings)

| PR | language | reference comments | V1 | V2 | V3 |
|---|---|---|---|---|---|
| bitcoin#31886 | C++ | 4 | 1 / 2 | 1 / 1 | 0 / 0 |
| dbeaver#37564 | Java | 4 | 2 / 6 | 2 / 3 | 0 / 0 |
| dbeaver#38915 | Java | 5 | 1 / 3 | 1 / 3 | 0 / 0 |
| aspnetcore#62734 | C# | 4 | 0 / 0 | 0 / 0 | 0 / 0 |
| react#33059 | JavaScript | 1 | 1 / 2 | 1 / 2 | 0 / 0 |
| gemini-cli#9177 | TypeScript | 7 | 2 / 3 | 1 / 2 | 0 / 0 |
| framework#56591 | PHP | 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| ollama#9379 | Go | 5 | 2 / 4 | 2 / 4 | 2 / 4 |
| valkey#1889 | C | 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| vllm#24425 | Python | 1 | 0 / 1 | 0 / 0 | 0 / 0 |

## Table 3: recall by context needed

| context | V1 | V2 | V3 |
|---|---|---|---|
| Diff Level | 5/14 (36%) | 5/14 (36%) | 3/14 (21%) |
| File Level | 5/15 (33%) | 4/15 (27%) | 0/15 (0%) |
| Repo Level | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |

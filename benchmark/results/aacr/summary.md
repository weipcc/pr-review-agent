Protocol: AACR-Bench file-level semantic coverage (official judge prompt, temperature 0, greedy dedup).
PRs with V1 and V2 judged: 50 (50 small, 0 large).

## Table 1: overall (PRs with both V1 and V2)

| version | PRs | recall | precision | balanced score | findings | mean input tokens | mean time (s) | PRs with no findings |
|---|---|---|---|---|---|---|---|---|
| V1 | 50 | 79/299 (26%) | 76/195 (39%) | 0.315 | 195 | 13,883 | 70 | 6 |
| V2 | 50 | 56/299 (19%) | 53/119 (45%) | 0.264 | 119 | 42,532 | 107 | 12 |

V3 was run on the pilot only (10 PRs); same PRs for comparison:

| version | PRs | recall | precision | balanced score | findings | mean input tokens | mean time (s) | PRs with no findings |
|---|---|---|---|---|---|---|---|---|
| V1 | 10 | 10/34 (29%) | 10/22 (45%) | 0.357 | 22 | 4,547 | 23 | 2 |
| V2 | 10 | 8/34 (24%) | 8/16 (50%) | 0.320 | 16 | 9,883 | 38 | 3 |
| V3 | 10 | 3/34 (9%) | 3/5 (60%) | 0.154 | 5 | 87,758 | 60 | 8 |

## Table 2: split by the mode V2 used on the PR (V1 always reviews file by file; its rows are the same PRs)

| PRs where V2 used ... / version | PRs | recall | precision | balanced score | findings | mean input tokens | mean time (s) | PRs with no findings |
|---|---|---|---|---|---|---|---|---|
| batch mode (21 PRs) / V1 | 21 | 15/74 (20%) | 15/39 (38%) | 0.265 | 39 | 4,239 | 28 | 4 |
| batch mode (21 PRs) / V2 | 21 | 12/74 (16%) | 12/30 (40%) | 0.231 | 30 | 8,183 | 32 | 6 |
| per-file mode (29 PRs) / V1 | 29 | 64/225 (28%) | 61/156 (39%) | 0.329 | 156 | 20,866 | 100 | 2 |
| per-file mode (29 PRs) / V2 | 29 | 44/225 (20%) | 41/89 (46%) | 0.275 | 89 | 67,404 | 162 | 6 |

## Table 3: recall by context needed (AACR-Bench label)

| context | V1 | V2 |
|---|---|---|
| Diff Level | 33/136 (24%) | 20/136 (15%) |
| File Level | 33/107 (31%) | 22/107 (21%) |
| Repo Level | 13/56 (23%) | 14/56 (25%) |

## Table 4: per PR (covered references / effective findings / findings; input tokens)

| PR | group | V2 route | references | V1 | V2 |
|---|---|---|---|---|---|
| CherryHQ/cherry-studio#5540 | small | per-file | 19 | 8 / 8 / 16; 20,781 | 10 / 8 / 14; 81,311 |
| ClickHouse/ClickHouse#80391 | small | batch | 7 | 1 / 1 / 4; 4,922 | 1 / 1 / 2; 9,499 |
| FreeCAD/FreeCAD#19426 | small | per-file | 23 | 1 / 1 / 5; 17,244 | 1 / 1 / 4; 44,261 |
| FreeCAD/FreeCAD#22235 | small | per-file | 6 | 0 / 0 / 0; 41,301 | 0 / 0 / 0; 115,727 |
| alibaba/spring-ai-alibaba#869 | small | per-file | 4 | 2 / 2 / 4; 7,944 | 1 / 1 / 2; 31,491 |
| appwrite/appwrite#9999 | small | per-file | 18 | 7 / 7 / 11; 34,810 | 0 / 0 / 0; 138,249 |
| astral-sh/uv#12941 | small | batch | 1 | 0 / 0 / 2; 6,043 | 0 / 0 / 0; 10,015 |
| bitcoin/bitcoin#31886 | small | batch | 4 | 1 / 1 / 2; 4,628 | 1 / 1 / 1; 8,865 |
| bitcoin/bitcoin#32425 | small | per-file | 4 | 1 / 1 / 4; 20,393 | 0 / 0 / 1; 58,714 |
| bluewave-labs/Checkmate#1897 | small | batch | 9 | 2 / 2 / 8; 5,270 | 1 / 1 / 5; 9,538 |
| browser-use/browser-use#1482 | small | per-file | 2 | 1 / 1 / 5; 20,622 | 1 / 1 / 5; 59,264 |
| browser-use/browser-use#857 | small | per-file | 2 | 1 / 1 / 5; 29,790 | 1 / 1 / 4; 103,424 |
| cline/cline#2347 | small | per-file | 9 | 2 / 2 / 8; 14,120 | 3 / 2 / 5; 38,104 |
| dbeaver/dbeaver#37564 | small | per-file | 4 | 2 / 2 / 6; 6,584 | 2 / 2 / 3; 15,885 |
| dbeaver/dbeaver#38915 | small | batch | 5 | 1 / 1 / 3; 5,011 | 1 / 1 / 3; 9,091 |
| dotnet/aspnetcore#61006 | small | batch | 6 | 2 / 2 / 3; 5,827 | 1 / 1 / 2; 9,560 |
| dotnet/aspnetcore#62734 | small | batch | 4 | 0 / 0 / 0; 5,072 | 0 / 0 / 0; 9,290 |
| dotnet/aspnetcore#62936 | small | batch | 5 | 0 / 0 / 1; 3,685 | 0 / 0 / 1; 7,869 |
| elastic/elasticsearch#118895 | small | per-file | 5 | 2 / 2 / 10; 33,052 | 3 / 3 / 5; 132,082 |
| elastic/elasticsearch#118998 | small | batch | 1 | 0 / 0 / 1; 6,485 | 0 / 0 / 1; 9,972 |
| elastic/elasticsearch#120546 | small | per-file | 3 | 0 / 0 / 0; 14,082 | 0 / 0 / 0; 36,913 |
| facebook/react#33059 | small | batch | 1 | 1 / 1 / 2; 2,783 | 1 / 1 / 2; 6,314 |
| filamentphp/filament#15756 | small | batch | 1 | 0 / 0 / 0; 2,120 | 0 / 0 / 0; 5,659 |
| gofr-dev/gofr#1696 | small | per-file | 4 | 2 / 2 / 9; 20,258 | 1 / 1 / 4; 66,940 |
| google-gemini/gemini-cli#9177 | small | batch | 7 | 2 / 2 / 3; 2,672 | 1 / 1 / 2; 6,770 |
| immich-app/immich#16893 | small | per-file | 19 | 3 / 2 / 9; 20,498 | 2 / 2 / 5; 81,982 |
| juspay/hyperswitch#7353 | small | per-file | 8 | 0 / 0 / 1; 9,197 | 0 / 0 / 0; 38,988 |
| laravel/framework#56591 | small | batch | 1 | 0 / 0 / 0; 1,986 | 0 / 0 / 0; 5,358 |
| libsdl-org/SDL#12718 | small | per-file | 8 | 2 / 2 / 2; 21,386 | 3 / 3 / 3; 24,461 |
| linera-io/linera-protocol#4639 | small | batch | 6 | 1 / 1 / 1; 5,012 | 1 / 1 / 2; 9,074 |
| lvgl/lvgl#8164 | small | per-file | 5 | 2 / 2 / 4; 11,803 | 0 / 0 / 0; 39,575 |
| microsoft/PowerToys#40814 | small | batch | 3 | 0 / 0 / 0; 5,888 | 0 / 0 / 0; 10,256 |
| microsoft/typescript-go#1446 | small | per-file | 13 | 3 / 3 / 4; 6,599 | 3 / 3 / 4; 21,959 |
| microsoft/typescript-go#691 | small | batch | 3 | 0 / 0 / 2; 3,384 | 0 / 0 / 2; 6,935 |
| mrdoob/three.js#30750 | small | batch | 4 | 1 / 1 / 3; 5,383 | 1 / 1 / 3; 8,904 |
| mrdoob/three.js#31402 | small | per-file | 10 | 4 / 4 / 8; 11,140 | 3 / 3 / 6; 32,296 |
| n8n-io/n8n#13323 | small | per-file | 13 | 4 / 4 / 8; 21,639 | 2 / 2 / 2; 63,931 |
| nextcloud/server#52846 | small | per-file | 2 | 2 / 2 / 3; 37,233 | 2 / 2 / 2; 192,699 |
| ollama/ollama#10135 | small | per-file | 3 | 0 / 0 / 5; 10,860 | 0 / 0 / 2; 33,419 |
| ollama/ollama#9379 | small | per-file | 5 | 2 / 2 / 4; 10,726 | 1 / 1 / 4; 22,637 |
| openai/codex#3212 | small | per-file | 7 | 2 / 2 / 5; 56,574 | 0 / 0 / 2; 177,205 |
| ppy/osu#31232 | small | batch | 2 | 1 / 1 / 1; 897 | 1 / 1 / 2; 4,959 |
| sveltejs/svelte#16232 | small | per-file | 5 | 0 / 0 / 1; 9,338 | 0 / 0 / 2; 91,615 |
| symfony/symfony#60956 | small | batch | 1 | 1 / 1 / 1; 5,956 | 1 / 1 / 1; 9,312 |
| timescale/timescaledb#7935 | small | per-file | 5 | 2 / 2 / 2; 25,597 | 1 / 1 / 1; 29,747 |
| valkey-io/valkey#1889 | small | batch | 2 | 1 / 1 / 1; 1,826 | 1 / 1 / 1; 5,779 |
| valkey-io/valkey#2059 | small | per-file | 11 | 6 / 5 / 10; 47,977 | 1 / 1 / 5; 125,678 |
| vllm-project/vllm#12608 | small | per-file | 6 | 2 / 1 / 4; 15,180 | 3 / 3 / 4; 37,129 |
| vllm-project/vllm#15998 | small | per-file | 2 | 1 / 1 / 3; 8,409 | 0 / 0 / 0; 19,055 |
| vllm-project/vllm#24425 | small | batch | 1 | 0 / 0 / 1; 4,182 | 0 / 0 / 0; 8,843 |

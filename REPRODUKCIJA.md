# Репродукција резултата из рада

## Окружење

Мерено на Python **3.14**, macOS, Apple silicon. Пакети су пиновани на верзије под којима
су објављена покретања урађена:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest        # свита пролази у целости
```

Верзија `cma` је битна: `pantograph/bilevel.py` се ослања на то како `cma` 4.4.4 узима
нормале (`opts['randn']`, `seed=NaN`). Са другом верзијом ток може тихо да се разликује.

## Режим поређења

Сва покретања из рада иду у режиму поређења: фиксна резолуција `N = 720`, буџет 100 000
позива симулатора за оба метода, без раног заустављања, уз запис најбоље грешке на решетки
од 500 позива. Једно покретање траје око 20 минута (bilevel) односно око 10 (baseline).

## Матрице

Скрипта `tools/matrica_poredjenja.sh` пушта целу матрицу и прескаче покретања која већ имају
`log.json`, па се сме прекинути и поновити.

```bash
# Е2 — променљиво n, шест кривих, шест seed-ова (72 покретања)
OUT=results/e2 SEEDOVI="1 2 3 4 5 6" PARALLEL=3 tools/matrica_poredjenja.sh

# Е1 — фиксно n, пет кривих без круга, по матрица за свако n
OUT=results/e1_n6 N_CVOROVA=6 SEEDOVI="1 2 3 4 5 6" KRIVE="ellipse egg limacon cassini superellipse" PARALLEL=3 tools/matrica_poredjenja.sh
OUT=results/e1_n7 N_CVOROVA=7 SEEDOVI="1 2 3 4 5"   KRIVE="ellipse egg limacon cassini superellipse" PARALLEL=3 tools/matrica_poredjenja.sh
OUT=results/e1_n8 N_CVOROVA=8 SEEDOVI="1 2 3 4 5 6" KRIVE="ellipse egg limacon cassini superellipse" PARALLEL=3 tools/matrica_poredjenja.sh
```

Једно покретање, онако како га скрипта позива:

```bash
.venv/bin/python run.py run --method bilevel --curve data/curves/limacon.txt --seed 2 \
    --budget 100000 --population 20 --max-link-ratio 5.0 --rezim poredjenje \
    --fiksno-n 8 --log-every 10 --outer-population 20 --k-max 200 --out results/e1_n8
```

Аблација (две криве, три seed-а, по две поставке):

```bash
# Б — двострука популација, исти буџет
.venv/bin/python run.py run --method baseline --curve data/curves/cassini.txt --seed 1 \
    --budget 100000 --population 200 --rezim poredjenje --log-every 50 --out results/abl_B
# В — иста популација, двоструки буџет
.venv/bin/python run.py run --method baseline --curve data/curves/cassini.txt --seed 1 \
    --budget 200000 --rezim poredjenje --log-every 50 --out results/abl_V
```

## Излаз

Свако покретање прави фолдер `results/<матрица>/<датум>_<метод>_<крива>_seed<N>/` са
`log.json` (цео ток: коначна грешка, крива грешка-по-позивима, по генерацији најбоља и
средња вредност, број чворова, угао преноса, гломазност, неважеће јединке), `best_genome.json`,
`command.txt` (тачан позив, git комит, sha256 циљне криве, верзија Python-а) и сликама.

Криве грешке се цртају са:

```bash
.venv/bin/python tools/kriva_greske.py results/e1_n8/2026*_seed* --out results/e1_n8/slike
```

## Комити под којима су објављена покретања урађена

| комит | шта |
|---|---|
| `c2262cf` | највећи део матрица Е2, `n=6`, `n=8` |
| `878084b` | поправка `ZeroDivisionError` на готово-нултој полузи; матрица `n=7` и два поновљена покретања |
| `47eb676` | аблација |

Сваки `log.json` носи свој комит у пољу `git_commit`, па се припадност покретања увек види
из самог лога.

## Објављени логови

Прилог `logovi_ieeestec_2026.tgz` уз ознаку `ieeestec-2026` садржи `log.json`, `best_genome.json` и
`command.txt` свих 266 покретања (242 у четири матрице + 24 у аблацији). Слике и међуфајлови
нису укључени.

# Извештај — ноћна bilevel сесија (05.09.)

> Ажурира се после сваке целине из `PROMPT_BILEVEL.md`, не тек на крају. Ако нешто заустави
> рад јер противречи закљученој одлуци, тај случај иде НА ВРХ овог фајла.

## Стање

_(попуњава се)_

## Целина А — одлуке (docs/DECISIONS.md §22, OPEN_QUESTIONS 2.1, CLAUDE.md §8)

- Урађено: §22 уписан у DECISIONS.md (А1–А7 из промпта, стил постојећих одлука), 2.1
  затворено у OPEN_QUESTIONS.md (детаљни унос + резиме табела на два места), CLAUDE.md §8
  табела ажурирана, нацрт дневника у `docs/DNEVNIK_NACRT_05_09.md` (није уписан у Notion).
- Ниједна измена кода — само документација. pytest се покреће пре комита ради сигурности
  (не очекује се промена резултата).
- Комит: у току.

## Целина Б — pantograph/geometry_vector.py

- Урађено: `genome.TopologySkeleton` (n_nodes + genes без ρ), `geometry_vector.py` са
  `skeleton_of`, `active_positions` (предачко стабло директно из скелета, без
  `prune_dead_nodes`), `to_x`, `to_sequence_from_x`, `evaluate_vector` (једина капија ка
  фитнесу, тачно један инкремент бројача по позиву — и на путу `from_sequence is None`,
  и на путу `fitness.evaluate`).
- `tests/test_geometry_vector.py`, 4 теста из промпта: round-trip 1e-12, димензија 2n−2 без
  мртвог терета, мртав чвор из `add_node` начина Б (димензија −2, `evaluate_vector` на
  мутираном == `fitness.evaluate` на оригиналном — путања се не мења), бројач тачно k за k
  позива (валидни и намерно поломљени вектори). Сва четири прошла из прве.
- pytest: цео пакет пролази.
- Комит: у току.

## Целина В — pantograph/bilevel.py

- Урађено: `TopologyRecord` (лењо конструисан `cma.CMAEvolutionStrategy`, `rng_state`
  изолација legacy `numpy.random` глобалног стања — в. напомена испод), `inner_cmaes`
  (динамички K преко `plateau_detected`, `fixed_k` режим за H3, прекид без `tell()` на
  непотпуној популацији), `warm_start` (језгро увек 1:1, `index_map` по операцији),
  `_abs_map_add_node_a/_b/_delete_node` (табела из В3), `outer_ga` (огледа
  `baseline.evolve`: распоред N, преоцена на промену N, плато заустављања на largest N,
  елита задржава живи ЦМА-ЕС, деца преко тополошке само-мутације).
- **Инжењерска напомена (није противречила ниједној одлуци, само техничка нужност):**
  `cma` библиотека узорке у `ask()` вуче из **legacy `numpy.random` глобалног стања**, не
  из сопственог изолованог генератора, иако прима опцију `seed` (потврђено експериментом
  — исти `seed` даје РАЗЛИЧИТ резултат ако се глобално стање између позива промени). Ово
  је тачно упозорење из PROMPT_BILEVEL целина Ђ.3 („ако cma унесе недетерминизам преко
  неког глобалног стања, нађи узрок"). Решење: сваки `TopologyRecord` носи сопствени
  `rng_state` (snapshot legacy стања), `_ask_isolated`/`_construct_cma_es` привремено
  постављају то стање, позивају `cma`, снимају ново стање назад у запис, па ВРАЋАЈУ
  позиваочево стање — потпуна изолација, независна од редоследа позива над другим
  записима и од било чега другог у процесу (нпр. другог pytest теста) што дира исти
  legacy РНГ. Провера: два узастопна `outer_ga` позива са истим seed-ом дала су
  бит-идентичан `RunLog` (records, `final_error`, `best_genome.coords`) — ручно тестирано
  пре писања `tests/test_bilevel.py` (та провера постаје Ђ.3).
- `delete_node_with_target` додато у `operators.py` (адитивно) — `delete_node` постаје
  тањи омотач, потпис и понашање (и потрошња `rng`-а) непромењени; bilevel-у треба
  обрисана позиција за `_abs_map_delete_node`, baseline је и даље не види.
- `experiment.spawn_rng_streams` добија опциони `n_streams=4` (подразумевано понашање
  непромењено — `SeedSequence.spawn` детерминистичка по редном броју, прва 4 тока
  бит-идентична независно од `n_streams`, проверено ручно). `Config` добија bilevel поља
  (outer_population, cma_lambda, cma_sigma0_new, cma_rho_lower/upper, cma_stds_q1/q2/gene,
  plateau_window_k/eps_k, k_max, fixed_k) — сва ван baseline путање, `QUICK_CONFIG`
  добија мању bilevel секцију (outer_population=8, plateau_window_k=4, k_max=15).
- Ручно тестирано (пре tests/test_bilevel.py): `outer_ga` завршава до краја на кругу,
  детерминизам (два позива исти seed → бит-идентичан лог укључујући координате), N
  транзиција (90→180, преоцена популације при промени), `fixed_k` режим.
- pytest (цео пакет): пролази.
- Комит: у току.

## Целина Г — run.py

- Урађено: `--method bilevel` више не диже `NotImplementedError`, позива `bilevel.outer_ga`
  истим током (репортер, сличносна трансформација, `log.save`, слике, `command.txt`).
  Нови аргументи `--outer-population`, `--k-max`, `--fixed-k` — сви резолвовани у
  `command.txt` (исти образац као `--max-link-ratio`, §20). `progress.ProgressReporter.start`
  добија ред „K режим" кад је метод bilevel (динамички ≤K или фиксно K).
  `--population` остаје баскелине-специфичан; bilevel чита `--outer-population`, са
  падом уназад на `--population` ако је прослеђен, па на профил.
- Ручно тестирано преко CLI: `python run.py run --method bilevel --curve
  data/curves/ellipse.txt --seed 1 --quick` — 6 генерација, 4s, коначна грешка
  1.16e-03, сви фајлови (`log.json`, `best_genome.json`, слике, `command.txt`) исправно
  снимљени, заглавље приказује „K режим динамички (плато, K ≤ 15)".
- pytest (цео пакет): пролази.
- Комит: у току.

## Целина Д — регресија baseline-a

_(попуњава се)_

## Целина Ђ — tests/test_bilevel.py

_(попуњава се)_

## Целина Ж — мерење

_(попуњава се)_

## Списак комитова на грани `bilevel`

_(попуњава се)_

## Одлуке које чекају потврду

_(попуњава се)_

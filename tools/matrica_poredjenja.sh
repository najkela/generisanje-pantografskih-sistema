#!/usr/bin/env bash
# Матрица режима поређења (DECISIONS §24.2): шест кривих нивоа 1 × три seed-а × два метода
# = 36 покретања, фиксно N=720, буџет 100 000 позива, без раног заустављања.
#
# Опционо, димензија броја чворова n (DECISIONS §26.3, Е1): N_CVOROVA="6 8" покрива обе
# вредности у једном покретању скрипте (свако `n` добија сопствену матрицу 36 покретања).
# Подразумевано празно — Е2 матрица (променљиво n, затечено понашање), потпуно нетакнута.
#
# Три паралелна процеса — не више: мерено 14.09. на macOS-у, 5 000 позива за 55 s ⇒
# 100 000 позива ≈ 18–20 минута по bilevel покретању (раније наведена процена „40–50 минута"
# је застарела). И даље вреди пуштати преко ноћи ако је N_CVOROVA непразно (дупло/троструко
# више покретања).
#
# Скрипта се сме прекинути и поново покренути: покретање које већ има `log.json` се
# прескаче. На крају исписује шта није завршило.
#
# Употреба:
#   tools/matrica_poredjenja.sh                    # Е2: све, подразумевани бројеви
#   N_CVOROVA="6 8" tools/matrica_poredjenja.sh     # Е1: исто, за оба фиксна n
#   PY=.venv/bin/python tools/matrica_poredjenja.sh
#   PARALLEL=2 BUDGET=50000 tools/matrica_poredjenja.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
OUT="${OUT:-results/poredjenje}"
BUDGET="${BUDGET:-100000}"
PARALLEL="${PARALLEL:-3}"
KRIVE="${KRIVE:-circle ellipse egg limacon cassini superellipse}"
SEEDOVI="${SEEDOVI:-1 2 3}"
METODE="${METODE:-baseline bilevel}"
N_CVOROVA="${N_CVOROVA:-}"   # DECISIONS §26.3 — нпр. "6 8"; празно = без фиксног n (Е2)

if [ ! -x "$PY" ]; then
  echo "Нема интерпретера на '$PY'. Постави PY=… (нпр. PY=\$(which python3))." >&2
  exit 1
fi

mkdir -p "$OUT" "$OUT/dnevnik"

# Једно покретање. Четврти аргумент је `n` или литерал „-" (без фиксног n — Е2). Прескаче
# ако већ постоји фолдер са log.json за исту четворку.
pokreni() {
  local metod="$1" kriva="$2" seed="$3" n="$4"
  local oznaka="${metod}_${kriva}_seed${seed}"
  local fiksno_n_flag=()
  if [ "$n" != "-" ]; then
    oznaka="${metod}_${kriva}_n${n}_seed${seed}"
    fiksno_n_flag=(--fiksno-n "$n")
  fi
  if compgen -G "$OUT/*_${oznaka}/log.json" > /dev/null; then
    echo "  прескачем $oznaka — већ завршено"
    return 0
  fi
  echo "  крећем $oznaka"
  MPLBACKEND=Agg "$PY" run.py run \
      --method "$metod" --curve "data/curves/${kriva}.txt" --seed "$seed" \
      --rezim poredjenje "${fiksno_n_flag[@]}" --budget "$BUDGET" --log-every 10 --out "$OUT" \
      > "$OUT/dnevnik/${oznaka}.log" 2>&1
  local kod=$?
  if [ $kod -ne 0 ]; then
    echo "  ПАО $oznaka (излаз $kod) — в. $OUT/dnevnik/${oznaka}.log"
  fi
  return 0
}
export -f pokreni
export OUT BUDGET PY

# Bilevel прво: дужа су покретања, па се боље распореде по процесима.
posao() {
  for metod in $METODE; do
    for kriva in $KRIVE; do
      for seed in $SEEDOVI; do
        if [ -z "$N_CVOROVA" ]; then
          echo "$metod $kriva $seed -"
        else
          for n in $N_CVOROVA; do
            echo "$metod $kriva $seed $n"
          done
        fi
      done
    done
  done
}

echo "Матрица: $(posao | wc -l | tr -d ' ') покретања, $PARALLEL паралелно, буџет $BUDGET"
[ -n "$N_CVOROVA" ] && echo "Фиксно n: $N_CVOROVA (DECISIONS §26.3)"
echo "Излаз: $OUT"
posao | sort -r | xargs -P "$PARALLEL" -n 4 bash -c 'pokreni "$0" "$1" "$2" "$3"'

echo
echo "=== шта није завршило ==="
nedostaje=0
for metod in $METODE; do
  for kriva in $KRIVE; do
    for seed in $SEEDOVI; do
      n_lista="${N_CVOROVA:--}"
      for n in $n_lista; do
        oznaka="${metod}_${kriva}_seed${seed}"
        opis="$metod $kriva seed$seed"
        if [ "$n" != "-" ]; then
          oznaka="${metod}_${kriva}_n${n}_seed${seed}"
          opis="$metod $kriva n$n seed$seed"
        fi
        if ! compgen -G "$OUT/*_${oznaka}/log.json" > /dev/null; then
          echo "  $opis"
          nedostaje=$((nedostaje + 1))
        fi
      done
    done
  done
done
[ "$nedostaje" -eq 0 ] && echo "  (све завршено)"
echo
echo "Слике и CSV: $PY tools/kriva_greske.py $OUT/*_seed* --out $OUT/slike"

#!/usr/bin/env bash
# Матрица режима поређења (DECISIONS §24.2): шест кривих нивоа 1 × три seed-а × два метода
# = 36 покретања, фиксно N=720, буџет 100 000 позива, без раног заустављања.
#
# Три паралелна процеса — не више: једно bilevel покретање на 100 000 позива траје 40–50
# минута, па је цела матрица једна ноћ.
#
# Скрипта се сме прекинути и поново покренути: покретање које већ има `log.json` се
# прескаче. На крају исписује шта није завршило.
#
# Употреба:
#   tools/matrica_poredjenja.sh                    # све, подразумевани бројеви
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

if [ ! -x "$PY" ]; then
  echo "Нема интерпретера на '$PY'. Постави PY=… (нпр. PY=\$(which python3))." >&2
  exit 1
fi

mkdir -p "$OUT" "$OUT/dnevnik"

# Једно покретање. Прескаче ако већ постоји фолдер са log.json за исту тројку.
pokreni() {
  local metod="$1" kriva="$2" seed="$3"
  local oznaka="${metod}_${kriva}_seed${seed}"
  if compgen -G "$OUT/*_${oznaka}/log.json" > /dev/null; then
    echo "  прескачем $oznaka — већ завршено"
    return 0
  fi
  echo "  крећем $oznaka"
  MPLBACKEND=Agg "$PY" run.py run \
      --method "$metod" --curve "data/curves/${kriva}.txt" --seed "$seed" \
      --rezim poredjenje --budget "$BUDGET" --log-every 10 --out "$OUT" \
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
        echo "$metod $kriva $seed"
      done
    done
  done
}

echo "Матрица: $(posao | wc -l | tr -d ' ') покретања, $PARALLEL паралелно, буџет $BUDGET"
echo "Излаз: $OUT"
posao | sort -r | xargs -P "$PARALLEL" -n 3 bash -c 'pokreni "$0" "$1" "$2"'

echo
echo "=== шта није завршило ==="
nedostaje=0
for metod in $METODE; do
  for kriva in $KRIVE; do
    for seed in $SEEDOVI; do
      if ! compgen -G "$OUT/*_${metod}_${kriva}_seed${seed}/log.json" > /dev/null; then
        echo "  $metod $kriva seed$seed"
        nedostaje=$((nedostaje + 1))
      fi
    done
  done
done
[ "$nedostaje" -eq 0 ] && echo "  (све завршено)"
echo
echo "Слике и CSV: $PY tools/kriva_greske.py $OUT/*_seed* --out $OUT/slike"

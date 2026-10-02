# Recenzija — `RAD_v1.md` (mentorske beleške)

Prva verzija rada, 22-23 septembar 2026. Beleške nastale pregledom teksta rada i
implementacije u `generisanje-pantografskih-sistema/` (grana `bilevel`, komit `47eb676`).

## Obim za IEEESTEC ✂️

Uputstvo za autore: <https://ieee.elfak.rs/instructions-for-authors/>, ograničenje **4
strane** u IEEE A4 formatu (dva stupca, 10pt).

Merenje sadašnjeg teksta: 27 330 znakova proze (4 257 reči), 10 tabela sa 44 reda podataka,
5 slika, 29 naslova, 13 referenci. Pri gustini od ~5 200–5 500 znakova po punoj strani tog
formata to izlazi na **oko 9 strana**. Za 4 strane ostaje ~2.45 strane proze (≈13 000
znakova, ≈2 050 reči), najviše dve slike i dve do tri male tabele.

**Rad treba prepoloviti, i nešto preko toga** — proza ispod polovine, slike 5 → 2, tabele
10 → 2–3.

Sve ostale napomene u ovom dokumentu tiču se **opšteg kvaliteta rada** i pisane su bez
obzira na ograničenje dužine. Ne bave se time šta čuvati a šta seći; to je zasebna odluka
koju treba napraviti nezavisno, ostavljam je tebi. Velika težina se stavlja na git repo (A6)
- praviti selekciju za sečenje sa time u vidu.

---

## Pohvale 🤌

- Preslikavanje sekvence sklapanja u `min f(c,x)` (§2.5) — to je noseća ideja rada.
- Disciplina oko jedinice troška (§3.3) i rešetke po pozivima umesto po generacijama.
- Negativni nalazi prijavljeni sami od sebe: §4.6 (konzistentnost „nije utvrđena"),
  §4.7 (pod greške), Tip IV kao „potkrepljeno, ne utvrđeno".
- Tabela 5 (fizička ispravnost) i ablacija §4.8 — obe su odgovori na pitanja koja bi
  recenzent ionako postavio.
- Interna aritmetika statistike je proverena: svi binomni `p` iz Tabele 3 se slažu
  (24/30 → 0.0014, 23/30 → 0.0052, 21/25 → 0.00091, 21/30 → 0.043, 89/115 → 3·10⁻⁹),
  kao i brojevi pokretanja (115·2 + 12 kruga = 242; 242 + 24 ablacije = 266) i keš
  (1250 generacija × 20 elitnih ≈ 24 980).

---

## A. Stvari koje pomeraju zaključak 🧨

### A1. Uokvirenje tvrdi uzročnost koju eksperiment ne izoluje

Rezultat stoji: pri istom budžetu dvonivovski `GA+CMA-ES` ubedljivo tuče monolitni `GA`
na ovom problemu. Sporno je **uokvirenje** — na nekoliko mesta rad tu razliku pripisuje
dvonivovskoj *podeli*, a poređenje menja dve stvari odjednom: podelu na nivoe i unutrašnji
optimizator. §5.1 daje razlog za sumnju da je drugi krivac pravi: van-dijagonalna energija
Hesijana 0.52–0.93, medijana 0.78 — na takvom pejzažu puna kovarijansa tuče per-node
izotropni Gaus sa ručno opadajućim σ, nezavisno od ikakve dvonivovske priče.

Ispravka je svuda uređivačka. Nijedno novo pokretanje nije potrebno.

**Najjači argument još nije iskorišćen.** U monolitnom zapisu CMA-ES se ne može pustiti, i
to strukturno: čim topološka mutacija promeni skelet, vektor `x` menja značenje (isti broj
u `ρ_a` opisuje drugu polugu, a kod `add_node` menja se i dimenzija). Kovarijansa
adaptirana nad jednim skeletom nad drugim ne važi. Prostor nad kojim se kovarijansa uopšte
može adaptirati **nastaje tek podelom na nivoe** — pa „unutrašnji optimizator je bolji" i
„podela omogućava bolji unutrašnji optimizator" nisu potpuno razdvojive tvrdnje.
Confound ostaje (IGBD isto radi, preko keša stanja po konfiguraciji), ali prigovor „pa to
je samo CMA-ES protiv Gausa" time gubi najveći deo snage. Ovo je pasus u §5.3.

**H1 se ne dira u ovom kontekstu** — operativno je napisan tačno kao ono što je mereno („dvonivovski metod
(GA nad topologijom + CMA-ES nad geometrijom) ... od monolitnog GA"). Isto i zaključak §6,
koji metode imenuje u apoziciji.

#### Mesta koja tvrde više nego što se meri

| gde | sadašnje | problem |
|---|---|---|
| naslov | „provera i proširenje ... dekompozicije" | „provera" je uzročna reč |
| apstrakt | „Ovaj rad proverava tu tvrdnju" | tvrdnja je o dekompoziciji naspram CatCMA/ICatCMA, ne naspram GA |
| §1, cilj 1 | „dvonivovska podela ... donosi prednost koju oni tvrde" | isto |
| §4.3 | „To je upravo mehanizam koji Ong i sar. navode kao izvor prednosti" | zaključuje mehanizam iz ishoda |
| §5.2 | „Tvrdnja da razdvajanje pomaže kad je sprega jaka preživljava prelazak..." | najjača rečenica u radu, najslabije pokrivena |

#### Novi okvir: sa *provere tvrdnje* na *prenos preporuke*

Rad onda tvrdi četiri stvari koje sve stoje:

1. sinteza pantografa pri fiksnom `n` **jeste** instanca njihove klase problema (§2.5) —
   samostalan doprinos, ne zavisi ni od jednog pokretanja;
2. pejzaž je jako spregnut, najmanje Tip III (§5.1);
3. konkretna dvonivovska instanca koju njihov rad motiviše ubedljivo pobeđuje na tom
   problemu, i to pod uslovima kojih u njihovim test-funkcijama nema (višeopcione
   kategoričke, neizvodljivost, 28× manji budžet);
4. dakle njihova preporuka kao **praktično uputstvo** se prenosi na stvaran problem.

Neizolovan ostaje mehanizam — i to je poštena rečenica, ne priznanje poraza.

#### Predlozi konkretnih izmena (ćirilicom, jer idu u rad)

Naslov: „провера и проширење" → **„пренос и проширење"**. Jedna reč.

Apstrakt, umesto „Овај рад проверава ту тврдњу...":

> Овај рад преноси њихову препоруку на стварни инжењерски проблем и проширује је на
> простор механизама са променљивим бројем чворова, који њихов оквир не покрива.

§5.2, umesto „Тврдња да раздвајање помаже ... преживљава прелазак":

> Препорука коју њихов рад даје — категорички део оцењивати тек пошто му се континуални
> оптимизује — на овом стварном задатку даје убедљиву предност, и то под три околности
> које њихови тест-проблеми немају: (1) вишеопционе категоричке променљиве...
> [ostatak rečenice ostaje nepromenjen]
>
> Шта од те предности долази од саме поделе, а шта од тога што подела **тек омогућава**
> унутрашњи метод са пуном коваријансом, овај рад не раздваја. Те две ствари овде и нису
> сасвим раздвојиве: у монолитном запису вектор геометрије мења значење при свакој
> тополошкој мутацији, па расподела адаптирана над једним скелетом над другим не важи.
> Прави начин да се раздвоје јесу CatCMA и IGBD као трећа и четврта тачка поређења
> (одељак 5.3).

§1 (cilj 1) i §4.3 — isto omekšavanje, formulacija po istom obrascu.

### A2. H1 nije oštro postavljena — „i/ili" treba zameniti

Problem nije što je disjunkcija logički neoboriva — oboriva je, pada kad padnu obe strane.
Problem je što je slaba na tri načina:

- daje dve šanse za pobedu, a nigde ne kaže koja je primarna;
- drugi disjunkt je skoro sadržan u prvom: ako DN završi ispod MGA, monotona kriva
  najboljeg garantuje da je negde pre kraja budžeta prošao kroz MGA-ovu konačnu grešku.
  §4.2 to i primećuje („То није друго мерење него иста чињеница"), pa disjunkcija izgleda
  kao dva uslova a nosi jedan i po;
- nigde nema praga — „niža greška" bila bi zadovoljena i sa 0.99×.

Tri izlaza; ja bih uzeo prvi, ali ti proceni. Ostala dva su takođe ok, mislim da je ovo malo stvar ukusa.

**(1) Izbaciti brzinu konvergencije iz hipoteze.** H1 = samo niža konačna greška; brzina
ostaje rezultat koji se izveštava (§4.2), ne tvrdnja koja se testira.

- *Obara se:* medijana odnosa ≥ 1, ili unutar dogovorenog pojasa šuma.
- *Menja se:* jedna rečenica u §1; §4.2 dobija uvodnu napomenu „ово није део хипотезе,
  него мерење уз њу".
- *Cena:* nikakva. Najmanje posla, najčistija tvrdnja.

**(2) Dve odvojeno oborive hipoteze.**

> **H1a.** При истом буџету позива симулатора двонивовски метод даје нижу коначну грешку
> од монолитног.
>
> **H1b.** Та предност се појављује рано: ДН достиже коначну грешку МГА на мање од
> половине буџета.

- *Obara se:* H1a kao gore; H1b ako je medijana udela budžeta ≥ 50%.
- H1b se mora meriti **necenzurisano po ishodu** — parovi u kojima DN nikad ne stigne
  ulaze cenzurisani na 100% (B3). Inače je uslovljena na ishod i prag ništa ne znači.
- *Menja se:* §1, §3.5 (kako se H1b meri), §4.2 preračunati.
- *Cena:* jedno preračunavanje iz već snimljenih `calls_curve`, bez novih pokretanja.

**(3) Zadržati „i/ili", ali dodefinisati obaranje.** Npr. „H1 се одбацује ако медијана
односа грешака пређе X **и** медијана удела буџета пређе Y."

- Najbliže sadašnjem tekstu i pošteno.
- Ali tvrdnja ostaje slaba, i mora se reći da disjunkti nisu nezavisni (druga tačka gore).
- *Cena:* dve rečenice u §1.

**Posledica koju povlači svaki izbor sa pragom:** §3.5 već ima projektno pravilo da su
odnosi 0.6–1.2× unutar šuma između seed-ova, ali se nigde ne primenjuje. Primenjeno na
Tabelu 3: E2 (0.56×) i `n=7` (0.39×) prolaze, `n=6` (0.71×) je **unutar** pojasa, `n=8`
(0.61×) na ivici. Čim H1 dobije prag, rad mora ili da prizna da je praktična značajnost
jasna u dve od četiri postavke (statistička jeste u sve četiri), ili da eksplicitno kaže
zašto se pravilo kalibrisano za pojedinačna poređenja ne primenjuje na agregate od 25–30
parova.

### A3. §3.3 je faktički netačan — ispravka je u korist rezultata

> „Neizvodljiva topološka mutacija u DN košta jedan poziv i ne dobija CMA-ES."

Tačno samo za nevažeću strukturu i za pad `to_sequence`. Dominantna korpa po sopstvenom
merenju (`docs/IZVESTAJ_FIKSNO_N.md`) je korpa **(v) — putanja degenerisana tokom punog
obrtaja, 56–85%**, koju `_make_child` ne vidi: `validate` i `to_sequence` rade na θ=0.
Takvo dete prođe, dobije zapis, i `inner_cmaes` mu vrti CMA-ES. Svi kandidati vraćaju
`PENALTY`, `record.history` se puni konstantom, `plateau_detected` puca tek kad se
napuni prozor od 10 → **~10 iteracija × λ≈11 ≈ 110 poziva po mrtvom skeletu**.

Sa 16 dece po generaciji i ~60% korpe (v), reda 1000+ poziva od ~2900 po spoljašnjoj
generaciji.

- Izmeriti tačno: dodati u `RunLog` zbir `record.calls` za zapise koji završe na `PENALTY`.
- Ako se potvrdi — to **jača** rad: DN pobeđuje trošeći trećinu budžeta na mrtve skelete.
- Očigledno pojačanje koje sledi: jedan probni poziv pre konstrukcije CMA-ES-a.

### A4. Uparenje po seed-u nije stvarno uparenje

`spawn_rng_streams(seed)` daje MGA 100 genoma, a DN 20 zapisa — različiti crteži, ništa
zajedničko osim broja. Par (MGA seed 3, DN seed 3) nema zajednički izvor šuma koji
upareni test pretpostavlja; oznake seed-ova unutar krive mogu se permutovati i dobiju se
drugi odnosi.

Uz to, 115 parova nisu nezavisni — istih 5 krivih kroz 4 postavke, efekat krive se
broji četiri puta. `p = 5.9·10⁻¹¹` je precenjen, verovatno za nekoliko redova veličine.

Dve putanje:

- **uparenje postaje stvarno** — oba metoda kreću iz istog skupa početnih topologija
  (prvih 20 od MGA-ovih 100, ili oba iz istih 20);
- **ili se odustaje od uparenja** — mešoviti model sa krivom kao slučajnim efektom, ili
  primarni test na nivou 20 ćelija (kriva × postavka) umesto 115 parova.

Efekat preživljava oboje. Broj 10⁻¹¹ ne sme da ostane.

### A5. MGA je hendikepiran na mrtvom teretu

`operators.mutate_coords` mutira **sve** čvorove `1..n−1`, uključujući one van predačkog
stabla tragača. DN ih eksplicitno izbacuje iz vektora — i rad to navodi kao svoju
odluku (§3.2, prva alineja).

Tabela 6: u E2 MGA nosi medijalno 8.5 čvorova / 5 radnih → **~40% koordinatnih mutacija
ide u čvorove koji po konstrukciji ne mogu da promene fitnes.** Nije posledica
dvonivovske podele nego nejednake implementacije, i naduvava E2 rezultat (0.56×).

U E1 je bezazleno (6/6, 7/7, 8/7) — dakle E1 je čist, E2 nije. To vredi reći, jer brani E1.

Izbor: popraviti `mutate_coords` da preskače mrtve čvorove (jedna provera) i ponoviti E2,
ili staviti asimetriju eksplicitno u §5.4. Drugo je legitimno i košta jedan pasus — ali
mora da piše. Tvrdnja „MGA mora biti najbolji mogući, a ne namerno oslabljen" (§3.1) ovako
ne stoji. U odeljku o formulacijama je dat predlog oko ovoga (C5).

### A6. Repozitorijum nije u stanju u kom se može pregledati / lako reprodukovati

Dva zahteva, oba obavezna, ali drugi je preči.

**Rad mora da imenuje repozitorijum.** Trenutno se na njega oslanja tri puta a nigde ga ne
identifikuje: §3.1 i §3.2 („у репозиторијуму *baseline*" / „*bilevel*") i §3.4 („Свако
покретање бележи git комит кода"). Treba URL + **grana** + heš komita pod kojim je puštena
objavljena matrica. Grana je bitna jer *baseline* i *bilevel* iz §3.1–3.2 zvuče kao imena
grana, a to su imena metoda — grana `baseline` ne postoji.

**Ali pre toga repo mora da se sredi**; imenovati repo koji se ne može pregledati ne pomaže
nikom. Zatečeno stanje:

- `main` je praktično prazna — samo prototip `simulator.py` i PDF predloga projekta.
- `dev`, koja je po `CLAUDE.md` radna grana, je **32 komita iza** `bilevel` i nula ispred.
- Sav kod kojim su dobijeni rezultati — režim poređenja, režim fiksnog `n`, alat za Hesijan,
  popravka nulte poluge — postoji **samo** na nespojenoj grani `bilevel`.
- `README.md` (64 KB) je formalna specifikacija problema (Deo I–IV). Nijednom ne pominje
  `run.py`, `venv` ni `pip install`; §4.3 „Репродуцибилност" govori o RNG tokovima i
  logovanju, ne o tome kako se išta pokreće. Uputstvo za okruženje postoji samo u
  `CLAUDE.md` §6, koji je pisan za agenta, ne za čitaoca.
- `tools/matrica_poredjenja.sh` jeste de-facto ulazna tačka, ali mu je podrazumevano
  `SEEDOVI="1 2 3"`, a zaglavlje opisuje matricu od 36 pokretanja. Rad koristi šest seed-ova
  (pet za `n=7`) i 242 pokretanja. Tačni pozivi kojima je objavljena matrica puštena ne
  postoje nigde.
- `results/` je u `.gitignore` — nijedan log iza brojeva iz Tabele 3 nije objavljen.
- `requirements.txt` daje donje granice, ne pinove. `bilevel._construct_cma_es` u docstringu
  eksplicitno vezuje ponašanje za `cma` 4.4.4 (`opts['randn']`, `seed=NaN`), a zahtev je
  `cma>=3.3` — reprodukcija sa drugom verzijom može tiho da se razlikuje.

Šta treba uraditi:

1. Spojiti `bilevel` u `main` tako da najsvežiji kod bude na jednom očiglednom mestu.
2. `README.md` dobija odeljak **Reprodukcija**, ili kratak odeljak sa linkom na
   `docs/REPRODUKCIJA.md`: instalacija okruženja, tačne komande za E1 i E2, očekivano
   trajanje, gde izlaz završava.
3. Zapisati tačne pozive kojima je dobijena objavljena matrica (seed-ovi, `n`, krive).
4. Pinovati verzije u `requirements.txt`, bar `cma`.
5. Zamrznuti stanje i tek onda staviti link u rad (v. ispod).

#### Zamrznuti artefakt za objavu

U radu ne sme da stoji link na nešto što se pomera. Tri opcije:

| | šta je | drži li rezultate | zaista zamrznuto |
|---|---|---|---|
| **(1)** release nad tagom | tag + prilozi | **da** — arhiva rezultata ide kao prilog, van git istorije | da |
| **(2)** tagovan komit na `main` (`rad-v1`) | jeftino, standardno, stalan link na stablo | samo ako `results/` uđe u git | praktično da (anotiran tag se po dogovoru ne pomera) |
| **(3)** grana posvećena radu (`rad-v1`), nikad se ne spaja | pregledna, čitljiva, prirodno se skalira ako bude više radova | samo ako `results/` uđe u git | **ne** — grana se po definiciji pomera, može se i force-push-ovati |

**Preporuka: (1) izgrađeno nad (2).** Anotiran tag `rad-v1` na `main`, pa release nad njim,
sa `results.tar.gz` kao prilogom. Proveriti
veličinu arhive pre nego što se odluči.

Opciju (3) uzeti samo kao dodatak ako zaista bude više radova i ako se želi grana po radu
— sama po sebi nije zamrzavanje. Ako se traži trajan, citabilan link (DOI), Zenodo se
povezuje sa GitHub release-om i pravi ga automatski.

---

## B. Predlozi za merenja / preračun 🤓

### B1. Ciljna kriva sa poznatim rešenjem  ← najveća isplata po uloženom

Nasumičan validan mehanizam pri `n=6` i `n=8`, njegova putanja kao cilj. Pod je **0 po
konstrukciji**. Zatvara pitanje iz §4.7 („rešenje ne postoji" vs „ne nalazimo ga") bez
ijedne nove linije koda, i prvi put daje apsolutnu meru koliko daleko od optimuma
metodi staju.

Isto rešava blokator 6.1 za Jansena: objavljene dužine Jansenove noge kroz *sopstveni*
simulator daju tačke. Ne treba „pouzdan spoljašnji izvor uporediv sa literaturom" — treba
mehanizam.

### B2. Koliko skeleta MGA stvarno poseti

§4.3 poredi 125 000 *jedinki* sa 580 *skeleta*. Nije isto — većina MGA jedinki deli
skelet sa roditeljem (`p_topo = 0.30`). Hešovati `skeleton_of(to_sequence(...))` i
prebrojati jedinstvene. Ako ispadne 2 000 a ne 125 000, §4.3 se preokreće i postaje
zanimljiviji, ne slabiji.

### B3. Ubrzanje iz §4.2 je uslovljeno na ishod

18% budžeta se računa samo na 89 parova u kojima DN pobeđuje — selekcija na zavisnu
promenljivu. Cenzurisana verzija preko svih 115 (parovi gde DN nikad ne stigne
cenzurisani na 100%), ili simetrična: udeo budžeta na kom *svaki* metod stigne do
konačne greške onog drugog. Sve postoji u snimljenim `calls_curve` — nula novih pokretanja.

### B4. Više seed-ova

27 minuta po pokretanju × 8 jezgara → 200 pokretanja stane u jednu noć. Dvadeset seed-ova na
`n=7` i na dve sporne krive uklanja najveću slabost iz §5.4.1 i izjednačava sa Ong i sar.

Usput: zašto `n=7` ima pet a ne šest seed-ova? Ako je pokretanje izgubljeno, to mora da piše.

### B5. Za Tip IV koristiti ono što alat već računa

`tools/analiza_hesijana.py` već vraća `kappa_q` i `eff_dim`, a rad izveštava samo raspon
`λ_max`. `λ_max` i posle kanonizacije razmere zavisi od broja aktivnih dimenzija, pa
26.6× meša i to.

Merenje koje tvrdnju može da zatvori: kontrolisan niz skeleta (fiksno `n`, ista kriva,
isti broj aktivnih dimenzija, sistematski menjan par oslonaca) + `eff_dim`/`kappa_q`.
Režim fiksnog `n` to već omogućava (`OPEN_QUESTIONS` 7.5).

### B6. Ablacija (§4.8) nije simetrična po budžetu

`baseline.evolve` skalira koordinatnu mutaciju po **udelu** potrošenog budžeta
(`fraction = spent / total_budget`, pa `k = k_start·(k_end/k_start)^fraction`). Sa duplim
budžetom MGA na 100 000 poziva stoji na sredini rasporeda `0.15 → 0.02`, a u osnovnoj
postavci na istom mestu već na kraju — postavka **V** za MGA menja i budžet i raspored
mutacije. DN nema nijedan raspored vezan za `total_budget` (unutrašnji CMA-ES ne koristi
`k`), pa je za njega ista postavka čista.

Zaključak §4.8 („МГА је исцрпео оно што може, а ДН још није") leži tačno na toj
asimetriji. Superelipsa, gde MGA sa duplim budžetom ispada **gori** (1.88×), verovatnije
je to nego rasipanje između seed-ova kako rad tumači.

Popravka: ponoviti postavku V za MGA sa `k` vezanim za apsolutan broj poziva osnovne
postavke, ili prosto sa `k_start = k_end`. Šest pokretanja (2 krive × 3 seed-a).

---

## C. Preciznost formulacija 📏

| # | mesto | šta |
|---|---|---|
| C1 | Tabela 5 | Ugao prenosa je preklopljen preko 90° (`arcsin`). „10.5°" znači 10.5° **ili** 169.5°. Navesti da se meri odstupanje od mrtve tačke. |
| C2 | §2.2 | „zapis je jednoznačan" tačno je u smeru genom → mehanizam — to je ono što treba za determinističku rekonstrukciju — ali je napisano tako da se čita kao bijekcija. Obrnuti smer ne važi: mrtav teret i permutacija međusobno nezavisnih čvorova daju više zapisa za isti mehanizam. |
| C3 | Tabela 1 (§2.5) | `\|C\|` broji zapise, ne različite mehanizme. Mrtav čvor na poziciji `k` sažima faktor `2·C(k,2)` — od 6 (`k=3`) do 42 (`k=7`) pri `n=8`; §5.1 meri da pri `n=8` sva osam čvorova rade u samo 25.5% generacija. Neizvodljivost tu ne ulazi — ona zavisi od `x`, ne od `c` samog. Možda pokriti fusnotom. |
| C4 | §2.1 vs §2.5 | „čvorovi 0 i 1 su nepokretni" naspram položaja čvora 1 u vektoru `x`. Nepokretan tokom obrtaja, ali projektna promenljiva. |
| C5 | §3.1 | „MGA mora biti najbolji mogući" je suviše jako (v. A5). Reformulisati u „standardan GA po uzoru na Cabrera i sar. (2002)". |
| C6 | Tabele 4 i 7 | Različita agregacija (odnos medijana po ćeliji naspram medijane uparenih odnosa) daje različite brojeve za istu krivu — elipsa 0.55–0.69 u T4, 0.44 u T7. Fusnota, inače izgleda kao greška. |
| C7 | apstrakt, §1 | „Provera" je prejako: Ong i sar. porede sa CatCMA/ICatCMA, ne sa GA. Ovo je provera **izvedene** tvrdnje s drugim osnovnim metodom. Stoji u §5.3 — treba i napred. |
| C8 | §5.4 | Ravna kazna 1e9 naspram citata Sleesongsom i Bureerat (2018), koji je upravo o rukovanju ograničenjima. Rang-bazirani CMA-ES na platou kazne nema šta da nauči (v. A3). Jedna rečenica. |
| C9 | §5.4.7 | Reći koliko je pokretanja palo na starom kodu i bilo ponovljeno, ne samo da „popravka menja tok samo tamo gde je padalo". |
| C10 | Tabela 2 | E2 je ispred E1, a naracija ide provera → proširenje. Obrnuti. |
| C11 | ceo rad | Termin za jedno izvršavanje eksperimenta nije ujednačen: „прогон" i „покретање", naizmenično za istu stvar. `docs/` i `README.md` su odavno slegli na „покретање". Ujednačiti — predlog „покретање", jer ne nosi drugo značenje; ako ostane „прогон", onda dosledno i u radu i u `docs/`. |

---

## Predloženi redosled (paušalno) 🗓️

1. **A6** — sređivanje repoa; obavezno, ne blokira nijednu drugu stavku, može paralelno.
2. **A3** — greška u tekstu, popravlja se odmah i verovatno jača rezultat.
3. **A2** — izbor oblika hipoteze; gate za §1 i §4.2, pa ide pre pisanja rezultata.
4. **B1** — kriva sa poznatim rešenjem.
5. **A4** — statistika; ne traži nova pokretanja.
6. **B3** — cenzurisano ubrzanje; takođe bez novih pokretanja.
7. **A1** — preformulacija uokvirenja + pasus o nerazdvojivosti u §5.3; bez novih pokretanja.
8. **A5** — politička odluka (popraviti baseline i ponoviti E2, ili napisati asimetriju).
9. **C** — celina, pri finalnom prolazu kroz tekst.

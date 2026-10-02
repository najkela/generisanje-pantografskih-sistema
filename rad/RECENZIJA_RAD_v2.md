# Recenzija — `RAD_v2.md` (mentorske beleške)

Druga verzija rada, 30 septembar. Beleške nastale pregledom teksta rada i implementacije u `generisanje-pantografskih-sistema/` (grana `bilevel`, komit `47eb676`).

## Čistoća rezultata

### Heuristički prag
Praktični prag od 0.6x - 1.2x nije vredan truda u samom radu. Ako ga izbacimo efektivno ništa ne gubimo, tekst postaje nešto kraći (pa možeš da ubaciš nešto drugo ako hoćeš) a smanjuje se i konfuzija. Trenutno mi ide na živac što je 0.62x unutar tog opsega, i mi znamo da je to ok jer se odnosi na grupni rezultat, ali sad se treba ograđivati od toga, objašnjavati itd itd. Prag je praktični, ne statistički, tako da rezultat apsolutno može da stoji bez njega. n=6 i n=8 su jasno na granici, zato što im je 95% interval do 0.97 i 0.94 - skoro 1. Meni je tu odmah jasno da su oni tanki. Pored toga, taj praktični prag je interna heuristika, recenzent nema kako da je validira.

Ako nam je stalo da pokrijemo praktičnu značajnost u radu, to i dalje možemo da uradimo kroz pojas šuma i da bude sasvim validno. Pod greške zavisi od n — na superelipsi 3.5·10⁻³ pri n = 6 naspram 2.0·10⁻³ pri n = 8, dakle ceo jedan dodatni čvor vredi 1.75×. Prema tome merilu je 0.62× krupno. Ali ovo je lepa stvar koja se dodaje ako ima mesta, a ne teret na statistički metod.

Šta se menja (otprilike)
1. H1. Sada glasi „одбацује се ако медијана односа грешака не сиђе испод пројектног појаса шума". Zamena je standardna i jača: odbacuje se ako interval poverenja medijane odnosa obuhvata 1.
2. „Статистичка обрада". Poslednja rečenica o pragu se briše. Ništa je ne zamenjuje.
3. Rezultati. Umesto pozivanja na pojas: „…предност постоји у све четири поставке, али интервали за n = 6 (0.46–0.97) и n = 8 (0.37–0.94) скоро додирују 1 — доследност кроz поставке јача је од величине ефекта у две од четири." Isti nalaz, bez uvođenja aparata.
4. Apstrakt se sam ograđuje ako uz 0.62× nosi (95% 0.40–0.76)

### Ćelije vs seedovi
Definitivno proceni sam da li vredi praviti ovu promenu, moguće da je previše pedantno. Malo je čudno što tabela 1 meša poređenje po seed-ovima u pojedinačnim eksperimentima i po ćelijama u totalu. U statističkoj obradi prvo kažemo da je poređenje po seed-ovima problematično, i zato nam je glavi rezultat po ćelijama, a onda kasnije u rezultatima imamo tvrdnju „Предност постоји у све четири поставке засебно", ali ta prednost je po meri koja nam je slabija. Ako možemo ova merenja da zemenimo sa ćelijskim, ceo taj segment je čistiji. Opet, kažem, možda preterujem.

## Sređivanje teksta

Ovo su samo artefakti koji su ostali od sređivanja iz v1 u v2. Treba ispraviti, ali jasno da je ovo samo stvar peglanja.
* „Тренд по величини проблема није монотон (0.64× / 0.34× / 0.63× за n = 6/7/8)" (rekao bih da je ovo ostalo iz v1)
* Postavka kaže 242. Rezultati kažu „нема ни у једном од 266 покретања". Tih 266 je u v1 uključivalo 24 ablaciona pokretanja — a ablacije u v2 više nema. Uz to, pošto je krug „изостављен из свих бројева", analizirani skup je zapravo 230.
* Treba prilagoditi se IEEESTEC normama, ali kontam da ti je to u procesu (abstrakt je sadržaj, autorski blokovi, ključne reči, numerisanje podnaslova)
* Pasus "Шта јесте, а шта није изоловано." mislim da nije loše da prepišeš ručno. Trenutno sadrži neke konstrukcije iz mojih komentara koje su neformalne. A i pasus deluje malo kao usmena odbrana projekta ili jedna strana razgovora (ne razdvaja ... ipak nisu sasvim razdvojive ... razdvaja ih tek). Lepše je da ovaj deo bude napisan kao niz konstatacija. Ispod u numerisanoj listi imaš neke vodilje (ne moraš da ih pratiš ako ne želiš) kako da pristupiš prepisivanju.

1. naslov kao konstatacija, ne kao pitanje;
2. tri rečenice, po jedna: šta poređenje menja istovremeno; zašto monolitna kontrola sa punom kovarijansom nije konstruktibilna; da CatCMA izoluje podelu a IGBD oblik spoljašnjeg nivoa;
3. nijedno „прича", „ипак", „самим";
4. bez „Оне" — imenovati o čemu je reč;
5. jedna rečenica zašto argument predviđa da CatCMA ovde slabo prolazi.

# Рецензија четврте верзије рада (06.10.2026.)

Текст рецензије, преписан дословно како је стигао.

```
Generalni utisak: kvalitetno istrazivanje sa dosta formalizama i jako dobrim pristupom i strukturom. Autor treba da prezentuje ove rezultate, medjutim, radu su potrebe sitne popravke:

Pismo: zbornik radova se izdaje na latinici pa pismo mora da bude konzistentno sa tom cinjenicom. Mislim da ce recenzenti ovo traziti da se prebaci na latinicu pa treba biti spreman na to. 

Apstrakt: GA i CMA-Es treba danpisati punim imenom. 

Uvod:

Синтеза механизама са променљивим бројем зглобова
излази из тог оквира, јер број чворова мења и dc и
dx
. Рад има два циља: (1) пренос — показати да је
синтеза пантографа при фиксном n инстанца њихове класе
проблема и да препорука која из тог рада следи даје
предност и на стварном задатку; (2) проширење — да
двонивовска подела са популацијским спољашњим нивоом
ради и када се број чворова мења током претраге.

Ovde se prvi put pominje termin pantografa, koji je povezan sa opisanim sistemom iznad, ali se nigde ne kaze to eksplicitno. Dodatno, nisam siguran jel pantograf dobar termin uopste, mozda usvojiti i konzistentno koristiti naziv iznad. 

Dalje, prenos - bolje reci "pokazati da je sinteza pantografa sa fiksnim n instanca klase problema iz [ref] ...". 

Frekvencija uzorkovanja je stavljena na 720 uglova u opsegu [0, 360) - sto je 0.5 stepeni razmaka, sto je sasvim precizno. Medjutim, nije navedeno sto bas toliko uzorkovanje, sto ne vece ili manje? Jel ovo neki standard iz literature, jel ovo slobodna procena autora upotrebom common sense ili je autor sproveo neko preliminarno istrazivanje da utvrdi da je potrebno 720 uglova, a ne recimo 180.

E sad, ogranicenja i rukovanje ogranicenjima - ovde je lepo napomenuta razlika u odnosu na previous work sto je dobro, ono sto nije jasno je zasto su uvedena drugacija ogranicenja? Da li je autor primetio nesto sto prethodni autori nisu primetili? 

Preslikavanje na min f(c,x) - ovde mi nije bilo bas najjasnije objasnjenje. Potrebna je malo bolje pojasniti ako je moguce sa datim ogranicenjem na broj strana, u suprotnom, za konferencijski rad na Petnickoj konferenciji to treba objasniti detaljnije i sveobuhvatnije. Dodatno pitanje, nevezano za rad, jel je ovde rec o redukciji? Neka je I instanca klase problema A i neka je S resenje. Onda za (f,h) kazemo da redukuje A na B ako je f funkcija koja preslikava instancu I problema A na instancu f(I) problema B i h koje preslikava resenje S instance f(I) problema B na resenje h(S) instance I problema A. Ako da, onda samo definisi f i h i pokazi da vaze ova dva tvrdjenja. 

Svuda u radu postoje pasusi koji imaju jednu-dve boldovane reci kao prvu recenicu: npr: "Brzina. Preko svih 115 parova...." - cemu ovo "Brzina." kao jedna boldovana rec u jednoj recenici na pocetku pasusa. 

"фиксном n ДН побеђује са мање радних чворова (5–6 наспрам 6–7)." - izbaciti termin pobedjuje - ostvaruje manju gresku.

Mesa se pasiv i izlaganje u prvom licu mnozine. 

U diskusiji se koristi fitness za funkciju cilja, a svuda pre toga se koristi termin funkcija cilja. 

Merenje hesijana je radjeno na razlicitim pokretanjima, sa razlicitom implementacijom iz rada, to treba ponoviti sada jer nije konzistentno ili izbaciti pominjanje hesijana u radu generalno. Ovo da svaki log ima svoj komit ne znam ni sta je ni zasto tu stoji i koja je vrednost te informacije. 

"Наставак: CatCMA и IGBD као трећа и четврта тачка поређења, што једино раздваја допринос поделе од доприноса унутрашњег оптимизатора; циљна крива са познатим решењем, која под грешке чини нулом по конструкцији; контролисан низ скелета, који затвара питање типа IV; и тежи циљеви — слова и Јансенова крива." - ovde bukvalno nista nije jasno

BTW: tokom citanja rada sam imao osecaj da je AI-generated, pogledati kakva je politika vezano za to na IEEESTEC. Ako je koriscen alat baziran na LLM za pisanje rada, onda to treba naglasiti. Recenzenti mogu da odbiju rad ako vide da je AI generisan, a ta napomena nije navedena. 

BTW2: Moram da pohvalim izuzetno lepu statisticku obradu i formulaciju hipoteza kao i mogucnost reprodukcije (github sa checkpointovanim komitom).

```

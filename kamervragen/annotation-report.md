# Pilootannotatie: bevindingen en aanbevelingen

**Piloot:** 50 items, enkelvoudig geannoteerd (geen IAA in deze ronde)
**Schema versie:** 0.1

---

## Labelfrequenties

| Label | N | % |
|---|---|---|
| INF | 26 | 52% |
| VEC | 9 | 18% |
| LEG | 7 | 14% |
| POL | 6 | 12% |
| DEF | 1 | 2% |
| MET | 1 | 2% |
| AGN | 0 | 0% |

**Zekerheid annotator:** H=37, M=11, L=2

---

## Wat werkt

### 1. INF, VEC en LEG zijn goed herkenbaar

De kern van het schema — het onderscheid tussen feitelijke informatievragen (INF), verantwoordingsvragen (VEC) en normatieve vragen (LEG) — werkt goed in de praktijk. De labelgrens is helder geformuleerd en in de meeste gevallen direct af te leiden uit de vraagtekst:

- INF: "hoeveel", "welke", "wat waren de kosten van"
- VEC: "waarom heeft de Minister niet", "waarom is er geen vooruitgang geboekt"
- LEG: "welke verbeteringen acht u noodzakelijk", "welke lessen trekt u"

De drie categorieën samen omvatten 84% van de vragen en zijn voldoende gediversifieerd om inhoudelijk interessante uitspraken over te doen.

### 2. POL is onderscheidbaar maar vraagt oefening

De POL-categorie (6 items, 12%) is conceptueel scherpst te onderscheiden van VEC wanneer de vraag *leidend* is geformuleerd of *ideologisch geladen* (items 31, 34): de parlementariër voegt zelf een interpretatie toe en vraagt de Rekenkamer die te bevestigen. Dat is anders dan een VEC-vraag waarbij de Rekenkamer de tekortkoming heeft geconstateerd en de parlementariër uitleg vraagt.

Het VEC/POL-onderscheid loopt goed parallel aan het oppositie/coalitie-contrast uit Van Druenen & Keulen: POL-vragen zijn vaker politiek opgeladen en afkomstig van parlementariërs die het kabinet bestrijden.

### 3. +CAB is bruikbaar als modificator

De +CAB-modificator duikt organisch op bij VEC-vragen (en bij twee POL-vragen) en markeert precies wat de literatuur aanduidt als "fire alarm"-gebruik: de parlementariër richt de vraag op de minister, niet op de Rekenkamer.

---

## Wat niet werkt

### 4. INF is een vuilnisbak (52% — te groot)

Het grootste probleem. INF omvat minstens drie functioneel verschillende vraagtypes die beleidsmatig anders werken:

| INF-subtype | Omschrijving | Voorbeelditems |
|---|---|---|
| **INF-CIJ** | Feitelijke data, cijfers, overzichten | 5, 11, 24, 27, 29, 44 |
| **INF-OOR** | Causale verklaring of evaluatief oordeel over bevinding | 3, 32, 39, 43, 47 |
| **INF-SCO** | Scope-extensie: geldt bevinding ook voor X? | 22, 30 |

INF-CIJ-vragen zijn het meest instrumenteel van aard (Weiss: instrumenteel gebruik). INF-OOR-vragen hebben een conceptueler karakter: de parlementariër vraagt om een interpretatie, niet louter een getal. INF-SCO-vragen zijn potentieel strategisch: ze proberen de geldigheid van een bevinding uit te breiden naar niet-onderzochte domeinen.

**Aanbeveling:** Splits INF in INF-CIJ en INF-OOR als minimale aanpassing. INF-SCO kan als modificator `+SCO` bij beide subtypes.

### 5. DEF en MET zijn nagenoeg afwezig (elk 1 geval)

In de piloot: DEF=1, MET=1. Dat is te weinig om als zelfstandige categorieën te functioneren in een classifier. Er zijn twee mogelijke verklaringen:

1. De vragen zijn *feitelijke* vragen (de kategorie heet "lijst van vragen en antwoorden"). Methodologische discussie vindt plaats in commissievergaderingen, niet in deze formele vragenlijst.
2. Vragen die methodologisch bedoeld zijn, worden toch als INF-OOR gecodeerd ("hoe kan X verklaard worden?" is zowel MET als INF-OOR).

**Aanbeveling:** Verwijder DEF en MET als zelfstandige labels. Voeg een modificator `+MET` toe voor vragen die uitdrukkelijk de methode bevragen, als aanvulling op het primaire label.

### 6. AGN is niet aangetroffen (0 gevallen)

Nul AGN-vragen in 50 items. Dat kan steekproeftoeval zijn — het is een klein aandeel van de totale dataset. Maar het kan ook zijn dat vervolgonderzoek-verzoeken in de dataset zeldzaam zijn: parlementariërs beseffen dat de Rekenkamer haar eigen agenda bepaalt.

**Aanbeveling:** Laat AGN in het schema maar verwacht een lage prevalentie (< 5%). Vermeld dit in de codeerhandleiding als waarschuwing zodat annotators niet "te snel" naar AGN grijpen.

### 7. Grensgevallen VEC/POL zijn tijdrovend

Vier van de elf VEC+POL-gevallen hadden middel-zekerheid. De moeilijkste grensgevallen:

- **Item 16** (LEG of POL?): "Wat is het oordeel van de Rekenkamer over het feit dat meer zorguitgaven uit de begroting gefinancierd worden" — de vraag is feitelijk leidend maar niet overduidelijk partijpolitiek. Gecoded als POL maar LEG verdedigbaar.
- **Item 38** (VEC of INF?): "Kan het gevolg van de 75%-drempel zijn dat een klas geen activiteiten onderneemt?" — impliciete contestatie van een beleidsdrempel. Gecoded als VEC maar de tekst heeft geen expliciete normatieve marker.
- **Item 40** (VEC of POL?): "Wat vindt u ervan dat de Minister uw aanbeveling niet overneemt?" — is dit VEC (kabinet ter verantwoording) of POL (AR wordt ingezet als bondgenoot)? Gecoded VEC.

**Aanbeveling:** Voeg een concrete beslisboom toe aan de richtlijnen: als de vraag begint met "onderschrijft u", "bevestigt u", "vindt u in tegenstelling tot" → POL. Als de vraag begint met "waarom heeft de Minister niet" → VEC.

### 8. PDF-artefacten maken ~5% van de vragen oncodeerbaar

Items 18 en 33 zijn te sterk afgekapt of vervormd door PDF-extractie om betrouwbaar te coderen. Item 18 begint midden in een zin; item 33 is onleesbaar. Bij een grotere annotatie-batch moet een `?`-categorie worden ingevoerd en deze items moeten voor de classifier worden uitgesloten of handmatig worden geredresseerd.

---

## Herzien categorieënvoorstel

Op basis van de piloot, als minimale aanpassing op het bestaande schema:

| Code | Naam | Verandering t.o.v. v0.1 |
|---|---|---|
| **INF-CIJ** | Feitelijke/cijfermatige informatievraag | Opsplitsing van INF |
| **INF-OOR** | Verklarende/evaluatieve informatievraag | Opsplitsing van INF |
| **VEC** | Verantwoording & contestatie | Ongewijzigd |
| **POL** | Politiek-strategisch gebruik | Ongewijzigd |
| **LEG** | Legitimatie & normstelling | Ongewijzigd |
| ~~DEF~~ | ~~Definitie & afbakening~~ | Verwijderd → modificator `+DEF` |
| ~~MET~~ | ~~Methodologische verantwoording~~ | Verwijderd → modificator `+MET` |
| **AGN** | Agendering & escalatie | Gehandhaafd, verwachte prevalentie < 5% |

Modificatoren:

| Code | Naam | Verandering |
|---|---|---|
| `+CAB` | Kabinetsadressering | Ongewijzigd |
| `+GAP` | Dataleemte | Ongewijzigd |
| `+COMP` | Vergelijking | Ongewijzigd |
| `+FUP` | Follow-up verzoek | Ongewijzigd |
| `+DEF` | Definitie bevraagd | Nieuw |
| `+MET` | Methode bevraagd | Nieuw |
| `+SCO` | Scope-extensie | Nieuw |

---

## Verwachte labelfrequenties bij volledige dataset (projectie)

Op basis van de piloot, extrapolatie naar 1148 vragen:

| Label | Verwacht N |
|---|---|
| INF-CIJ | ~350 |
| INF-OOR | ~250 |
| VEC | ~205 |
| LEG | ~160 |
| POL | ~140 |
| AGN | ~40 |
| ? (oncodeerbaar) | ~20 |

---

## Aanbevelingen voor stap 3 (supervisede taxonomie)

1. **Gebruik het herziene 5-labels schema** (INF-CIJ, INF-OOR, VEC, POL, LEG) voor de supervised classifier; houd AGN als restcategorie.
2. **Streefschema voor handmatige annotatie:** 150 items total, minimaal 25 per label (behalve AGN: 10 volstaat). Stratificeer op rapport_type (jaarverslag vs. thematisch) en vergaderjaar.
3. **Voer IAA uit op de VEC/POL-grens** als eerste prioriteit; bereken κ per labelgrens, niet alleen overall.
4. **Overweeg een LLM-assisted aanpak** voor INF-CIJ: deze vragen zijn zo uniform dat een eenvoudige heuristische classifier (bevat de vraag een bedrag, percentage of specifiek jaar?) al ~80% recall haalt, zonder handmatige labels.
5. **Repareer de PDF-extractie** voor items met lage betrouwbaarheid vóór de productie-annotatie. De `questions`-kolom bevat geregeld artefacten uit de paginanummering en koppen.

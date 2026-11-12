# Voorbereidende opdracht week 3: Classificatie & Gradient Descent (~2h)

## 1. Classificeren met een sigmoïdefunctie

### De functie $$\sigma(z)$$

In het hoorcollege hebben we gezien dat we een sigmoïdefunctie kunnen gebruiken om
een classificatie te modelleren. De functie daarvoor is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

In afbeelding 1 zie je de grafiek van deze functie.

![Grafiek van de sigmoïdefunctie. De rode stippellijn markeert het omslagpunt voor
de classificatie.](standard_sigmoid.png)

*Afbeelding 1. De grafiek van de sigmoïdefunctie. De rode stippellijn markeert de
σ-waarde waar het omslagpunt voor de classificatie plaatsvindt.*

Een voordeel van de sigmoïdefunctie ten opzichte van een trapfunctie is dat de
sigmoïdefunctie differentieerbaar is. Daardoor kunnen we ook de loss functie
differentiëren. De sigmoïdefunctie heeft daarnaast andere nuttige eigenschappen.

We kunnen de functie bijvoorbeeld gebruiken om te voorspellen of een paddenstoel
giftig is. Anders dan een trapfunctie neemt de sigmoïdefunctie niet alleen de waarden
0 en 1 aan. De functiewaarden liggen namelijk in het open interval $$(0,1)$$. De functie maakt
een vloeiende overgang van waarden dicht bij 0 naar waarden dicht bij 1. Precies deze vloeiende
overgang maakt deze functie differentieerbaar.

We kunnen de sigmoïdewaarde, $$\sigma(z)$$, omzetten in een classificatie door de volgende beslisregel te gebruiken:

$$
\begin{cases}
1 & \text{als } \sigma(z)\geq 0.5,\\
0 & \text{als } \sigma(z)<0.5.
\end{cases}
$$

Bij $$z=0$$ geldt $$\sigma(z)=0.5$$. Daarom is $$z=0$$ het omslagpunt waarop de voorspelde
klasse verandert. Dit omslagpunt, wat de grens is tussen de twee klassen, heet de *decision boundary* (beslissingsgrens).

Omdat de sigmoïdewaarden tussen 0 en 1 liggen, kunnen we ze interpreteren als waarschijnlijkheden.
Een kleine sigmoïdewaarde, $$\sigma(z)$$, duidt op een kleine voorspelde kans om tot klasse 1 te behoren; een
grote waarde duidt op een grote voorspelde kans. In de buurt van $$z=0$$ is de sigmoïdewaarde
ongeveer 0.5, wat overeenkomt met een waarschijnlijkheid van rond de 50% dat een z-waarde bij
de ene of de andere klasse behoord. Met een trapfunctie kan je deze onzekerheid rondom het
omslagpunt (de *decision boundary*) niet weergeven.

### Classificeren met een sigmoïdefunctie

Bij classificatie is $$z$$ in de sigmoïdefunctie een lineaire functie van de kenmerken. Stel
dat we voorspellingen doen op basis van één kenmerk $$x$$. Dan geldt:

$$
z=wx+b,
$$

waarbij $$w$$ de *weight* en $$b$$ de bias is. Door de waarden van $$w$$ en $$b$$
aan te passen, veranderen we de positie en de steilheid van de sigmoïdecurve.

De beslissingsgrens ligt waar:

$$
0=z=wx+b.
$$

Als we hierin $$x$$ vrij maken krijgen wij als *decision boundary*:

$$
x=-\frac{b}{w}.
$$

Stel dat we bijvoorbeeld $$w=2$$ en $$b=4$$ kiezen, dan is de beslissingsgrens $$x=-2$$. Afbeelding 2
toont de bijbehorende sigmoïdecurve voor deze weight en bias. De overgang tussen de klassen is steiler
geworden en de beslissingsgrens is opgeschoven naar links (in vergelijking met afbeelding 1).

![Grafiek van de sigmoïdefunctie voor z = 2x + 4. De rode stippellijn markeert de $$\sigma$$-waarde
bij het omslagpunt; de groene lijn markeert de beslissingsgrens.](specifiek_sigmoid.png)

*Afbeelding 2. De sigmoïdefunctie voor $$z=2x+4$$. De rode stippellijn markeert de $$\sigma$$-waarde
bij het omslagpunt; de groene lijn markeert de beslissingsgrens.*

In het voorbeeld uit het hoorcollege voorspelden we irissoorten op basis van twee kenmerken:
*petal width* (kroonbladbreedte) en *petal length* (kroonbladlengte). De lineaire functie waarop
we de sigmoïdefunctie toepassen, heeft dan de volgende vorm:

$$
z=w_1x_1+w_2x_2+b.
$$

De beslissingsgrens is dan niet een enkel omslagpunt, maar een lijn in de tweedimensionale
ruimte van kenmerken.

### Opgave 1

- **i.** Laat zien dat de beslissingsgrens voor

$$
z=w_1x_1+w_2x_2+b
$$

wordt beschreven door de lineaire functie

$$
x_2=-\frac{w_1}{w_2}x_1-\frac{b}{w_2}.
$$

**Tip:** de decision boundary ligt waar $$z=0$$ is.

- **ii.** Stel dat de parameters zijn:

$$
w_1=-2,\qquad w_2=1,\qquad b=-1.
$$

Bepaal de decision boundary van deze classificatie en teken deze in afbeelding 3
(hoef je niet online in te leveren). Wat kun je zeggen over de "kwaliteit" van de classificatie?

- **iii.** Waar zou de decision boundary in afbeelding 3 moeten liggen om de dataset
goed te classificeren? Teken ook deze grens in het diagram en bepaal bij benadering de formule
voor $$z$$ die bij jouw lijn hoort. Wat kun je zeggen over de loss van jouw model? Kun je voor deze dataset een loss van 0 behalen?

![Dataset voor de classificatieopgave.](decision_boundary_exercise.png)

*Afbeelding 3. De dataset voor de classificatieopgave.*

## 2. Gradient descent

Stel dat we de volgende kleine dataset hebben. Voor vier studenten kennen we het gemiddelde aantal
uren dat ze per week aan hun huiswerk besteden en of ze het vak halen. De vier datapunten zijn
weergegeven in afbeelding 4.

| Datapunt | Huiswerktijd per week (uur), $$x$$ | Vak gehaald, $$y$$ |
| ---: | ---: | ---: |
| 1 | 2 | 0 |
| 2 | 2,5 | 0 |
| 3 | 6 | 1 |
| 4 | 8 | 1 |

![Visualisatie van de dataset.](mini_dataset.png)

*Afbeelding 4. Visualisatie van de dataset.*

We willen een classificatiemodel trainen dat op basis van de wekelijkse huiswerktijd voorspelt
of nieuwe studenten het vak halen. Net als in het hoorcollege gebruiken we een lineair model met
een sigmoïdefunctie. We gebruiken in dat geval één invoerkenmerk: het gemiddelde aantal uren dat een student
per week aan huiswerk besteedt. Ons model heeft daarom twee parameters, $$w$$ en $$b$$:

$$
\hat{y}=\frac{1}{1+e^{-z}},
\qquad z=wx+b.
$$

We willen hiervoor de optimale waarden van de parameters $$w$$ en $$b$$ bepalen, zodat het model de
studenten zo goed mogelijk  classificeert. Dat is een optimaliseringsprobleem: we zoeken de
parameterwaarden waarvoor de fout op de trainingsdata zo klein mogelijk is. Oftewel, het minimum van
de loss functie, in dit geval de squared error:

$$
\mathrm{Loss}(w,b)=\frac{1}{2}\sum_{j=1}^{N}(y_j-\hat{y}_j)^2.
$$

Hierbij is $$j$$ de index van een datapunt, $$y_j$$ de werkelijke klasse en $$\hat{y}_j$$ de
voorspelling van het model voor dat datapunt.

Zoals we in het hoorcollege hebben besproken, kunnen we voor een model met een sigmoïdefunctie de
afgeleiden naar de parameters bepalen. Het resulterende stelsel is echter lastig rechtstreeks naar
$$w$$ en $$b$$ op te lossen. Daarom gebruiken we een numerieke methode om de loss te minimaliseren: *gradient descent*.

Bij deze methode beginnen we met een gok (initialisatie) voor de parameterwaardes en passen we die herhaaldelijk
aan in de richting waarin de loss afneemt. Het algoritme staat hieronder nog een keer weergegeven.

### Gradient-descentalgoritme

- **Initialize** $$w$$ en $$b$$
- **While** het minimum niet is bereikt:
  - $$w := w - \alpha \frac{\partial \mathrm{Loss}}{\partial w}$$
  - $$b := b - \alpha \frac{\partial \mathrm{Loss}}{\partial b}$$

Hierbij is $$\alpha$$ de *learning rate* (leersnelheid). Deze bepaalt hoe groot de stappen zijn die
we zetten. In dit voorbeeld gebruiken we $$\alpha=1$$, deze kunnen we dus nu negeren in het algoritme.

Voor de gemiddelde gekwadrateerde fout hierboven zijn de partiële afgeleiden:

$$
\frac{\partial\mathrm{Loss}}{\partial w}
=\frac{1}{N}\sum_{j=1}^{N}(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j)x_j,
$$

$$
\frac{\partial\mathrm{Loss}}{\partial b}
=\frac{1}{N}\sum_{j=1}^{N}(\hat{y}_j-y_j)\hat{y}_j(1-\hat{y}_j).
$$

Voor elk datapunt $$j$$ is $$y_j$$ de werkelijke klasse en is $$\hat{y}_j$$ de voorspelling
van de sigmoïdefunctie die wij proberen te opimaliseren, oftewel:

$$
\hat{y}_j=\frac{1}{1+e^{-z_j}},
waarbij \qquad z_j=wx_j+b.
$$

Voor een gegeven keuze van $$w$$ en $$b$$ kunnen we de partiële afgeleiden van de loss met deze
dataset berekenen.

### Opgave 2

- **i.** Stel dat we beginnen met de parameters $$w=2$$ en $$b=-3$$. Bepaal voor deze waarden de *decision boundary*
en teken deze in afbeelding 4. Wordt de trainingsdataset met deze grens al goed geclassificeerd?

- **ii.** In de tabel hieronder is het Gradient Descent algoritme gedurende vier updates uitgevoerd.
Afbeelding 5 toont hoe de sigmoïdefunctie zich steeds beter aanpast aan de trainingsdata.
Bereken voor de update van iteratie 4 naar iteratie 5 zelf de partiële afgeleiden van de loss naar
$$w$$ en $$b$$ en de nieuwe parameterwaarden. Vul de vraagtekens in de tabel in. Bepaal vervolgens
de nieuwe beslissingsgrens. Is het model voldoende geoptimaliseerd voor de trainingsdata?

- **iii.** Afbeelding 6 toont testdata voor 50 andere studenten: hun wekelijkse huiswerktijd en
of ze het vak hebben gehaald. Teken de door jou berekende beslissingsgrens in de afbeelding.
Is het model voldoende getraind?

| Iteratie | $$w$$ | $$b$$ | Loss | $$\partial L/\partial w$$ | $$\partial L/\partial b$$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2,000000 | -3,000000 | 0,327563 | 0,259332 | 0,118106 |
| 1 | 1,740668 | -3,118106 | 0,236912 | 0,311797 | 0,138989 |
| 2 | 1,428871 | -3,257095 | 0,123857 | 0,272651 | 0,118724 |
| 3 | 1,156220 | -3,375819 | 0,052954 | 0,159035 | 0,069038 |
| 4 | 0,997185 | -3,444857 | 0,029778 | ? | ? |
| 5 | ? | ? | — | — | — |

![Ontwikkeling van de sigmoïdefunctie tijdens de iteraties.](evolution_sigmoid.png)

*Afbeelding 5. Ontwikkeling van de sigmoïdefunctie tijdens de iteraties.*

![Testdata voor het classificatiemodel.](testdata.png)

*Afbeelding 6. Testdata voor het classificatiemodel.*

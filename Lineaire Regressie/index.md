# Uitleg van Lineaire Regressie

## 1. Wat gaan we doen?

Stel dat we een dataset hebben, zoals de fictieve gegevens over de oppervlakte en de prijzen van huizen die we in het hoorcollege hebben gebruikt. We willen de prijs van een nieuw huis voorspellen. Nadat we de gegevens hebben bekeken, besluiten we dat we dit probleem met een lineaire functie als model kunnen beschrijven:

$$
\hat{y} = wx + b
$$

We zoeken een lineaire functie die onze gegevens zo goed mogelijk beschrijft. Daarvoor moeten we $$w$$ en $$b$$ bepalen: de parameters van de lineaire functie. Zodra we deze hebben bepaald, kunnen we de functie — het model — gebruiken om voorspellingen te maken voor nieuwe datapunten.

Maar welke lineaire functie, en dus welke keuze van $$w$$ en $$b$$, is het beste? We beoordelen hoe goed een functie de gegevens beschrijft door de *loss* te berekenen. Deze vat de fouten over de datapunten samen. De fout voor een datapunt is het verschil tussen de voorspelde waarde en de werkelijke outputwaarde. Een manier om deze fouten samen te vatten is de *squared error loss function*:

$$
\begin{aligned}
\mathrm{Loss}(\hat{y})
&= \frac{1}{2}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2 \\
&= \frac{1}{2}\sum_{i=1}^{N}\bigl(y_i-(wx_i+b)\bigr)^2.
\end{aligned}
$$

De waarden $$x_i$$ en $$y_i$$ zijn gegeven door onze dataset. De waarden $$x_i$$ zijn de inputwaarden en de waarden $$y_i$$ zijn de outputwaarden voor elk datapunt. Het aantal datapunten is $$N$$.

De parameters $$w$$ en $$b$$ willen we zo kiezen dat de loss over de dataset zo klein mogelijk is. Dit is een **optimaliseringsprobleem**.

**Let op:** we onderzoeken hoe de loss afhangt van de parameters $$w$$ en $$b$$, niet van de waarden $$x_i$$ en $$y_i$$. In deze context zijn $$w$$ en $$b$$ de variabelen die we willen bepalen. De waarden $$x_i$$ en $$y_i$$ zijn constanten: de gegeven getallen in de dataset.

Om de optimale parameters te bepalen, doorlopen we de volgende stappen:

1. We bepalen de eerste partiële afgeleide van de lossfunctie naar elk van de parameters $$w$$ en $$b$$.
2. We stellen beide afgeleiden gelijk aan nul en lossen het resulterende stelsel van vergelijkingen op naar $$w$$ en $$b$$.
3. We vullen de gegevens uit de dataset in de gevonden formules in om de waarden van $$w$$ en $$b$$ te berekenen.
4. We vullen de parameters in de lineaire functie $$\hat{y}=wx+b$$ in. Met dit model kunnen we vervolgens de prijs van nieuwe huizen voorspellen op basis van hun oppervlakte.

In dit eenvoudige voorbeeld van *lineaire regressie* kunnen we deze stappen rechtstreeks doorlopen en zo het model bepalen. Dit is echter niet voor alle modellen mogelijk. Bij sommige modellen of lossfuncties, bijvoorbeeld bij bepaalde vormen van classificatie, is de afgeleide niet overal gedefinieerd. Dan kunnen we stap 1 niet zonder meer uitvoeren en hebben we een andere aanpak nodig.

In machine learning werken we vaak met modellen en lossfuncties waarvoor de benodigde afgeleiden wel bestaan. Toch is het niet altijd mogelijk om het resulterende stelsel van vergelijkingen algebraïsch op te lossen naar de parameters. Dan loopt deze aanpak vast bij stap 2. In zulke gevallen kunnen we bijvoorbeeld *gradient descent* toepassen. Hoe deze methode werkt, bespreken we volgende week. In het programmeergedeelte van het vak werken jullie toe naar het zelf implementeren van deze methode.

## 2. De wiskunde

Voor deze vorm van lineaire regressie kunnen we de parameters wel algebraïsch bepalen. Hieronder werken we dit uit.

### Stap 1: de afgeleiden bepalen

Voor beide parameters, $$w$$ en $$b$$, bepalen we de *partiële afgeleide* van de lossfunctie.

#### De partiële afgeleide naar de parameter b

We beginnen met de afgeleide naar $$b$$. Bij het toepassen van de kettingregel is de afgeleide van $$y_i-(wx_i+b)$$ naar $$b$$ gelijk aan $$-1$$:

$$
\begin{aligned}
\frac{\partial}{\partial b}\mathrm{Loss}(\hat{y})
&= \frac{\partial}{\partial b}\left[\frac{1}{2}\sum_{i=1}^{N}\bigl(y_i-(wx_i+b)\bigr)^2\right] \\
&= \frac{1}{2}\sum_{i=1}^{N}2\bigl(y_i-(wx_i+b)\bigr)(-1) \\
&= -\sum_{i=1}^{N}(y_i-wx_i-b) \\
&= -\sum_{i=1}^{N}y_i + \sum_{i=1}^{N}wx_i + \sum_{i=1}^{N}b \\
&= -\sum_{i=1}^{N}y_i + w\sum_{i=1}^{N}x_i + Nb.
\end{aligned}
$$

#### De partiële afgeleide naar de parameter w

Vervolgens bepalen we de afgeleide naar $$w$$. De afgeleide van $$y_i-(wx_i+b)$$ naar $$w$$ is $$-x_i$$:

$$
\begin{aligned}
\frac{\partial}{\partial w}\mathrm{Loss}(\hat{y})
&= \frac{\partial}{\partial w}\left[\frac{1}{2}\sum_{i=1}^{N}\bigl(y_i-(wx_i+b)\bigr)^2\right] \\
&= \frac{1}{2}\sum_{i=1}^{N}2\bigl(y_i-(wx_i+b)\bigr)(-x_i) \\
&= -\sum_{i=1}^{N}(x_i y_i-wx_i^2-bx_i) \\
&= -\sum_{i=1}^{N}x_i y_i + \sum_{i=1}^{N}wx_i^2 + \sum_{i=1}^{N}bx_i \\
&= -\sum_{i=1}^{N}x_i y_i + w\sum_{i=1}^{N}x_i^2 + b\sum_{i=1}^{N}x_i.
\end{aligned}
$$

### Stap 2: het stelsel van vergelijkingen oplossen

We stellen beide afgeleiden gelijk aan nul. Door beide vergelijkingen met $$-1$$ te vermenigvuldigen, krijgen we het volgende equivalente stelsel:

**Vergelijking 1**

$$
0 = \sum_{i=1}^{N}y_i - w\sum_{i=1}^{N}x_i - Nb.
$$

**Vergelijking 2**

$$
0 = \sum_{i=1}^{N}x_i y_i - w\sum_{i=1}^{N}x_i^2 - b\sum_{i=1}^{N}x_i.
$$

We willen dit stelsel oplossen naar $$w$$ en $$b$$. Daarvoor maken we eerst $$b$$ vrij in vergelijking 1:

$$
\begin{aligned}
0 &= \sum_{i=1}^{N}y_i - w\sum_{i=1}^{N}x_i - Nb, \\
Nb &= \sum_{i=1}^{N}y_i - w\sum_{i=1}^{N}x_i, \\
b &= \frac{1}{N}\left(\sum_{i=1}^{N}y_i - w\sum_{i=1}^{N}x_i\right).
\end{aligned}
$$

Deze uitdrukking voor $$b$$ vullen we in vergelijking 2 in:

$$
\begin{aligned}
0
&= \sum_{i=1}^{N}x_i y_i - w\sum_{i=1}^{N}x_i^2
- \frac{1}{N}\left(\sum_{i=1}^{N}y_i-w\sum_{i=1}^{N}x_i\right)\left(\sum_{i=1}^{N}x_i\right) \\
&= \sum_{i=1}^{N}x_i y_i - w\sum_{i=1}^{N}x_i^2
- \frac{1}{N}\left(\sum_{i=1}^{N}y_i\right)\left(\sum_{i=1}^{N}x_i\right)
+ \frac{w}{N}\left(\sum_{i=1}^{N}x_i\right)\left(\sum_{i=1}^{N}x_i\right) \\
&= \sum_{i=1}^{N}x_i y_i - w\sum_{i=1}^{N}x_i^2
- \frac{1}{N}\left(\sum_{i=1}^{N}y_i\right)\left(\sum_{i=1}^{N}x_i\right)
+ \frac{w}{N}\left(\sum_{i=1}^{N}x_i\right)^2 \\
&= \sum_{i=1}^{N}x_i y_i
- \frac{1}{N}\left(\sum_{i=1}^{N}y_i\right)\left(\sum_{i=1}^{N}x_i\right)
- w\left[\sum_{i=1}^{N}x_i^2 - \frac{1}{N}\left(\sum_{i=1}^{N}x_i\right)^2\right].
\end{aligned}
$$

Nu kunnen we $$w$$ vrijmaken:

$$
w = \frac{\displaystyle\sum_{i=1}^{N}x_i y_i - \frac{1}{N}\left(\sum_{i=1}^{N}y_i\right)\left(\sum_{i=1}^{N}x_i\right)}{\displaystyle\sum_{i=1}^{N}x_i^2 - \frac{1}{N}\left(\sum_{i=1}^{N}x_i\right)^2}.
$$

Door de teller en de noemer met $$N$$ te vermenigvuldigen, krijgen we:

$$
w = \frac{\displaystyle N\sum_{i=1}^{N}x_i y_i - \left(\sum_{i=1}^{N}y_i\right)\left(\sum_{i=1}^{N}x_i\right)}{\displaystyle N\sum_{i=1}^{N}x_i^2 - \left(\sum_{i=1}^{N}x_i\right)^2}.
$$

**Voorwaarde:** deze formule voor $$w$$ geldt wanneer de noemer niet nul is, dus wanneer de inputwaarden niet allemaal gelijk zijn. Dan leveren de gevonden parameters het unieke minimum van deze kwadratische lossfunctie.

### Stappen 3 en 4: de optimale parameters en het model bepalen

We hebben nu formules voor $$w$$ en $$b$$. Gegeven een dataset met waarden voor $$x_i$$ en $$y_i$$ kunnen we de optimale parameterwaarden berekenen.

Eerst berekenen we $$w$$ met de bovenstaande formule. Daarna berekenen we $$b$$:

$$
b = \frac{1}{N}\left(\sum_{i=1}^{N}y_i-w\sum_{i=1}^{N}x_i\right).
$$

Door beide parameters in de volgende functie in te vullen, leggen we ons lineaire model volledig vast:

$$
\hat{y}=wx+b.
$$

We kunnen het model nu gebruiken om de prijs van nieuwe huizen te voorspellen op basis van hun oppervlakte.

# Voorbereidende opdracht week 2 deel II: Partiële Afgeleide (~30min)

## Ter herinnering

Voor een functie van twee variabelen is de **gradiënt** als volgt gedefinieerd:

$$
\nabla f(x,y)=
\begin{pmatrix}
\dfrac{\partial f}{\partial x} \\
\dfrac{\partial f}{\partial y}
\end{pmatrix}.
$$

Bij het berekenen van een partiële afgeleide naar één variabele houd je de andere onafhankelijke variabelen constant.

Je kunt de volgende afgeleiden gebruiken, samen met de productregel en de kettingregel:

$$
\begin{aligned}
\frac{d}{ds}s^n &= ns^{n-1}, \\
\frac{d}{ds}e^s &= e^s, \\
\frac{d}{ds}\log s &= \frac{1}{s}, \qquad s>0.
\end{aligned}
$$

Hier staat $$\log$$ voor de natuurlijke logaritme.

## A. Opgaven

### Opgave 1

Bepaal de partiële afgeleiden $$\frac{\partial f}{\partial x}$$ en $$\frac{\partial f}{\partial y}$$ van de volgende functie:

$$
f(x,y)=x^2+3y^2-4x+5.
$$

### Opgave 2

Bepaal de partiële afgeleiden $$\frac{\partial f}{\partial x}$$ en $$\frac{\partial f}{\partial y}$$ van de volgende functie:

$$
f(x,y)=x^3y+2xy^2-4y.
$$

### Opgave 3

Bepaal de partiële afgeleiden $$\frac{\partial f}{\partial x}$$, $$\frac{\partial f}{\partial y}$$ en $$\frac{\partial f}{\partial z}$$ van de volgende functie:

$$
f(x,y,z)=2x^2y+3y^2z-z^3.
$$

### Opgave 4

Beantwoord de onderstaande vragen voor de functie:

$$
f(x,y)=(x-y)^2+x^2.
$$

**a.** Bepaal de gradiënt $$\nabla f(x,y)$$.

**b.** Bereken de gradiënt $$\nabla f(1,2)$$.

**c.** Stel dat we vanuit het punt $$(1,2)$$ de waarde van $$y$$ een klein beetje verhogen, terwijl we $$x$$ constant houden. Neemt de functiewaarde $$f$$ dan toe of af?

<!--
## B. Eenvoudige functies voor machine learning

In opgaven 6–8 is $$u$$ een vaste invoerwaarde. Waar $$t$$ voorkomt, is dit een vaste doelwaarde. We differentiëren naar de parameters $$w$$ en $$b$$ en houden de invoer- en doelwaarden daarbij constant.

### Opgave 6: lineaire voorspelling

Bepaal de partiële afgeleiden $$\frac{\partial\widehat{f}}{\partial w}$$ en $$\frac{\partial\widehat{f}}{\partial b}$$ van de volgende functie:

$$
\widehat{f}(w,b)=wx+b.
$$

### Opgave 7: kwadratische lossfunctie

Voor één datapunt definiëren we de volgende lossfunctie:

$$
L(w,b)=\frac{1}{2}(wu+b-t)^2.
$$

**a.** Gebruik de kettingregel om de partiële afgeleiden $$\frac{\partial L}{\partial w}$$ en $$\frac{\partial L}{\partial b}$$ te bepalen.

**b.** Bereken de waarden van beide afgeleiden voor:

$$
u=2,\qquad t=3,\qquad w=1,\qquad b=0.
$$

**c.** Voer vanuit deze parameterwaarden één stap van *gradient descent* uit, waarbij je beide parameters gelijktijdig bijwerkt. Gebruik een leersnelheid (*learning rate*) van $$\eta=0{,}1$$:

$$
\begin{aligned}
w_{\mathrm{nieuw}} &= w-\eta\frac{\partial L}{\partial w}, \\
b_{\mathrm{nieuw}} &= b-\eta\frac{\partial L}{\partial b}.
\end{aligned}
$$

Gebruik voor beide updates de afgeleiden die je bij de oorspronkelijke parameterwaarden hebt berekend.

### Opgave 8: voorspelling met de sigmoidfunctie

We definiëren de sigmoidfunctie en de voorspelfunctie als volgt:

$$
\sigma(s)=\frac{1}{1+e^{-s}},
\qquad
p(w,b)=\sigma(wu+b).
$$

**a.** Laat zien dat:

$$
\sigma'(s)=\sigma(s)\bigl(1-\sigma(s)\bigr).
$$

**b.** Bepaal de partiële afgeleiden $$\frac{\partial p}{\partial w}$$ en $$\frac{\partial p}{\partial b}$$. Je mag je antwoorden uitdrukken in termen van $$p$$.

**c.** Bereken de waarden van beide afgeleiden voor:

$$
u=2,\qquad w=0,\qquad b=0.
$$

### Opgave 9: voorspelling met de softplus-functie

De glooiende activatiefunctie *softplus* levert de volgende functie van twee variabelen op:

$$
f(x,y)=\log\bigl(1+e^{x+y}\bigr).
$$

Bepaal de partiële afgeleiden $$\frac{\partial f}{\partial x}$$ en $$\frac{\partial f}{\partial y}$$. Schrijf elk antwoord op twee manieren: als een breuk en met behulp van de sigmoidfunctie uit opgave 8.

-->

### Einde

Heb je de partiële afgeleide opdrachten gemaakt? Ga dan door met de opgaven van lineaire regressie.
Lever deze daarna gezamenlijk in bij de submit-pagina.

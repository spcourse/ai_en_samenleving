# Partiële afgeleide


## A. Oplossingen

### Opgave 1

$$
\frac{\partial f}{\partial x}=2x-4,
\qquad
\frac{\partial f}{\partial y}=6y.
$$

### Opgave 2

$$
\frac{\partial f}{\partial x}=3x^2y+2y^2,
\qquad
\frac{\partial f}{\partial y}=x^3+4xy-4.
$$

### Opgave 3

$$
\begin{aligned}
\frac{\partial f}{\partial x}&=4xy, \\
\frac{\partial f}{\partial y}&=2x^2+6yz, \\
\frac{\partial f}{\partial z}&=3y^2-3z^2.
\end{aligned}
$$

### Opgave 4

**a.** De gradiënt is:

$$
\nabla f(x,y)=\bigl(2(x-y)+2x,\,-2(x-y)\bigr).
$$

**b.** In het punt $$(1,2)$$ is de gradiënt:

$$
\nabla f(1,2)=(0,2).
$$

**c.** De functie groeit.

## B. Eenvoudige functies voor machine learning

In opgaven 6–8 is $$u$$ een vaste inputwaarde en is $$t$$, waar aanwezig, een vaste doelwaarde. De variabelen waarnaar we differentiëren zijn de parameters $$w$$ en $$b$$.

### Opgave 6: lineaire voorspelling

Bepaal de partiële afgeleiden $$\frac{\partial\widehat{f}}{\partial w}$$ en $$\frac{\partial\widehat{f}}{\partial b}$$ voor de volgende functie:

$$
\widehat{f}(w,b)=wu+b.
$$

### Opgave 7: kwadratische verliesfunctie

Voor één datapunt definiëren we:

$$
L(w,b)=\frac{1}{2}(wu+b-t)^2.
$$

**a.** Bepaal de volgende partiële afgeleiden met behulp van de kettingregel:

$$
\frac{\partial L}{\partial w}
\qquad\text{en}\qquad
\frac{\partial L}{\partial b}.
$$

**b.** Bereken beide afgeleiden voor de volgende waarden:

$$
u=2,\qquad t=3,\qquad w=1,\qquad b=0.
$$

**c.** Voer vanuit deze parameterwaarden één gelijktijdige *gradient-descent*-stap uit met leerstapgrootte $$\eta=0{,}1$$:

$$
\begin{aligned}
w_{\mathrm{nieuw}}&=w-\eta\frac{\partial L}{\partial w}, \\
b_{\mathrm{nieuw}}&=b-\eta\frac{\partial L}{\partial b}.
\end{aligned}
$$

Gebruik voor beide updates de afgeleiden die zijn berekend bij de oorspronkelijke parameterwaarden.

### Opgave 8: voorspelling met de sigmoidfunctie

We definiëren:

$$
\sigma(s)=\frac{1}{1+e^{-s}},
\qquad
p(w,b)=\sigma(wu+b).
$$

**a.** Laat zien dat:

$$
\sigma'(s)=\sigma(s)\bigl(1-\sigma(s)\bigr).
$$

**b.** Bepaal de volgende partiële afgeleiden. Je mag je antwoorden uitdrukken in termen van $$p$$:

$$
\frac{\partial p}{\partial w}
\qquad\text{en}\qquad
\frac{\partial p}{\partial b}.
$$

**c.** Bereken beide afgeleiden voor:

$$
u=2,\qquad w=0,\qquad b=0.
$$

### Opgave 9: softplus

Een gladde activatiefunctie geeft de volgende functie van twee variabelen:

$$
f(x,y)=\log\bigl(1+e^{x+y}\bigr).
$$

Bepaal de partiële afgeleiden:

$$
\frac{\partial f}{\partial x}
\qquad\text{en}\qquad
\frac{\partial f}{\partial y}.
$$

Druk elk antwoord zowel uit als een breuk als met behulp van de sigmoidfunctie uit opgave 8.

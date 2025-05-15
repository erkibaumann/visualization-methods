# Elektritarbimise visualiseerimine

See projekt visualiseerib elektritarbimise andmeid kahel viisil, kasutades Pythonit (Pandas, Plotly) andmete töötlemiseks ja visualiseerimiseks ning HTML/CSS-i tulemuste veebis esitamiseks.

## Ülesanded

1.  **Ühe andmestiku 100 päeva mustrid**: Visualiseerib ühe valitud elektritarbimise andmestiku (tavaliselt esimese 100 päeva) andmed, et esile tuua korduvaid mustreid.
2.  **Mitme andmestiku sama päeva võrdlus**: Visualiseerib paljude erinevate andmestike tarbimise ühel konkreetsel päeval, et näidata sarnasusi ja erinevusi mustrites.
3.  **Veebis publitseerimine**: Genereeritud visualiseeringud on kättesaadavad GitHub Pages kaudu.

## Failid projektis

*   `visualize_electricity.py`: Pythoni skript andmete allalaadimiseks, töötlemiseks ja visualiseeringute genereerimiseks. Genereerib `task1_visualization.html` ja `task2_visualization.html`.
*   `index.html`: Peamine HTML fail, mis kuvab teavet projekti kohta ja lingib/embeddib visualiseeringud.
*   `style.css`: CSS fail `index.html`-i stiliseerimiseks.
*   `task1_visualization.html`: Genereeritud HTML fail Ülesanne 1 visualiseeringuga (tekib Pythoni skripti käivitamisel).
*   `task2_visualization.html`: Genereeritud HTML fail Ülesanne 2 visualiseeringuga (tekib Pythoni skripti käivitamisel).
*   `README.md`: See fail.

## Märkused visualiseeringute kohta

*   **Interaktiivsus**: Mõlemad Plotly graafikud on interaktiivsed. Saad hiirega sisse suumida, panoraamida ja üksikute andmepunktide kohta infot vaadata (hover).
*   **Mustrid**:
    *   **Ülesanne 1**: Otsi päevaseid kordumisi (nt kõrge tarbimine hommikul ja õhtul, madal öösel) ja nädalaseid variatsioone (nt erinev muster tööpäevadel vs. nädalavahetustel). Värviskaala on piiratud 1. ja 99. pertentiiliga, et vähendada äärmuslike väärtuste mõju.
    *   **Ülesanne 2**: Võrdle, kuidas erinevad tarbijad (dataset hash) samal päeval käituvad. Mõnel võib olla sarnane profiil, teistel aga täiesti erinev. Graafik on sorteeritud päevase kogutarbimise järgi (suurimad tarbijad üleval). Värviskaala on piiratud 1. ja 95. pertentiiliga.
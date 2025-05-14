# Elektritarbimise Visualiseerimise Projekt

See projekt visualiseerib elektritarbimise andmeid kahel viisil, kasutades Pythonit (Pandas, Plotly) andmete töötlemiseks ja visualiseerimiseks ning HTML/CSS-i tulemuste veebis esitamiseks.

## Ülesanded

1.  **Ühe andmestiku 100 päeva mustrid**: Visualiseerib ühe valitud elektritarbimise andmestiku (tavaliselt esimese 100 päeva) andmed, et esile tuua korduvaid mustreid.
2.  **Mitme andmestiku sama päeva võrdlus**: Visualiseerib paljude erinevate andmestike tarbimise ühel konkreetsel päeval, et näidata sarnasusi ja erinevusi mustrites.
3.  **Veebis publitseerimine**: Genereeritud visualiseeringud on kättesaadavad GitHub Pages kaudu.

## Failid Projektis

*   `visualize_electricity.py`: Pythoni skript andmete allalaadimiseks, töötlemiseks ja visualiseeringute genereerimiseks. Genereerib `task1_visualization.html` ja `task2_visualization.html`.
*   `index.html`: Peamine HTML fail, mis kuvab teavet projekti kohta ja lingib/embeddib visualiseeringud.
*   `style.css`: CSS fail `index.html`-i stiliseerimiseks.
*   `task1_visualization.html`: Genereeritud HTML fail Ülesanne 1 visualiseeringuga (tekib Pythoni skripti käivitamisel).
*   `task2_visualization.html`: Genereeritud HTML fail Ülesanne 2 visualiseeringuga (tekib Pythoni skripti käivitamisel).
*   `README.md`: See fail.

## Seadistamine ja Käivitamine

### Eeldused
*   Python 3.7+
*   Vajalikud Pythoni teegid: `requests`, `pandas`, `numpy`, `plotly`.
    Paigalda need käsuga (soovitatavalt virtuaalses keskkonnas):
    ```bash
    pip install requests pandas numpy plotly
    ```

### Skripti Käivitamine
1.  Veendu, et oled samas kaustas, kus on `visualize_electricity.py`.
2.  Käivita Pythoni skript terminalist või käsurealt:
    ```bash
    python visualize_electricity.py
    ```
3.  Skript laeb alla vajalikud andmed ja genereerib kaks HTML faili: `task1_visualization.html` ja `task2_visualization.html` samasse kausta.
    *   **Märkus**: Andmete allalaadimine ja töötlemine võib võtta aega, eriti esimesel käivitamisel või kui töödeldakse paljusid andmestikke.
    *   **Märkus**: Skriptis on `target_date_for_task2` (Ülesanne 2 jaoks) seatud kindlale kuupäevale (`2023-03-15`). Vajadusel muuda seda kuupäeva `visualize_electricity.py` failis, et analüüsida teist päeva. Veendu, et valitud kuupäeval oleks andmeid. Samuti on `dataset_index` Ülesanne 1 jaoks fikseeritud (vaikimisi `13`).

## Veebis Publitseerimine GitHub Pages kaudu

1.  **Loo GitHub'i repositoorium**: Kui sul seda veel pole, loo uus repositoorium GitHubis (nt `electricity-visualization`).
2.  **Lisa failid repositooriumisse**:
    *   Lisa oma projekti failid (`visualize_electricity.py`, `index.html`, `style.css`, `README.md`) oma lokaalsesse Git repositooriumisse.
    *   Pärast Pythoni skripti (`visualize_electricity.py`) edukat käivitamist lisa ka genereeritud `task1_visualization.html` ja `task2_visualization.html` failid.
    *   Tee `commit` ja `push` muudatustega GitHubi peaharru (tavaliselt `main` või `master`).
    ```bash
    git init # Kui see on uus projektikaust
    git add .
    git commit -m "Add project files and initial visualizations"
    # Lingi lokaalne repo GitHubi omaga (juhised GitHubist peale repo loomist)
    # git remote add origin https://github.com/SINU_KASUTAJANIMI/REPO_NIMI.git 
    git push -u origin main # Või sinu peaharu nimi
    ```
3.  **Aktiveeri GitHub Pages**:
    *   Mine oma GitHubi repositooriumis "Settings" (Seaded) vahekaardile.
    *   Vasakust menüüst vali "Pages" (jaotises "Code and automation").
    *   "Build and deployment" all, vali "Source" jaoks "Deploy from a branch".
    *   Vali haru (Branch), kust sisu serveerida (tavaliselt `main` või `master`).
    *   Kaustaks vali `/ (root)`.
    *   Vajuta "Save".
4.  **Vaata lehte**: Mõne minuti pärast peaks sinu leht olema kättesaadav aadressil kujul `https://<SINU_KASUTAJANIMI>.github.io/<REPO_NIMI>/`. Täpne link kuvatakse GitHub Pages seadete lehel, kui leht on avaldatud. `index.html` on sellel aadressil vaikimisi kuvatav leht.

    Näiteks, kui sinu kasutajanimi on `testuser` ja repo nimi `elekter-visual`, siis aadress võiks olla `https://testuser.github.io/elekter-visual/`.

## Märkused visualiseeringute kohta

*   **Interaktiivsus**: Mõlemad Plotly graafikud on interaktiivsed. Saad hiirega sisse suumida, panoraamida ja üksikute andmepunktide kohta infot vaadata (hover).
*   **Mustrid**:
    *   **Ülesanne 1**: Otsi päevaseid kordumisi (nt kõrge tarbimine hommikul ja õhtul, madal öösel) ja nädalaseid variatsioone (nt erinev muster tööpäevadel vs. nädalavahetustel). Värviskaala on piiratud 1. ja 99. pertentiiliga, et vähendada äärmuslike väärtuste mõju.
    *   **Ülesanne 2**: Võrdle, kuidas erinevad tarbijad (dataset hash) samal päeval käituvad. Mõnel võib olla sarnane profiil, teistel aga täiesti erinev. Graafik on sorteeritud päevase kogutarbimise järgi (suurimad tarbijad üleval). Värviskaala on piiratud 1. ja 95. pertentiiliga.
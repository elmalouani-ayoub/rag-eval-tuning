# Projet 1 — Système RAG avec évaluation et tuning : le guide complet

> **À qui s'adresse ce document ?** À tout le monde. Un lecteur non technique doit
> pouvoir comprendre *l'intuition* et *le pourquoi* ; un lecteur technique doit y
> trouver la *rigueur* (formules, variables, limites). On avance du simple au précis.

---

## Comment lire ce document

Chaque concept important est expliqué en **trois temps**, toujours dans le même ordre :

1. **L'intuition** — une image mentale, sans jargon. « De quoi on parle, au fond ? »
2. **L'explication (périmètre & variables)** — la définition rigoureuse, les paramètres
   qui la font varier, la formule quand il y en a une.
3. **Les limites** — ce que le concept ne fait *pas*, ses angles morts, ses pièges.

Vous pouvez lire en diagonale (les intuitions) ou en profondeur (les trois temps).

**Table des matières**

1. [L'intuition du projet](#1-lintuition-du-projet)
2. [Ce que fait le projet, concrètement](#2-ce-que-fait-le-projet-concrètement)
3. [Les prérequis pratiques](#3-les-prérequis-pratiques)
4. [Tous les concepts, expliqués](#4-tous-les-concepts-expliqués)
5. [Pourquoi ces choix techniques](#5-pourquoi-ces-choix-techniques)
6. [Le déroulé du code, étape par étape](#6-le-déroulé-du-code-étape-par-étape)
7. [Les ressources externes à rassembler](#7-les-ressources-externes-à-rassembler)
8. [Déposer le projet sur GitHub, pas à pas](#8-déposer-le-projet-sur-github-pas-à-pas)
9. [Comment le valoriser (CV / entretien)](#9-comment-le-valoriser-cv--entretien)
10. [Glossaire express](#10-glossaire-express)

---

## 1. L'intuition du projet

### Le problème de départ

Imaginez une entreprise avec des centaines de pages de documents internes (règles RH,
procédures IT, notes de frais…). Un salarié pose une question simple : *« Combien de
jours de télétravail par semaine ? »*. La réponse existe, mais elle est noyée quelque
part dans un PDF.

On aimerait un assistant à qui l'on pose la question en français, et qui répond en
citant le bon document. C'est exactement ce que fait un système **RAG**.

Mais il y a un piège. Un RAG a beaucoup de **réglages** (quelle taille de morceaux de
texte ? combien de documents aller chercher ? quelle méthode pour « comprendre » le
sens ?). Selon les réglages, la qualité des réponses change du tout au tout. La plupart
des gens branchent un RAG « au hasard » et espèrent que ça marche.

### L'idée du projet

Ce projet ne se contente pas de *construire* un RAG. Il **mesure** quelle configuration
marche le mieux, et **le prouve avec des chiffres et des graphiques**.

C'est la différence entre dire *« j'ai branché un RAG »* et dire *« j'ai construit un RAG,
puis testé 88 configurations différentes, et voici celle qui donne les meilleures
réponses, avec les preuves »*. La deuxième phrase est beaucoup plus crédible — c'est
un travail d'ingénieur, pas de bricoleur.

### Pourquoi c'est malin

L'auteur du projet est déjà fort en **optimisation** et en **évaluation de modèles** (son
métier). Le RAG est justement un domaine où l'optimisation et l'évaluation font toute la
différence. Le projet met donc en valeur une compétence *déjà existante*, en la rendant
visible sur un sujet à la mode (l'IA générative).

---

## 2. Ce que fait le projet, concrètement

En une phrase : **le projet prend une base de documents et un ensemble de questions,
puis teste automatiquement des dizaines de réglages du RAG pour trouver le meilleur, avec
des métriques et des graphiques à l'appui.**

Le déroulé :

1. **Une base documentaire** : 10 documents (une base de connaissances métier fictive :
   congés, télétravail, sécurité informatique…).
2. **Un jeu de 20 questions** dont on connaît d'avance la bonne réponse et le document où
   elle se trouve (la « vérité-terrain »).
3. **Un balayage** (*sweep*) de **88 configurations** : on fait varier la taille des
   morceaux de texte, leur chevauchement, le nombre de documents récupérés, et la méthode
   de « compréhension » du texte.
4. **Une mesure** : pour chaque configuration, on calcule 4 métriques de qualité.
5. **Une analyse** : un tableau, des graphiques, et la désignation de la meilleure
   configuration — avec l'explication du *pourquoi*.

Résultat obtenu dans ce projet : la meilleure configuration atteint **100 % de bonnes
réponses trouvées**, et on comprend *pourquoi* (voir §6).

---

## 3. Les prérequis pratiques

Avant les concepts, voici ce qu'il faut **savoir faire** et **avoir installé**.

**Compétences utiles (pas besoin d'être expert) :**

- Notions de **Python** (lire un script, lancer une fonction).
- Savoir ouvrir un **terminal** et taper quelques commandes.
- Notions de **Git** (on les rappelle au §8).

**À installer sur votre machine :**

- **Python 3.10 ou plus** — le langage du projet.
- **pip** — l'outil qui installe les librairies Python (fourni avec Python).
- **Git** — pour versionner et déposer le code.
- *(optionnel)* **Ollama** — pour faire tourner un vrai modèle d'IA en local (voir §7).

**Le cœur du projet tourne sans rien de plus** : les librairies s'installent en une
commande (`pip install -r requirements.txt`), et l'évaluation fonctionne **100 %
hors-ligne**, sans carte graphique ni compte payant. C'est un choix volontaire (voir §5).

---

## 4. Tous les concepts, expliqués

On applique ici la méthode en trois temps (intuition → explication/variables → limites).

### 4.1 Le LLM (grand modèle de langage)

- **Intuition.** Un LLM (comme ChatGPT) est un programme qui a lu énormément de texte et
  qui sait produire du texte plausible. Voyez-le comme un « auto-complète » très
  sophistiqué : on lui donne un début, il devine la suite mot après mot.
- **Explication & variables.** Techniquement, un LLM prédit le prochain « token » (un mot
  ou un morceau de mot) en fonction de tout ce qui précède. Il est réglé par des
  paramètres comme la **température** (0 = réponses stables et factuelles ; élevée =
  réponses créatives et variées). Dans ce projet, on peut brancher un LLM **open-weight**
  (dont les poids sont librement téléchargeables), comme *Mistral 7B* ou *Llama 3 8B*,
  exécuté localement.
- **Limites.** Un LLM seul **invente** parfois (on parle d'**hallucination**) : il produit
  une réponse fluide mais fausse, car il n'a pas de source. Il ne « connaît » pas vos
  documents internes. C'est précisément le problème que le RAG vient corriger.

### 4.2 Le RAG (génération augmentée par la recherche)

- **Intuition.** RAG = *Retrieval-Augmented Generation*. On ajoute au LLM une étape de
  **recherche documentaire** : avant de répondre, le système va d'abord **retrouver** les
  passages pertinents dans vos documents, puis demande au LLM de répondre *en s'appuyant
  sur ces passages*. C'est la différence entre un étudiant qui répond de mémoire et un
  étudiant qui a le droit d'ouvrir le manuel à la bonne page.
- **Explication & variables.** Un RAG enchaîne cinq étapes :
  *découpage (chunking) → transformation en vecteurs (embeddings) → indexation → recherche
  (retrieval) → génération*. Chaque étape a ses réglages (voir plus bas). L'intérêt : les
  réponses sont **ancrées** dans des sources vérifiables.
- **Limites.** Un RAG n'est bon que si la **recherche** est bonne : si l'étape « retrouver
  le bon passage » échoue, le LLM répondra à côté (ou inventera). D'où l'importance
  d'**évaluer et d'optimiser le retrieval** — c'est tout le sujet du projet.

### 4.3 Les embeddings (vecteurs de sens)

- **Intuition.** Un ordinateur ne comprend pas les mots, il comprend les nombres. Un
  **embedding** transforme un texte en une liste de nombres (un « vecteur ») de telle
  sorte que **deux textes de sens proche donnent deux vecteurs proches**. C'est comme
  poser chaque phrase sur une carte : les phrases qui parlent de la même chose se
  retrouvent voisines.
- **Explication & variables.** Un embedding est un vecteur de dimension fixe (par ex.
  quelques centaines de nombres). On compare deux vecteurs par leur **proximité**
  (voir la similarité cosinus, §4.6). Les variables importantes : la **méthode**
  d'embedding (lexicale ou sémantique) et la **dimension** du vecteur.
- **Limites.** Un embedding n'est qu'une *approximation* du sens. Selon la méthode, il peut
  capturer les mots exacts mais rater les reformulations (méthode lexicale), ou l'inverse.
  Le choix de la méthode a un impact énorme sur la qualité — c'est l'un des réglages testés.

### 4.4 Les méthodes d'embedding : lexical vs sémantique

Le projet compare plusieurs façons de fabriquer les vecteurs.

**TF-IDF (méthode lexicale, utilisée par défaut, hors-ligne)**

- **Intuition.** On représente un texte par les **mots qu'il contient**, en donnant plus
  de poids aux mots **rares et discriminants** (« kilométrique ») qu'aux mots courants
  (« le », « de »).
- **Explication & variables.** TF-IDF = *Term Frequency × Inverse Document Frequency*.
  La *TF* mesure combien de fois un mot apparaît dans un texte ; l'*IDF* pénalise les mots
  présents partout. Le score d'un mot monte s'il est fréquent *ici* mais rare *ailleurs*.
  Variables : la taille du vocabulaire, la prise en compte ou non des paires de mots
  (*n-grammes*).
- **Limites.** TF-IDF est **purement lexical** : il compare des mots, pas des idées. Si la
  question dit « congés » et le document dit « vacances », il peut passer à côté. Il ne
  gère pas les synonymes ni les reformulations.

**Hashing (méthode lexicale sans vocabulaire)**

- **Intuition.** Variante de TF-IDF qui n'apprend pas de vocabulaire : elle « range » chaque
  mot dans un casier via une fonction de hachage.
- **Explication & variables.** Rapide et sans état, mais deux mots différents peuvent
  tomber dans le même casier (*collision*), et il n'y a pas de pondération IDF.
- **Limites.** Justement à cause des collisions et de l'absence d'IDF, il est **moins
  précis** que TF-IDF (le projet le démontre : TF-IDF gagne sur toutes les métriques). Il
  sert ici de **point de comparaison**.

**Sémantique (sentence-transformers / Ollama, méthode « intelligente »)**

- **Intuition.** Un petit modèle d'IA lit la phrase et produit un vecteur qui capture le
  **sens**, pas seulement les mots. « Congés » et « vacances » deviennent proches.
- **Explication & variables.** Ces modèles (ex. *sentence-transformers*, ou
  *nomic-embed-text* via Ollama) sont entraînés pour rapprocher les phrases de sens
  similaire. Variables : le modèle choisi, sa langue, sa dimension.
- **Limites.** Nécessite de **télécharger un modèle** (plus lourd, pas hors-ligne pur), et
  reste imparfait sur le jargon très spécifique. C'est le chemin « production » du projet,
  activable en option.

### 4.5 Le chunking (découpage en morceaux)

- **Intuition.** On ne peut pas donner un document entier au moteur de recherche d'un coup :
  on le **découpe en morceaux** (*chunks*). Trop gros, le morceau mélange plusieurs sujets ;
  trop petit, il coupe une réponse en deux. Comme découper un texte en paragraphes de la
  bonne taille pour retrouver la bonne info.
- **Explication & variables.** Deux réglages clés :
  - **`chunk_size`** : la taille d'un morceau (ici, en nombre de mots).
  - **`chunk_overlap`** : le **chevauchement** entre deux morceaux consécutifs. Un
    chevauchement évite de couper une phrase importante juste à la frontière de deux
    morceaux. Le « pas » d'avancement vaut `chunk_size − chunk_overlap`.
- **Limites.** Il n'existe pas de taille universelle : elle dépend des documents. Trop de
  chevauchement gonfle le nombre de morceaux (donc le coût) sans forcément améliorer la
  qualité. C'est pourquoi on **teste** plusieurs valeurs plutôt que d'en supposer une.

### 4.6 La base vectorielle & la similarité cosinus

- **Intuition.** Une fois tous les morceaux transformés en vecteurs, on les range dans une
  **base vectorielle**. Pour répondre à une question, on transforme la question en vecteur
  et on cherche les morceaux **les plus proches**. « Proche » se mesure par l'**angle**
  entre deux vecteurs : plus l'angle est petit, plus les sens sont alignés.
- **Explication & variables.** La mesure utilisée est la **similarité cosinus** : le cosinus
  de l'angle entre deux vecteurs (1 = même direction, 0 = sans rapport). Astuce technique du
  projet : on **normalise** les vecteurs (longueur ramenée à 1, dite « norme L2 »), ce qui
  rend la similarité cosinus égale à un simple **produit scalaire** — plus rapide à calculer.
- **Limites.** Sur un très grand corpus, comparer la question à *tous* les vecteurs devient
  lent. Le projet utilise une recherche exacte (le corpus est petit), et prévoit une
  variante **FAISS** (index optimisé) pour passer à l'échelle.

### 4.7 Le retrieval et le paramètre `top_k`

- **Intuition.** *Retrieval* = l'étape « aller chercher les bons morceaux ». On décide
  **combien** de morceaux on ramène : c'est le **`top_k`** (les *k* plus proches). Ramener
  peu de morceaux = précis mais risqué (on peut rater la réponse) ; en ramener beaucoup =
  on ne rate rien, mais on ramène aussi du bruit.
- **Explication & variables.** `top_k` est un entier (le projet teste 1, 3, 5, 10). Il
  contrôle directement le **compromis précision/rappel** (§4.11).
- **Limites.** Un `top_k` trop grand noie la réponse dans du texte hors-sujet, ce qui
  dégrade la génération du LLM et augmente le coût (prompt plus long). Il faut trouver le
  bon équilibre — encore un réglage qu'on mesure.

### 4.8 Les hyperparamètres et le tuning (recherche en grille)

- **Intuition.** Un **hyperparamètre** est un « bouton de réglage » qu'on fixe *avant* de
  faire tourner le système (taille de chunk, `top_k`, méthode d'embedding…). Le **tuning**,
  c'est tourner ces boutons pour trouver la meilleure combinaison. La **recherche en
  grille** (*grid search*) teste **toutes les combinaisons possibles** de façon systématique.
- **Explication & variables.** Dans ce projet, la grille est :
  `chunk_size ∈ {40, 80, 120, 200}` × `chunk_overlap ∈ {0, 20, 40}` ×
  `top_k ∈ {1, 3, 5, 10}` × `embedding ∈ {tfidf, hashing}`.
  Soit 4 × 3 × 4 × 2 = **96 combinaisons**, dont on retire les combinaisons incohérentes
  (chevauchement ≥ taille), ce qui laisse **88 configurations** réellement évaluées.
- **Limites.** La recherche en grille **explose** avec le nombre de réglages (c'est un
  produit, pas une somme). Sur de grandes grilles, on lui préfère des méthodes plus malines
  (recherche aléatoire, optimisation bayésienne). Ici, la grille est volontairement petite
  et lisible.

### 4.9 La vérité-terrain (ground truth)

- **Intuition.** Pour noter un système, il faut connaître la **bonne réponse à l'avance**,
  comme un corrigé d'examen. La **vérité-terrain**, c'est ce corrigé : pour chaque question,
  on sait quel document contient la réponse et quel extrait exact la donne.
- **Explication & variables.** Chaque question du jeu d'évaluation porte deux informations
  de référence : `source_doc` (le bon document) et `answer_span` (un extrait de la réponse
  attendu, mot pour mot). Ces références permettent de dire objectivement si le système a
  « trouvé juste ».
- **Limites.** Construire une vérité-terrain est **coûteux** et **subjectif** (il faut la
  rédiger à la main). Un jeu de 20 questions est suffisant pour un projet d'apprentissage,
  mais trop petit pour des conclusions statistiques solides.

### 4.10 Les métriques d'évaluation

Le projet mesure la qualité du **retrieval** avec quatre métriques. Point crucial : elles
ne dépendent **pas** du LLM — uniquement des morceaux récupérés — ce qui les rend
**reproductibles** sans carte graphique.

**hit@k (taux de réussite)**

- **Intuition.** « Le bon document est-il dans les *k* morceaux ramenés, oui ou non ? »
- **Explication & variables.** Vaut 1 si au moins un morceau du bon document figure dans le
  `top_k`, sinon 0. On fait la moyenne sur toutes les questions → un taux entre 0 et 1.
- **Limites.** Binaire : ne dit pas *à quelle position* se trouve le bon document (1re ou
  10e). D'où la métrique suivante.

**MRR (rang réciproque moyen)**

- **Intuition.** Récompense le fait de placer le bon document **en haut** de la liste. Être
  1er vaut mieux qu'être 5e.
- **Explication & variables.** Pour une question, on prend `1 / rang` du premier bon
  morceau (1er → 1 ; 2e → 0,5 ; 3e → 0,33…), et 0 s'il est absent. Le **MRR** est la
  moyenne de ces valeurs. Plus il est proche de 1, mieux le bon document est classé.
- **Limites.** Ne regarde que le **premier** bon résultat ; il ignore les éventuels autres
  bons documents plus bas dans la liste.

**span_recall@k (couverture de la réponse)**

- **Intuition.** « L'extrait exact de la réponse est-il réellement présent dans les morceaux
  ramenés ? » On vérifie non seulement le bon document, mais que **le bon passage** y est.
- **Explication & variables.** Vaut 1 si l'`answer_span` attendu apparaît dans l'un des
  morceaux récupérés (après normalisation, §4.12), sinon 0 ; puis moyenne sur les questions.
- **Limites.** Repose sur une correspondance de texte : si la réponse est reformulée
  autrement que dans le document, l'extrait exact peut ne pas correspondre.

**precision@k (pureté des résultats)**

- **Intuition.** « Parmi les *k* morceaux ramenés, quelle proportion vient vraiment du bon
  document ? » Mesure le **bruit**.
- **Explication & variables.** = (nombre de morceaux issus du bon document) / `top_k`.
  Quand `top_k` augmente, la précision tend à baisser (on ramène plus de bruit).
- **Limites.** Une précision basse n'est pas forcément grave si le bon passage est présent ;
  elle compte surtout parce qu'un contexte bruité **dégrade la génération** du LLM.

### 4.11 Le compromis précision / rappel

- **Intuition.** Deux objectifs qui tirent en sens opposés. Le **rappel** = « ne rien
  rater » (ramener le bon document même au prix de bruit). La **précision** = « ne ramener
  que du pertinent » (au risque de rater quelque chose). Filet large vs filet fin.
- **Explication & variables.** Augmenter `top_k` améliore le rappel (`hit@k` monte) mais
  dégrade la précision (`precision@k` chute). Le projet montre ce compromis en graphique :
  au-delà de `top_k = 3`, le rappel plafonne mais la précision s'effondre.
- **Limites.** Il n'y a pas de « bon » choix universel : cela dépend de l'usage. Pour un
  RAG, un peu de rappel supplémentaire ne sert à rien si ça noie le LLM sous du bruit.

### 4.12 La normalisation de texte

- **Intuition.** Pour comparer deux textes de façon fiable, on les met d'abord dans une
  forme « standard » : tout en minuscules, sans accents, sans mise en forme. Ainsi
  « Congés » et « conges » sont considérés identiques.
- **Explication & variables.** La normalisation du projet : minuscules → suppression des
  accents → suppression des caractères de mise en forme Markdown (`#`, `*`) → compactage
  des espaces. Elle rend la comparaison des extraits (span_recall) robuste.
- **Limites.** Une normalisation trop agressive peut effacer des distinctions utiles
  (ponctuation signifiante, casse d'un sigle). Il faut doser.

### 4.13 Reproductibilité et mode hors-ligne

- **Intuition.** Un résultat scientifique doit pouvoir être **rejoué à l'identique** par
  n'importe qui. Si votre évaluation dépend d'un modèle payant ou d'internet, personne ne
  peut la reproduire facilement.
- **Explication & variables.** Le projet rend l'évaluation **déterministe** : mêmes données,
  mêmes réglages → mêmes chiffres. Les métriques de retrieval n'utilisent aucun LLM, et les
  embeddings par défaut (TF-IDF) ne nécessitent aucun téléchargement.
- **Limites.** La reproductibilité parfaite est plus difficile dès qu'on introduit un LLM
  (réponses variables selon la température, versions du modèle). D'où le choix d'évaluer
  surtout le retrieval, qui, lui, est stable.

### 4.14 Ollama et les LLM locaux

- **Intuition.** **Ollama** est un logiciel qui fait tourner des modèles d'IA **sur votre
  propre ordinateur**, gratuitement et sans envoyer vos données à un service tiers.
- **Explication & variables.** On « tire » un modèle (`ollama pull mistral`) puis on
  l'interroge via une petite API locale. Variables : le modèle choisi, sa taille (7B, 8B…),
  la mémoire disponible.
- **Limites.** La qualité dépend de la puissance de la machine ; les gros modèles sont
  lents sur un laptop modeste. C'est pourquoi le projet fonctionne aussi **sans** Ollama
  (mode extractif de repli).

### 4.15 RAGAS (évaluation de la génération)

- **Intuition.** Les métriques de retrieval jugent « a-t-on trouvé le bon passage ? ».
  **RAGAS** juge la **réponse rédigée** : est-elle *fidèle* aux sources (pas d'invention) et
  *pertinente* par rapport à la question ?
- **Explication & variables.** RAGAS utilise un LLM « juge » pour scorer la **fidélité** et
  la **pertinence**. Le projet fournit ce branchement (activable avec Ollama), mais ne
  l'impose pas dans le balayage hors-ligne.
- **Limites.** Nécessite un LLM (donc une machine capable), et un « juge » IA reste
  imparfait et un peu variable. C'est l'extension naturelle du projet, pas son cœur.

### 4.16 Tests, intégration continue et notebook

- **Intuition.** Les **tests** vérifient automatiquement que le code fait ce qu'il doit.
  L'**intégration continue** (CI) rejoue ces tests à chaque modification. Le **notebook**
  est un document interactif qui mélange code, résultats et explications.
- **Explication & variables.** Le projet utilise **pytest** (7 tests), un workflow **GitHub
  Actions** (qui relance tests + évaluation à chaque `push`), et un **notebook Jupyter**
  d'analyse. Ces éléments prouvent le **sérieux méthodologique**.
- **Limites.** Des tests ne prouvent pas l'absence totale de bugs, seulement que les cas
  prévus passent. Un notebook, s'il n'est pas ré-exécuté, peut afficher des résultats périmés.

---

## 5. Pourquoi ces choix techniques

Chaque décision répond à une contrainte concrète.

- **Évaluer le retrieval sans LLM.** C'est le choix central. Il rend le tuning
  **déterministe, gratuit et reproductible** (pas de carte graphique, pas d'aléa). Sans lui,
  chaque mesure dépendrait d'un modèle lourd et variable — impossible de conclure proprement.
- **TF-IDF par défaut (embeddings hors-ligne).** Aucun téléchargement, tourne partout, y
  compris dans la CI. On garde les embeddings sémantiques (plus lourds) en **option**.
- **Comparer TF-IDF *et* hashing.** Avoir au moins deux méthodes permet une **comparaison**
  chiffrée, plutôt qu'une affirmation gratuite. Le projet *montre* que TF-IDF gagne.
- **Découpage en mots (et pas en caractères).** Plus lisible, indépendant de la langue,
  et cohérent avec la façon dont on lit un texte.
- **Base vectorielle en numpy + option FAISS.** Le corpus est petit : une recherche exacte
  en numpy suffit et évite une dépendance. FAISS est prévu pour montrer le passage à
  l'échelle sans réécrire la logique.
- **Architecture « pluggable » (embeddings et LLM interchangeables).** On peut passer du
  mode hors-ligne au mode Ollama **sans changer le code**, juste la configuration. C'est un
  bon réflexe d'ingénierie (et un bon argument d'entretien).
- **Un jeu de 20 questions avec vérité-terrain explicite.** Petit mais suffisant pour un
  projet d'apprentissage, et surtout **honnête** : chaque question sait où est sa réponse.
- **Tests + CI + notebook.** Ils transforment un script en **projet crédible** : quelqu'un
  peut vérifier, rejouer, et voir la démarche.

En résumé : **tout est optimisé pour qu'un tiers puisse reproduire les résultats et croire
les conclusions.** C'est ça, la rigueur.

---

## 6. Le déroulé du code, étape par étape

Le projet est organisé en un petit package (`ragkit`) et des scripts.

1. **Chargement** (`loaders.py`) — lit les 10 documents et les 20 questions.
2. **Découpage** (`chunking.py`) — coupe chaque document en morceaux selon `chunk_size` et
   `chunk_overlap`.
3. **Embeddings** (`embeddings.py`) — transforme chaque morceau en vecteur (TF-IDF par défaut).
4. **Indexation & recherche** (`vectorstore.py`) — range les vecteurs et retrouve les plus
   proches d'une question par similarité cosinus.
5. **Pipeline** (`pipeline.py`) — assemble le tout : question → recherche → (réponse).
6. **Métriques** (`metrics.py`) — le cœur : calcule hit@k, MRR, span_recall@k, precision@k
   par rapport à la vérité-terrain.
7. **Balayage** (`evaluate.py` + `scripts/run_eval.py`) — teste les 88 configurations et
   écrit un tableau de résultats (`results.csv`).
8. **Figures** (`scripts/make_figures.py`) — génère les graphiques d'analyse.
9. **Analyse** (`notebooks/analysis.ipynb`) — désigne la meilleure configuration et l'explique.

**Ce que révèlent les résultats :**

- La **méthode d'embedding** est le levier n°1 : TF-IDF bat le hashing partout.
- **`top_k = 3`** capte tout le rappel utile sans effondrer la précision.
- Une **taille de chunk de 80–120 mots** offre le meilleur compromis.
- Meilleure configuration : `tfidf · chunk_size=80 · chunk_overlap=40 · top_k=1`, qui
  atteint **1,0 sur toutes les métriques** sur ce jeu de test.

---

## 7. Les ressources externes à rassembler

Pour faire tourner le projet de bout en bout, voici ce que **vous** devez fournir.

**Indispensable (mode hors-ligne) :**

- **Python 3.10+** — [python.org](https://www.python.org)
- Les **librairies** listées dans `requirements.txt` (installées via `pip`) : numpy,
  pandas, scikit-learn, matplotlib, PyYAML.
- *(Pour rejouer le notebook)* Jupyter (`pip install jupyter`).

**Votre propre base documentaire (recommandé pour personnaliser).** Le projet fournit un
corpus fictif ; pour le rendre « vôtre », remplacez les 10 fichiers du dossier
`data/corpus/` par **vos** documents (cours, documentation, rapports publics). Pensez alors
à **réécrire le jeu de questions** (`data/eval/questions.json`) avec la vérité-terrain
correspondante.

**Optionnel (mode « production » avec un vrai LLM) :**

- **Ollama** — [ollama.com](https://ollama.com), puis `ollama pull mistral` et
  `ollama pull nomic-embed-text`.
- La librairie **requests** (`pip install requests`) pour dialoguer avec Ollama.
- *(Pour les embeddings sémantiques)* **sentence-transformers** — modèles sur
  [huggingface.co](https://huggingface.co).

**Un compte GitHub** — [github.com](https://github.com) — pour publier le projet (voir §8).

**Pour approfondir les concepts (lecture) :** la documentation d'**Ollama**, de
**scikit-learn** (TF-IDF, similarité cosinus), de **LangChain** et de **RAGAS**. Cherchez
leur documentation officielle ; privilégiez toujours les sources officielles aux forums.

---

## 8. Déposer le projet sur GitHub, pas à pas

**Prérequis :** un compte GitHub, et Git installé (`git --version` doit répondre).
Configurez votre identité une seule fois :

```bash
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

**Étape 1 — Créer le dépôt local.** Placez-vous dans le dossier du projet :

```bash
cd rag-eval-tuning
git init
```

**Étape 2 — Vérifier le `.gitignore`.** Il évite d'envoyer les fichiers inutiles (caches,
environnements). Le projet en fournit déjà un.

**Étape 3 — Des commits progressifs (important).** Ne faites **pas** un seul commit
« final ». Un historique qui montre la construction pas à pas est plus crédible :

```bash
git add data/ src/
git commit -m "Ajout du corpus, du jeu d'évaluation et du package ragkit"

git add scripts/ config.yaml
git commit -m "Ajout du balayage d'hyperparamètres et des scripts"

git add results/ notebooks/
git commit -m "Résultats de l'évaluation, figures et notebook d'analyse"

git add tests/ .github/ README.md
git commit -m "Tests, intégration continue et documentation"
```

**Étape 4 — Créer le dépôt distant.** Sur github.com : bouton *New repository*, nommez-le
`rag-eval-tuning`, laissez-le vide (sans README, vous en avez déjà un), puis *Create*.

**Étape 5 — Relier et pousser.** GitHub vous donne l'URL. Puis :

```bash
git remote add origin https://github.com/VOTRE-UTILISATEUR/rag-eval-tuning.git
git branch -M main
git push -u origin main
```

**Étape 6 — Soigner la vitrine.**

- Remplacez `<ton-user>` dans le badge du `README.md` par votre identifiant GitHub, pour
  que le **badge de build vert** s'affiche.
- Vérifiez que les **graphiques** et le **notebook** s'affichent bien sur la page du dépôt.
- Ajoutez le **lien GitHub** en évidence sur votre CV.

> **Règle d'or.** Un recruteur passe ~30 secondes sur un dépôt : c'est le **README** qui
> fait tout (contexte, architecture, résultats, comment lancer). Le vôtre est déjà soigné —
> gardez-le à jour.

---

## 9. Comment le valoriser (CV / entretien)

**Sur le CV**, la ligne passe d'un déclaratif faible à un fait démontré :

- Avant : *« Forte appétence pour le RAG »*.
- Après : *« Projet personnel : conception et optimisation d'un système RAG (TF-IDF/Ollama),
  évaluation rigoureuse (hit@k, MRR, RAGAS) et tuning d'hyperparamètres sur 88 configurations. »*

**En entretien**, l'argument clé : vous ne dites plus « je m'intéresse au RAG », vous dites
*« j'ai construit un RAG **et mesuré** l'impact des hyperparamètres sur la qualité des
réponses »*. Vous pouvez parler concrètement de compromis précision/rappel, de vérité-terrain,
de reproductibilité — le vocabulaire d'un ingénieur.

**Restez honnête sur l'échelle.** C'est un projet d'apprentissage, pas un système de
production. Présentez-le comme *« j'ai voulu comprendre X en le construisant »*. Cette
honnêteté est un atout, pas une faiblesse.

---

## 10. Glossaire express

| Terme | En une phrase |
|-------|---------------|
| **LLM** | Programme qui génère du texte plausible mot à mot ; peut inventer. |
| **RAG** | LLM + recherche documentaire : répond en s'appuyant sur des sources. |
| **Embedding** | Transformation d'un texte en vecteur de nombres qui capture le sens. |
| **TF-IDF** | Embedding lexical qui valorise les mots rares et discriminants. |
| **Chunk** | Morceau de document ; réglé par sa taille et son chevauchement. |
| **Similarité cosinus** | Mesure de proximité entre deux vecteurs (angle). |
| **top_k** | Nombre de morceaux ramenés à la recherche. |
| **Hyperparamètre** | Réglage fixé avant l'exécution (taille, top_k, méthode…). |
| **Grid search** | Test systématique de toutes les combinaisons de réglages. |
| **Vérité-terrain** | Le « corrigé » : bonne réponse et bon document connus d'avance. |
| **hit@k** | Le bon document est-il dans les k résultats ? (0/1) |
| **MRR** | Récompense le fait de classer le bon document en tête. |
| **span_recall@k** | L'extrait exact de la réponse est-il dans les résultats ? |
| **precision@k** | Proportion des résultats issus du bon document. |
| **Précision/rappel** | Compromis « ne ramener que du pertinent » vs « ne rien rater ». |
| **Reproductibilité** | Capacité à rejouer les mêmes résultats à l'identique. |
| **Ollama** | Logiciel pour faire tourner un LLM localement et gratuitement. |
| **RAGAS** | Évaluation de la réponse rédigée (fidélité, pertinence). |
| **CI** | Intégration continue : rejoue tests et évaluation à chaque modification. |

---

*Ce document accompagne le dépôt `rag-eval-tuning`. Pour les détails d'exécution
(commandes, dépendances), voir le `README.md` du dépôt.*

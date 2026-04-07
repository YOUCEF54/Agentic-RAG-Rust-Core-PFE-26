# Script de Présentation Orale
## RAG Agentique — Rapport d'Avancement
### Brahim Bazi

---

## 🎯 Slide 1 — Page de Garde
*(~30 secondes)*

Bonjour à tous. Je m'appelle Brahim Bazi, et je vais vous présenter l'avancement de notre projet de fin d'études portant sur l'implémentation d'un système RAG Agentique. Ce rapport couvre les résultats de nos benchmarks de performance ainsi que l'architecture multi-agents que nous avons conçue et implémentée.

---

## 📋 Slide 2 — Sommaire
*(~15 secondes)*

Voici le plan de cette présentation. Nous commencerons par les résultats de nos tests de latence, puis nous détaillerons l'architecture multi-agents, les optimisations clés, le stack technique, et nous terminerons par une démonstration et les prochaines étapes.

---

## 🔬 Slide 3 — Contexte des Tests de Performance
*(~45 secondes)*

Pour rappel, la semaine dernière on avait commencé les tests de performance,mais on n'avait pas eu le temps de les finaliser.
Cette semaine, on vous présente les résultats complets

Le protocole est simple : nous avons exécuté le pipeline RAG 100 fois de suite, avec la même requête et le même corpus de documents, en mesurant la latence à chaque étape.

Nous avons comparé trois approches : une implémentation 100% Python, une 100% Rust, et une approche hybride qui combine l'orchestration Python avec des modules critiques en Rust.

---

## 📊 Slide 4 — Résultat 1 : Embeddings
*(~60 secondes)*

Le premier résultat concerne le temps moyen de génération des embeddings — c'est l'étape la plus coûteuse du pipeline.

Les chiffres parlent d'eux-mêmes : l'approche **Hybride atteint 33 956ms**, **Rust 62 842ms**, et **Python 92 411ms**. Concrètement, l'Hybride est **2,7 fois plus rapide que Python** et presque **2 fois plus rapide que Rust**.

Pourquoi ? Parce que l'approche hybride utilise le runtime ONNX directement via Rust pour les embeddings, alors que Python passe par des couches d'abstraction supplémentaires. C'est le gain principal de notre architecture hybride.

---

## 📊 Slide 5 — Résultat 2 : Pipeline hors embeddings
*(~60 secondes)*

Pour aller plus loin, nous avons isolé le coût pur du pipeline en excluant le temps d'embedding. Ce graphique empilé décompose chaque étape : lecture PDF en violet, chunking en bleu-vert, insertion DB en vert, et recherche vectorielle en jaune.

Deux observations importantes. Premièrement, l'**Hybride reste le plus rapide à environ 300ms**, contre 450ms pour Rust et 870ms pour Python. Deuxièmement — et c'est ce que le graphique empilé révèle clairement — pour Python, la **barre jaune de recherche vectorielle est disproportionnellement grande**. C'est parce que la recherche vectorielle en Python pur est moins optimisée que les bindings Rust de LanceDB que nous utilisons.

Ce résultat confirme que l'avantage de l'approche hybride ne vient pas uniquement des embeddings — tout le pipeline bénéficie des modules Rust.

---

## 📈 Slide 6 — Résultat 3 : Tendances sur 100 runs
*(~45 secondes)*

Ce graphique montre cinq métriques mesurées sur les 100 exécutions : lecture PDF, chunking, insertion DB, recherche, et enfin embedding.

Regardons le dernier graphique — model_embedding — qui est le plus révélateur. La courbe bleue de Python **monte progressivement** au fil des runs, signe de dégradation mémoire. La courbe orange de Rust est plus stable mais présente des pics occasionnels. La courbe verte de l'Hybride, elle, reste **pratiquement plate du début à la fin**.

Sur le graphique de recherche vectorielle, on voit que Python a un pic important autour du run 80 — probablement un passage du Garbage Collector. Rust et Hybride n'ont aucun pic comparable.

---

## 🔍 Slide 7 — Analyse des tendances
*(~45 secondes)*

Ce qui est intéressant ici, c'est la **stabilité**. Python présente des pics de latence occasionnels, dus au Garbage Collector et à la surcharge mémoire. Rust et l'approche hybride, en revanche, affichent des courbes beaucoup plus stables et prévisibles.

En production, cette prévisibilité est cruciale. Un système qui répond en 200ms en moyenne mais avec des pics à 2 secondes est moins fiable qu'un système qui répond toujours en 250ms.

**Conclusion : l'approche hybride est notre choix pour la suite du projet.**

---

## 🏗️ Slide 8 — Architecture Multi-Agents
*(~90 secondes)*

Passons maintenant à l'architecture que nous avons implémentée cette semaine.

Nous avons conçu un système à 5 agents, **sans aucun framework externe** — pas de LangChain, pas d'AutoGen. Tout est implémenté en Python et Rust pur.

Le flux est le suivant : quand l'utilisateur pose une question, le **UserProxy** orchestre le pipeline. La requête passe d'abord par le **Raffineur**, qui la réécrit pour améliorer la récupération. Ensuite, le **Retriever** recherche les chunks pertinents dans la base vectorielle. Le **Générateur** produit une réponse en se basant uniquement sur ce contexte. Enfin, l'**Évaluateur** note la réponse.

Si le score est insuffisant, le système réessaie automatiquement — jusqu'à un maximum de k tentatives. Si le score est satisfaisant, la réponse est retournée à l'utilisateur.

---

## 🔗 Slide 9 — Le State Dict
*(~60 secondes)*

La question qui se pose naturellement est : comment ces agents communiquent-ils entre eux ?

Nous avons choisi un pattern simple et élégant : le **dictionnaire d'état partagé**. Chaque agent reçoit ce dictionnaire, lit ce dont il a besoin, écrit son résultat, et le passe au suivant.

Par exemple, le Raffineur lit `query` et écrit `refined_query`. Le Retriever lit `refined_query` et écrit `chunks`. Le Générateur lit `chunks` et écrit `answer`. L'Évaluateur lit `answer` et écrit `score` et `should_retry`.

Les avantages sont clairs : chaque agent est **indépendant**, le système est **extensible** — ajouter un nouvel agent revient simplement à ajouter des clés au dictionnaire — et l'état complet est **traçable** à tout moment.

---

## ⚖️ Slide 10 — LLM-as-a-Judge
*(~60 secondes)*

L'une des optimisations les plus importantes est notre évaluateur. Nous avons implémenté un pattern appelé **LLM-as-a-Judge**, qui est l'état de l'art pour l'évaluation automatique des systèmes RAG — utilisé notamment par RAGAS et TruLens.

Concrètement, l'évaluateur envoie la question, le contexte récupéré, et la réponse générée à un LLM, avec deux critères d'évaluation : la **fidélité** — est-ce que la réponse vient du contexte sans hallucination ? — et la **pertinence** — est-ce que la réponse répond vraiment à la question ?

Le LLM retourne un raisonnement en une phrase, puis un score entre 0 et 1 que nous extrayons par regex. Si le score est inférieur au seuil configuré et qu'il reste des tentatives, le système réessaie automatiquement.

---

## ⚡ Slide 11 — Retrieval et Cache
*(~45 secondes)*

Deux autres optimisations méritent d'être mentionnées.

La première concerne le modèle d'embedding BGE. Ce modèle nécessite un préfixe spécial pour les requêtes de recherche. En l'ajoutant, nous avons observé une amélioration significative de la qualité de récupération — la distance entre la requête et les chunks pertinents diminue sensiblement.

La deuxième est le **cache par hash SHA-256**. Lors de l'indexation, nous calculons un hash des PDFs et des paramètres de chunking. Si les PDFs n'ont pas changé, nous sautons complètement l'étape d'embedding — ce qui ramène le temps d'indexation de plusieurs minutes à **0.01 millisecondes** sur un cache hit.

---

## 🛠️ Slide 12 — Stack Technique
*(~40 secondes)*

Voici le stack complet que nous utilisons. Le langage principal est Python, avec des modules critiques en Rust via PyO3. Les embeddings sont générés par fastembed avec le runtime ONNX directement en Rust — ce n'est pas sentence-transformers comme initialement prévu, mais une solution plus rapide. La base vectorielle est LanceDB avec des bindings Rust. L'ingestion PDF utilise pdf-oxide avec Rayon pour le parallélisme. Et l'interface est exposée via FastAPI.

---

## 🎬 Slide 13 — Démonstration
*(~60-90 secondes)*

Je vais maintenant vous montrer le système en action.

*(Lancer la vidéo ou faire une démo live)*

On voit ici le pipeline complet : la requête est d'abord réécrite par le Raffineur, puis les chunks sont récupérés, la réponse est générée, et l'Évaluateur attribue un score. Dans cet exemple, le score est 0.95 — pas de réessai nécessaire.

---

## 🚀 Slide 14 — Prochaines Étapes
*(~45 secondes)*

Pour la semaine prochaine, nous avons trois priorités.

La première est le **retrieval hybride** — combiner BM25 et la recherche dense avec Reciprocal Rank Fusion. Un de nos papiers de référence montre un gain de plus de 8 points de pourcentage en Recall@5 avec cette approche.

La deuxième est le **routage conditionnel** — si la distance de récupération est déjà faible, on court-circuite le Raffineur et l'Évaluateur, réduisant les appels LLM de 4 à 2.

La troisième est la **migration vers Ollama** pour avoir un LLM local et réduire la latence réseau.

---

## 📋 Slide 15 — Plan d'Action
*(~20 secondes)*

Ce tableau résume l'état d'avancement. Trois tâches sont terminées — le cache, le LLM-as-a-Judge, et le préfixe BGE. Deux sont en cours — le retrieval hybride et le routage conditionnel. Et deux restent à faire pour les semaines suivantes.

---

## 🙏 Slide 16 — Merci
*(~15 secondes)*

Merci pour votre attention. Je suis disponible pour répondre à vos questions.

---

## 📝 Notes pour Questions Fréquentes

**Q: Pourquoi Rust et pas Python pur ?**
Les modules critiques comme l'embedding, le chunking et la recherche vectorielle sont des opérations intensives. Rust nous donne un contrôle fin sur les threads et la mémoire, sans Garbage Collector, ce qui explique la stabilité et la performance observées dans les benchmarks.

**Q: Pourquoi ne pas utiliser LangChain ?**
Nous voulions comprendre et maîtriser chaque composant du système. Les frameworks comme LangChain ajoutent de l'abstraction et des dépendances. Notre implémentation custom est plus légère, plus rapide, et entièrement sous notre contrôle.

**Q: Le LLM-as-a-Judge n'est-il pas trop coûteux en appels API ?**
C'est un vrai compromis. En pratique, si le premier essai score bien (>0.75), on n'a pas de réessai. Et nous travaillons sur le routage conditionnel pour éviter l'évaluation quand la confiance de récupération est déjà élevée.

**Q: Quelle est la différence entre le Raffineur et le Générateur ?**
Le Raffineur réécrit la *question* pour améliorer la recherche — il ne répond pas. Le Générateur reçoit le contexte récupéré et produit la *réponse finale*. Ce sont deux rôles distincts avec des prompts très différents.
"""

#### 1) RULES

Rules for evaluation of KG triples against human-created triples gold standard:

1) RULE 1: It should be evaluated whether the triples have the correct direction and type of relationship/hierarchy, 
e.g. Josh - is son of -> Andrew (correct) vs. Andrew -is son of -> Josh (wrong)

i.e., checking the logical/relationship between the subjects and objects 
See RULE 3 for the one exception to this: a subject/object swap paired with
the predicate's logical inverse is NOT a directionality error, but a
same-predicate swap always is.


2) RULE 2: Both the entities but also especially the relationship evaluation should account for semantic similarity and not an exact match 
- to catch words with similar meaning, but different spelling. 


3) RULE 3: A predicted triple that swaps subject and object relative to the
gold triple should still be counted as a correct match if the predicate is
also replaced by its logical inverse - since this expresses the identical
real-world fact, just stated from the other entity's perspective, rather
than a genuine directionality error.

e.g. Josh - is son of -> Andrew (gold)
     Andrew - is father of -> Josh (predicted)  ->  CORRECT
     (same fact: "Josh is Andrew's son" = "Andrew is Josh's father")

     vs. Andrew - is son of -> Josh (predicted, SAME predicate, swapped
     entities)  ->  WRONG (this is the RULE 1 violation: it claims the
     opposite parent-child direction, not a restatement of the same fact)

Real example from the Altenheim toy dataset:
     Bewohner - istEin -> Akteur (gold)
     Akteure - umfassen -> Bewohner (predicted)  ->  CORRECT
     (same fact: "a resident IS-A type of actor" = "actors COMPRISE residents")

     vs. Akteur - istEin -> Bewohner (predicted, SAME predicate "istEin",
     swapped entities)  ->  WRONG (this claims actors are a type of resident,
     the reverse of the true class hierarchy - a genuine RULE 1 violation)

i.e., distinguishing a same-predicate entity swap (a real error, RULE 1)
from an inverse-predicate entity swap (an equivalent restatement, RULE 3).


#### 2) PROCESS

Two independent evaluation approaches are run in parallel, each with its own
complete Stage 1 (entity alignment) + Stage 2 (triple matching). Their outputs
are compared afterward rather than merged into a single pipeline — this keeps
disagreement between the two visible and inspectable, instead of hiding it
inside one collapsed score.


##### APPROACH A — Embedding-based

1) Stage 1A - Entity alignment (handles RULE 2 for entities)

Match predicted entities to gold entities using embedding similarity, not exact string match
- manual verification necessary for embedding threshold determination


2) Stage 2A - Triple matching (handles RULE 1 & 2 for relations)

For each gold triple (s_gold, p_gold, o_gold), check candidate predicted triples where:

- s_pred aligns to s_gold and o_pred aligns to o_gold positionally (via Stage 1A's
  alignment) - this enforces RULE 1 (directionality/hierarchy), since
  (Josh, son of, Andrew) won't match (Andrew, son of, Josh) unless entities are swapped
- p_pred is semantically similar to p_gold above some threshold (embedding
  similarity, not string match) - handling RULE 2 for relations
- a small curated inverse-predicate lookup (e.g. istEin <-> umfassen) additionally
  permits a swapped-entity match when the predicate is a known inverse pair.
  This is an interim, manually-maintained stopgap — not automatically scalable
  to larger corpora 



  ##### APPROACH B — LLM-based

1) Stage 1B - Entity alignment (handles RULE 2 for entities)

Match predicted entities to gold entities using LLM judgment of real-world-concept
equivalence, not embedding similarity or exact string match
- no fixed threshold to calibrate; disagreement with Approach A serves as the
  review signal instead

2) Stage 2B - Triple matching (handles RULE 1 & 2 for relations)

For each gold triple (s_gold, p_gold, o_gold), check candidate predicted triples where:

- entities align (via Stage 1B) in EITHER the direct order (s_pred=s_gold,
  o_pred=o_gold) OR the swapped order (s_pred=o_gold, o_pred=s_gold) - both
  orientations are always tried, since there is no fixed lookup deciding
  which predicates might be inverses ahead of time
- the LLM judges whether p_pred expresses the SAME relation as p_gold (direct
  orientation) or the LOGICAL INVERSE of p_gold (swapped orientation)
- this handles inverse-predicate pairs (e.g. istEin vs umfassen) automatically,
  generalizing to any predicate pair the LLM can reason about, without needing
  a hardcoded lookup table - this is the key structural advantage over Approach A

  
##### COMPARISON

Results from Approach A and Approach B are kept separate, at both the entity-
alignment level (Stage 1A vs 1B) and the triple-matching level (Stage 2A vs 2B).
Disagreements are flagged as a manual-review queue.



##### STAGE 3 — Evaluation Metrics

Using the matched/unmatched flags produced by Stage 2A and Stage 2B, Stage 3
computes:

- Primary: Precision, Recall, Micro-F1, and Macro-F1 (per gold relation type),
  computed separately for Approach A and Approach B. Macro-F1 requires
  attributing false positives to a specific relation type where possible
  (see note below on the small Stage 2 addition this requires).

- Secondary: a directionality-error breakdown, re-classifying a portion of
  the unmatched gold/predicted triples as specifically "reversed-direction"
  errors (same predicate, swapped entities - a genuine RULE 1 violation)
  rather than lumping them in with generic misses or hallucinations. This
  isolates how much of kg-gen's error is a directionality problem
  specifically, as opposed to missing or fabricated facts.



#### 3) EVALUATION METRICS


1) Primary:

- Precision (FP should be punished; e.g. LLM hallucinations)

- Recall (FN should be punished; e.g., LLM KG is incomplete)

- F1-Score 
    - Micro F1: (it matters because it reflects your extractor's aggregate performance as experienced across the whole triple set, but gets dominated by whichever relation types are most frequent.)
    - Macro F1: (computes F1 separately per relation type and then averages those scores unweighted; it matters because it exposes systematic failures on rare-but-meaningful relation types)

    
Literature, e.g.: https://www.mdpi.com/2076-3417/15/7/3727


2) Secondary:

- swapped-triple check -> isolating directionality errors


3) Tertiary:

- Pearson correlation coefficient: relationship between embedding similarities and ground-truth semantic relationships


Literature, e.g.: https://www.mdpi.com/2076-3417/15/7/3727

"""




""" 

#### PSEUDOCODE 


##### APPROACH A: EMBEDDING BASED

###### STAGE 1A - ENTITY ALIGNMENT:


INPUT:
    gold_entities:      list of strings, e.g. ["Akteur", "Altenheim", "Raum", ...]
    predicted_entities:  list of strings, e.g. ["Akteure", "Räume", "Zimmer", ...]
    similarity_threshold: float, e.g. 0.75  (to be calibrated later via Pearson correlation, see Tertiary metric)
    embedding_model: multilingual sentence embedding model (e..g, OpenAI text-embedding-3-large)


STEP 1 — Embed all entities
    gold_embeddings      = embedding_model.encode(gold_entities)       # shape: (n_gold, dim)
    predicted_embeddings = embedding_model.encode(predicted_entities)  # shape: (n_pred, dim)

STEP 2 — Compute full similarity matrix
    similarity_matrix = cosine_similarity(predicted_embeddings, gold_embeddings)
    # similarity_matrix[i][j] = similarity between predicted_entities[i] and gold_entities[j]


STEP 3 — For each predicted entity, find its best gold match
    entity_alignment = {}   # maps predicted_entity -> (matched_gold_entity, score) or None

    FOR i, pred_entity IN enumerate(predicted_entities):
        best_j     = argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_j]

        IF best_score >= similarity_threshold:
            entity_alignment[pred_entity] = (gold_entities[best_j], best_score)
        ELSE:
            entity_alignment[pred_entity] = None   # no gold concept matched -> flag as
                                                    # unaligned / potentially hallucinated entity
                                                  


##### STAGE 2A: Triple Matching with Inverse-Predicate Normalization


STEP 4 — Precompute predicate similarity matrix
    gold_edges, predicted_edges = unique predicates in each relation list
    predicate_similarity_matrix = cosine_similarity(embed(predicted_edges), embed(gold_edges))

STEP 5 — Build both orientations per predicted triple (direct + inverse-lookup)
    FOR each predicted_triple (s_pred, p_pred, o_pred) IN predicted_relations:
        orientations = [ (s_pred, p_pred, o_pred, flipped=False) ]
        IF p_pred IN INVERSE_PREDICATES:
            orientations.append( (o_pred, INVERSE_PREDICATES[p_pred], s_pred, flipped=True) )
        predicted_triple.orientations = orientations

STEP 6 — Match each gold triple against all predicted triples/orientations
    FOR each gold_triple (s_gold, p_gold, o_gold) IN gold_relations:
        candidates = []
        FOR each predicted_triple IN predicted_relations:
            FOR each (s_o, p_o, o_o, flipped) IN predicted_triple.orientations:
                s_match = entity_alignment_A.get(s_o)
                o_match = entity_alignment_A.get(o_o)
                IF s_match IS None OR o_match IS None: CONTINUE
                IF s_match.gold_entity != s_gold OR o_match.gold_entity != o_gold: CONTINUE

                p_score = predicate_similarity_matrix[p_o][p_gold]
                IF p_score >= relation_threshold:
                    candidates.append((predicted_triple, p_score, flipped))

        IF candidates is non-empty:
            gold_triple.matched_A = True
            gold_triple.best_match_A = argmax(candidates, key=score)
            FOR (predicted_triple, score, flipped) IN candidates:
                predicted_triple.matched_A = True
        ELSE:
            gold_triple.matched_A = False

OUTPUT:
    entity_alignment_A,
    gold_matched_count_A  = count(gold_relations, matched_A == True),
    predicted_matched_count_A = count(predicted_relations, matched_A == True)


    
#### STAGE 3 — EVALUATION METRICS

INPUT:
    gold_relations, predicted_relations       # with matched_A/B, near_misses_A/B
    entity_alignment_A, entity_alignment_B
    predicate_similarity_matrix               # from Stage 2A (reused here, not recomputed)
    relation_threshold


# ============================================================
# 1) PRIMARY METRICS — Precision, Recall, Micro-F1, Macro-F1
#    (computed once per approach: A and B)
# ============================================================

STEP 1 — Micro-level Precision / Recall / F1 (pooled, unaffected by type attribution)
    FOR approach IN [A, B]:
        gold_matched_count      = count(gold_relations, matched_<approach> == True)
        predicted_matched_count = count(predicted_relations, matched_<approach> == True)

        Recall_micro_<approach>    = gold_matched_count / len(gold_relations)
        Precision_micro_<approach> = predicted_matched_count / len(predicted_relations)
        F1_micro_<approach>        = harmonic_mean(Precision_micro_<approach>, Recall_micro_<approach>)

STEP 2 — Per-relation-type confusion counts (for Macro-F1)
    FOR approach IN [A, B]:
        gold_types = unique predicates in gold_relations

        FOR each type t IN gold_types:
            gold_of_type = [g for g in gold_relations if g.p_gold == t]

            TP_t = count(gold_of_type, matched_<approach> == True)
            FN_t = count(gold_of_type, matched_<approach> == False)

            # FP_t: near-miss predicted triples attributed to this type —
            # entities aligned correctly to a gold triple of type t, but the
            # relation didn't clear the bar. This is well-defined because
            # near_misses_<approach> already ties each near-miss to the
            # specific gold triple (and therefore type) it was checked against.
            FP_t = sum(len(g.near_misses_<approach>) for g IN gold_of_type)

            Precision_t = TP_t / (TP_t + FP_t)  IF (TP_t + FP_t) > 0 ELSE 0
            Recall_t    = TP_t / (TP_t + FN_t)  IF (TP_t + FN_t) > 0 ELSE 0
            F1_t        = harmonic_mean(Precision_t, Recall_t)

            per_type_scores_<approach>[t] = {"P": Precision_t, "R": Recall_t, "F1": F1_t,
                                               "support": len(gold_of_type)}

        # Predicted triples whose entities never aligned to ANY gold pair at
        # all (pure hallucinations, no near-miss attribution possible) —
        # these still count in Step 1's micro precision denominator, but
        # can't inform a per-type macro score. Reported separately so they
        # aren't silently dropped from the analysis.
        unattributed_FP_<approach> = [p for p IN predicted_relations
                                        if matched_<approach> == False
                                        AND p not in any near_misses_<approach> list]

STEP 3 — Macro-F1
    FOR approach IN [A, B]:
        Macro_F1_<approach> = mean([scores["F1"] for scores IN per_type_scores_<approach>.values()])
        # unweighted average across relation TYPES, not across triples —
        # this is what exposes rare-type failures that Micro-F1 hides


# ============================================================
# 2) SECONDARY METRIC — Directionality-error breakdown
#    (reclassifies existing FN/FP, no new comparisons needed)
# ============================================================

STEP 4 — Tag reversed-direction False Negatives
    FOR approach IN [A, B]:
        FOR each gold_triple (s_gold, p_gold, o_gold) IN gold_relations WHERE matched_<approach> == False:

            reversed_found = EXISTS predicted_triple (s_pred, p_pred, o_pred) IN predicted_relations
                WHERE entity_alignment_<approach>[s_pred] == o_gold        # swapped
                  AND entity_alignment_<approach>[o_pred] == s_gold        # swapped
                  AND p_pred == p_gold   # SAME predicate, not inverse — this is
                                         # deliberately a strict/exact check: this
                                         # diagnostic hunts specifically for the
                                         # textbook RULE 1 violation, not general
                                         # semantic near-matches

            gold_triple.fn_type_<approach> = "reversed_direction" IF reversed_found ELSE "missed_fact"

STEP 5 — Tag reversed-direction False Positives (symmetric check)
    FOR approach IN [A, B]:
        FOR each predicted_triple (s_pred, p_pred, o_pred) IN predicted_relations WHERE matched_<approach> == False:

            reversed_found = EXISTS gold_triple (s_gold, p_gold, o_gold) IN gold_relations
                WHERE entity_alignment_<approach>[s_pred] == o_gold
                  AND entity_alignment_<approach>[o_pred] == s_gold
                  AND p_pred == p_gold

            predicted_triple.fp_type_<approach> = "reversed_direction" IF reversed_found ELSE "hallucination_or_wrong_fact"

STEP 6 — Summarize
    FOR approach IN [A, B]:
        directionality_error_rate_<approach> = (
            count(gold_relations, fn_type_<approach> == "reversed_direction") +
            count(predicted_relations, fp_type_<approach> == "reversed_direction")
        ) / (len(gold_relations) + len(predicted_relations))
        # reported alongside Primary metrics: "X% of total errors under
        # Approach A/B were specifically directionality violations"

    

#### APPROACH B — LLM-Based Entity Alignment & Triple Matching

INPUT:
    gold_entities:        list of strings, e.g. ["Akteur", "Altenheim", "Raum", ...]
    predicted_entities:   list of strings, e.g. ["Akteure", "Räume", "Zimmer", ...]
    gold_relations:       list of (s_gold, p_gold, o_gold) triples
    predicted_relations:  list of (s_pred, p_pred, o_pred) triples
    llm_client:           LLM API client


# ============================================================
# STAGE 1B — Entity Alignment (handles RULE 2 for entities)
# ============================================================

STEP 1 — Batch LLM entity alignment (single call, not one per entity)
    entity_alignment_B = llm_client.query(
        predicted_entities = predicted_entities,
        gold_entities      = gold_entities,
        instruction = "For each predicted entity below, return the gold entity
                        that refers to the same real-world concept (accounting
                        for singular/plural, compound phrasing, or synonymy),
                        or 'None' if no gold entity corresponds to it."
    )
    # -> dict: predicted_entity -> gold_entity | None
    # No numeric threshold here — the LLM makes a categorical judgment directly,
    # unlike Approach A's cosine-similarity threshold.

STEP 2 — (bookkeeping, not scoring yet)
    unaligned_predicted_entities = [e for e, match in entity_alignment_B.items() if match is None]
    # Keep this list for precision-side error analysis later, same role as
    # Approach A's equivalent list.


# ============================================================
# STAGE 2B — Triple Matching (handles RULE 1 & RULE 2 for relations,
#             and RULE 3 for inverse-predicate equivalence)
# ============================================================

STEP 3 — Batched matching: each gold triple checked against the FULL predicted list at once
    FOR each gold_triple (s_gold, p_gold, o_gold) IN gold_relations:

        response = llm_client.query(
            gold_triple       = (s_gold, p_gold, o_gold),
            candidate_triples = predicted_relations,   # entire list, sent as one block
            instruction = "Below is one gold-standard fact and a list of
                            candidate predicted triples. Identify ALL predicted
                            triples (if any) that express the SAME real-world
                            fact as the gold triple. This includes cases where:
                            - entities are phrased differently but refer to the
                              same concept (singular/plural, compound phrasing,
                              synonyms)                                    [RULE 2]
                            - the predicate is phrased differently but means
                              the same thing                                [RULE 2]
                            - the predicate is the LOGICAL INVERSE of the gold
                              predicate, with subject and object swapped
                              (e.g. 'X isA Y' matches 'Y comprises X')       [RULE 3]
                            Do NOT count a match where subject and object are
                            swapped but the predicate stays the SAME — that is
                            a genuine directionality error, not a match.     [RULE 1]
                            Return the matching predicted triples as a list
                            (empty, one, or several), quoted verbatim from
                            the candidate list."
        )

        matched_triples = response.matching_triples   # list, possibly empty

STEP 4 — Validate the response before trusting it
        FOR each triple IN matched_triples:
            IF triple NOT IN predicted_relations:
                DISCARD triple   # guards against the LLM paraphrasing or
                                  # hallucinating a triple that wasn't actually
                                  # in the candidate list it was given
            # log discarded cases separately — a high discard rate signals the
            # prompt needs tightening (e.g. stronger "quote verbatim" instruction)

STEP 5 — Record matches
        IF matched_triples (post-validation) is non-empty:
            gold_triple.matched_B = True
            gold_triple.matches_B = matched_triples
            FOR predicted_triple IN matched_triples:
                predicted_triple.matched_B = True
        ELSE:
            gold_triple.matched_B = False


OUTPUT:
    entity_alignment_B,          # dict: predicted_entity -> gold_entity | None
    unaligned_predicted_entities,
    gold_matched_count_B      = count(gold_relations, matched_B == True),
    predicted_matched_count_B = count(predicted_relations, matched_B == True)
    # gold_matched_count_B / len(gold_relations)      -> Recall (Approach B)
    # predicted_matched_count_B / len(predicted_relations) -> Precision (Approach B)


    
#### STAGE 3 — EVALUATION METRICS

INPUT:
    gold_relations, predicted_relations       # with matched_A/B, near_misses_A/B
    entity_alignment_A, entity_alignment_B
    predicate_similarity_matrix               # from Stage 2A (reused here, not recomputed)
    relation_threshold


# ============================================================
# 1) PRIMARY METRICS — Precision, Recall, Micro-F1, Macro-F1
#    (computed once per approach: A and B)
# ============================================================

STEP 1 — Micro-level Precision / Recall / F1 (pooled, unaffected by type attribution)
    FOR approach IN [A, B]:
        gold_matched_count      = count(gold_relations, matched_<approach> == True)
        predicted_matched_count = count(predicted_relations, matched_<approach> == True)

        Recall_micro_<approach>    = gold_matched_count / len(gold_relations)
        Precision_micro_<approach> = predicted_matched_count / len(predicted_relations)
        F1_micro_<approach>        = harmonic_mean(Precision_micro_<approach>, Recall_micro_<approach>)

STEP 2 — Per-relation-type confusion counts (for Macro-F1)
    FOR approach IN [A, B]:
        gold_types = unique predicates in gold_relations

        FOR each type t IN gold_types:
            gold_of_type = [g for g in gold_relations if g.p_gold == t]

            TP_t = count(gold_of_type, matched_<approach> == True)
            FN_t = count(gold_of_type, matched_<approach> == False)

            # FP_t: near-miss predicted triples attributed to this type —
            # entities aligned correctly to a gold triple of type t, but the
            # relation didn't clear the bar. This is well-defined because
            # near_misses_<approach> already ties each near-miss to the
            # specific gold triple (and therefore type) it was checked against.
            FP_t = sum(len(g.near_misses_<approach>) for g IN gold_of_type)

            Precision_t = TP_t / (TP_t + FP_t)  IF (TP_t + FP_t) > 0 ELSE 0
            Recall_t    = TP_t / (TP_t + FN_t)  IF (TP_t + FN_t) > 0 ELSE 0
            F1_t        = harmonic_mean(Precision_t, Recall_t)

            per_type_scores_<approach>[t] = {"P": Precision_t, "R": Recall_t, "F1": F1_t,
                                               "support": len(gold_of_type)}

        # Predicted triples whose entities never aligned to ANY gold pair at
        # all (pure hallucinations, no near-miss attribution possible) —
        # these still count in Step 1's micro precision denominator, but
        # can't inform a per-type macro score. Reported separately so they
        # aren't silently dropped from the analysis.
        unattributed_FP_<approach> = [p for p IN predicted_relations
                                        if matched_<approach> == False
                                        AND p not in any near_misses_<approach> list]

STEP 3 — Macro-F1
    FOR approach IN [A, B]:
        Macro_F1_<approach> = mean([scores["F1"] for scores IN per_type_scores_<approach>.values()])
        # unweighted average across relation TYPES, not across triples —
        # this is what exposes rare-type failures that Micro-F1 hides


# ============================================================
# 2) SECONDARY METRIC — Directionality-error breakdown
#    (reclassifies existing FN/FP, no new comparisons needed)
# ============================================================

STEP 4 — Tag reversed-direction False Negatives
    FOR approach IN [A, B]:
        FOR each gold_triple (s_gold, p_gold, o_gold) IN gold_relations WHERE matched_<approach> == False:

            reversed_found = EXISTS predicted_triple (s_pred, p_pred, o_pred) IN predicted_relations
                WHERE entity_alignment_<approach>[s_pred] == o_gold        # swapped
                  AND entity_alignment_<approach>[o_pred] == s_gold        # swapped
                  AND p_pred == p_gold   # SAME predicate, not inverse — this is
                                         # deliberately a strict/exact check: this
                                         # diagnostic hunts specifically for the
                                         # textbook RULE 1 violation, not general
                                         # semantic near-matches

            gold_triple.fn_type_<approach> = "reversed_direction" IF reversed_found ELSE "missed_fact"

STEP 5 — Tag reversed-direction False Positives (symmetric check)
    FOR approach IN [A, B]:
        FOR each predicted_triple (s_pred, p_pred, o_pred) IN predicted_relations WHERE matched_<approach> == False:

            reversed_found = EXISTS gold_triple (s_gold, p_gold, o_gold) IN gold_relations
                WHERE entity_alignment_<approach>[s_pred] == o_gold
                  AND entity_alignment_<approach>[o_pred] == s_gold
                  AND p_pred == p_gold

            predicted_triple.fp_type_<approach> = "reversed_direction" IF reversed_found ELSE "hallucination_or_wrong_fact"

STEP 6 — Summarize
    FOR approach IN [A, B]:
        directionality_error_rate_<approach> = (
            count(gold_relations, fn_type_<approach> == "reversed_direction") +
            count(predicted_relations, fp_type_<approach> == "reversed_direction")
        ) / (len(gold_relations) + len(predicted_relations))
        # reported alongside Primary metrics: "X% of total errors under
        # Approach A/B were specifically directionality violations"

"""



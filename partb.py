import spacy
import csv
from collections import Counter
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


nlp = spacy.load("en_core_web_trf")

# Basic SRL function using spaCy dependency parse
def spacy_srl(sent):
    # pick the root predicate
    root = [t for t in sent if t.dep_ == "ROOT"]
    if not root:
        return None
    v = root[0]

    # subject in active:
    subj = next((c for c in v.children if c.dep_ in {"nsubj", "nsubjpass"}), None)

    # agent in passive: 
    agent = None
    agent_tok = next((c for c in v.children if c.dep_ == "agent"), None)
    if agent_tok:
        # agent phrase head is often the object of the "by" preposition
        agent = next((c for c in agent_tok.children if c.dep_ in {"pobj", "obj"}), None)

    # object/result complements:
    dobj = next((c for c in v.children if c.dep_ in {"dobj", "obj"}), None)
    attr = next((c for c in v.children if c.dep_ in {"attr", "oprd", "acomp", "xcomp"}), None)

    # prepositional modifiers and adverbs
    # preps = [c for c in v.children if c.dep_ == "prep"]
    # advmods = [c for c in v.children if c.dep_ == "advmod"]

    def subtree_text(tok):
        return " ".join([t.text for t in tok.subtree]) if tok else None

    voice = "passive" if any(c.dep_ == "nsubjpass" for c in v.children) else "active"
    agent = subtree_text(agent) if agent else None
    roles = {
        "predicate": v.lemma_,
        "voice": voice,
        "ARG0_agent": agent,
        "agent_and_voice": f"{agent}, {voice}",
        "ARG1_patient": subtree_text(subj) if subj and subj.dep_ == "nsubjpass" else subtree_text(dobj),
        "result_or_attribute": subtree_text(attr),
    }
    return roles

# Improved SRL cleaning noise in subjects / objects
ACTOR_ENTS = {"PERSON", "ORG", "GPE", "NORP"}
def is_pronoun_token(tok):
    return tok.pos_ == "PRON"

def first_entity_in_subtree(tok, allowed=ACTOR_ENTS):
    # pick the first entity whose span overlaps tok.subtree
    subtree_tokens = list(tok.subtree)
    subtree_start = subtree_tokens[0].idx
    subtree_end = subtree_tokens[-1].idx + len(subtree_tokens[-1])
    for ent in tok.doc.ents:
        if ent.label_ in allowed:
            if ent.start_char >= subtree_start and ent.end_char <= subtree_end:
                return ent.text, ent.label_
    return None, None

def noun_chunk_covering(tok):
    # find noun chunk that contains tok
    for nc in tok.doc.noun_chunks:
        if nc.start <= tok.i < nc.end:
            return nc.text
    return None

def canonical_mention(tok, prefer_ents=True, max_words=6):
    """
    Return a short canonical mention for an argument token:
    - Prefer named entity within subtree (PERSON/ORG/GPE/NORP)
    - Else use noun chunk covering token
    - Else pronoun / token text
    """
    if tok is None:
        return None, None

    # Pronouns: keep minimal
    if is_pronoun_token(tok):
        return tok.text, "PRON"

    if prefer_ents:
        ent_text, ent_label = first_entity_in_subtree(tok)
        if ent_text:
            return ent_text, ent_label

    nc = noun_chunk_covering(tok)
    if nc:
        # truncate long chunks for stability
        words = nc.split()
        nc_short = " ".join(words[:max_words])
        return nc_short, "NP"

    # fallback
    return tok.text, tok.ent_type_ or tok.pos_

def find_passive_agent(verb):
    """
    In passive, agent is often encoded as: 'agent' -> 'by' phrase -> pobj.
    """
    agent_prep = next((c for c in verb.children if c.dep_ == "agent"), None)
    if agent_prep:
        pobj = next((c for c in agent_prep.children if c.dep_ in {"pobj","obj"}), None)
        return pobj
    # fallback: explicit "by" preposition
    by_prep = next((c for c in verb.children if c.dep_ == "prep" and c.lower_ == "by"), None)
    if by_prep:
        pobj = next((c for c in by_prep.children if c.dep_ in {"pobj","obj"}), None)
        return pobj
    return None

def spacy_srl_improved(sent):
    """
    One main predicate frame per sentence ROOT verb.
    For complex sentences you can extend to handle conj verbs; see note below.
    """
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None:
        return None

    # if ROOT isn't a verb (rare), try to find a main verb child
    if root.pos_ not in {"VERB", "AUX"}:
        root = next((c for c in root.children if c.pos_ in {"VERB", "AUX"}), root)

    # detect passive by presence of nsubjpass
    nsubjpass = next((c for c in root.children if c.dep_ == "nsubjpass"), None)
    nsubj = next((c for c in root.children if c.dep_ == "nsubj"), None)
    dobj = next((c for c in root.children if c.dep_ in {"dobj","obj"}), None)

    voice = "passive" if nsubjpass is not None else "active"

    # ARG0 / ARG1 tokens
    if voice == "active":
        arg0_tok = nsubj
        arg1_tok = dobj
    else:
        arg1_tok = nsubjpass  # patient in passive
        arg0_tok = find_passive_agent(root)  # may be None

    # result/attribute complements (useful for "made X Y", "became X")
    attr_tok = next((c for c in root.children if c.dep_ in {"attr","oprd","acomp","xcomp"}), None)

    arg0_text, arg0_type = canonical_mention(arg0_tok)
    arg1_text, arg1_type = canonical_mention(arg1_tok)
    attr_text, attr_type = canonical_mention(attr_tok, prefer_ents=False, max_words=8)

    return {
        "predicate": root.lemma_,
        "voice": voice,
        "ARG0_agent": arg0_text,
        #"ARG0_type": (arg0_text, arg0_type),
        "ARG0_type": arg0_type,
        "ARG1_patient": arg1_text,
        #"ARG1_type": (arg1_text, arg1_type),
        "ARG1_type": arg1_type,
        "agent_and_voice": f"{arg0_text}, {voice}",
        "result_or_attribute": attr_text,
        #"attr_type": (attr_text, attr_type),
        "attr_type": attr_type,
    }

def role_mapping(sents, type="baseline"):
    if type == "baseline":
        srl_func = spacy_srl
        mapping = {"voice": [], "predicate": [], "ARG0_agent": [], "agent_and_voice": [], "ARG1_patient": [], "result_or_attribute": []}
    else:
        srl_func = spacy_srl_improved
        mapping = {"voice": [], "predicate": [], "ARG0_agent": [], "ARG0_type": [], "agent_and_voice": [], "ARG1_patient": [], "ARG1_type": [], "result_or_attribute": [], "attr_type": []}
    for sent in sents:
        roles = srl_func(sent)
        if roles:
            for k, v in roles.items():
                if v is not None:
                    mapping[k].append(v)
    for key, value in mapping.items():
        mapping[key] = Counter(value)
    return mapping

def srl_dataset(data, name, type="baseline"):
    print(f"Performing SRL analysis on dataset - {name} with srl model {type}")
    passive_count = 0
    passive_no_agent_count = 0
    actors = {'agents': Counter(), 'patients': Counter(), 'attributes': Counter()}
    if type == "improved":
        actor_types = {'ARG0_agent_type': Counter(), 'ARG1_patient_type': Counter(), 'attr_type': Counter()}
    sent_count = 0
    for article in data:
        doc = nlp(article['text'])
        sents = [sent for sent in doc.sents]
        sent_count += len(sents)
        mapping = role_mapping(sents, type)
        # calculate passive_rate
        passive_count += mapping["voice"]['passive']
        # passive_rate = map["voice"]['passive']/count_sent
        # avg_passive_rate += round(passive_rate,2)
        # calculate passive_no_agent_rate
        passive_no_agent_count += mapping["agent_and_voice"]['None, passive']
        # avg_passive_no_agent_rate += round(passive_no_agent_count/count_sent,2)
        # evaluate the semantic field of agent, patient, and attribute
        actors['agents'] += mapping["ARG0_agent"]
        actors['patients'] += mapping["ARG1_patient"]
        actors['attributes'] += mapping["result_or_attribute"] 
        if type == "improved":
            actor_types['ARG0_agent_type'] += mapping["ARG0_type"]
            actor_types['ARG1_patient_type'] += mapping["ARG1_type"]
            actor_types['attr_type'] += mapping["attr_type"]
    avg_passive_rate = passive_count/sent_count
    avg_passive_no_agent_rate = passive_no_agent_count/sent_count
    top_agents = actors['agents'].most_common(10)
    top_patients = actors['patients'].most_common(10)
    top_attributes = actors['attributes'].most_common(10)
    if type == "improved":
        top_agents = top_agents, actor_types['ARG0_agent_type'].most_common()
        top_patients = top_patients, actor_types['ARG1_patient_type'].most_common()
        top_attributes = top_attributes, actor_types['attr_type'].most_common()

    # data visualisation
    print("="*20)
    print(f"Semantic Role Labeling Analysis on {name}")
    print(f"Average Passive Rate: {avg_passive_rate}")
    print(f"Average Passive No Agent Rate: {avg_passive_no_agent_rate}")
    """if type == "baseline":
        print("Top 10 Agents:")
        for agent, count in top_agents:
            print(f"{agent}: {count}")
        print("Top 10 Patients:")
        for patient, count in top_patients:
            print(f"{patient}: {count}")
        print("Top 10 Attributes:")
        for attribute, count in top_attributes:
            print(f"{attribute}: {count}")"""
    if type == "improved":
        agent, a_types = top_agents
        patient, p_types = top_patients
        attribute, attr_types = top_attributes
        """print("Top 10 Agents:")
        for ag, count in agent:
            print(f"{ag}: {count}")
        print("Agent Types:")
        for atype, count in a_types:
            print(f"{atype}: {count}")
        print("Top 10 Patients:")
        for pa, count in patient:
            print(f"{pa}: {count}")
        print("Patient Types:")
        for ptype, count in p_types:
            print(f"{ptype}: {count}")
        print("Top 10 Attributes:")
        for at, count in attribute:
            print(f"{at}: {count}")
        print("Attribute Types:")
        for attrtype, count in attr_types:
            print(f"{attrtype}: {count}")
    print("="*20)"""

    return avg_passive_rate, avg_passive_no_agent_rate, top_agents, top_patients, top_attributes

ENT_LIST = ["PERSON", "ORG", "GPE", "NORP", "PRON", "NP"]
def prepare_agent_type_data(bbc_types, tnh_types, ent_list=ENT_LIST):
    data = []
    bbc_dict = dict(bbc_types)
    tnh_dict = dict(tnh_types)
    bbc_tot = sum([count for ent, count in bbc_types])
    tnh_tot = sum([count for ent, count in tnh_types])
    for ent in ent_list:
        bbc_count = bbc_dict.get(ent, 0)
        tnh_count = tnh_dict.get(ent, 0)
        data.append({"Entity_Type": ent, "Dataset": "BBC", "Percentage": bbc_count/bbc_tot})
        data.append({"Entity_Type": ent, "Dataset": "TNH", "Percentage": tnh_count/tnh_tot})
    return data

def plot_top_actors(agents1, agents2, actor, title1 = "BBC", title2 = "TNH"):
    print(f"Plotting top {actor} terms...")
    bbc_df = pd.DataFrame(agents1.items(), columns=["term", "count"]).sort_values("count", ascending=False).head(10).reset_index(drop=True)
    tnh_df = pd.DataFrame(agents2.items(), columns=["term", "count"]).sort_values("count", ascending=False).head(10).reset_index(drop=True)
    y = np.arange(len(bbc_df))

    bbc_labels  = [f"{r}. {t}" for r, t in zip(range(1, len(bbc_df) + 1), bbc_df["term"].astype(str))]
    tnh_labels = [f"{r}. {t}" for r, t in zip(range(1, len(tnh_df) + 1), tnh_df["term"].astype(str))]

    plt.figure(figsize=(12, 6))
    plt.barh(y - 0.2, -bbc_df["count"], height=0.4, label=title1)
    plt.barh(y + 0.2, tnh_df["count"], height=0.4, label=title2)
    plt.axvline(0, color="black", linewidth=1)

    ax = plt.gca()
    # Replace tick-label parsing with formatting from numeric tick values
    xticks = ax.get_xticks()
    try:
        xtick_labels = [str(int(abs(x))) for x in xticks]
    except Exception:
        # fallback: use the original text if numeric conversion fails
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticklabels(xtick_labels)
    
    ax.set_yticks(y)
    ax.set_yticklabels(bbc_labels, fontsize=10)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels(tnh_labels, fontsize=10)

    ax.set_xlabel("Word count")
    ax.set_ylabel("BBC top terms by rank")
    ax2.set_ylabel("TNH top terms by rank")

    plt.title(f"Top 10 {actor} Terms Comparison between BBC and TNH")
    plt.tight_layout()
    # show and then close to avoid blocking further script execution
    try:
        plt.show()
    finally:
        plt.close()


def main():
    with open("final/mixed_data.csv", "r") as f:
        reader = csv.DictReader(f)
        mixed_data = [row for row in reader]
        bbc_data = [row for row in mixed_data if row['label'] == 'bbc']
        tnh_data = [row for row in mixed_data if row['label'] == 'tnh']
        print(f"Total BBC articles: {len(bbc_data)}")
        print(f"Total TNH articles: {len(tnh_data)}")

    bbc_passive_rate, bbc_avg_passive_no_agent_rate, bbc_top_agents, bbc_top_patients, bbc_top_attributes=srl_dataset(bbc_data, "BBC", 'improved')
    tnh_passive_rate, tnh_avg_passive_no_agent_rate, tnh_top_agents, tnh_top_patients, tnh_top_attributes=srl_dataset(tnh_data, "TNH", 'improved')

    # plot bar chart between passive_rate or passive_no_agent_rate of BBC and TNH

    data_passive = [
    {"Dataset": "BBC", "Metric": "Passive rate", "Value": bbc_passive_rate},
    {"Dataset": "BBC", "Metric": "Passive no agent rate", "Value": bbc_avg_passive_no_agent_rate},
    {"Dataset": "TNH", "Metric": "Passive rate", "Value": tnh_passive_rate},
    {"Dataset": "TNH", "Metric": "Passive no agent rate", "Value": tnh_avg_passive_no_agent_rate},]
    df = pd.DataFrame(data_passive)
    fig = px.histogram(df, x="Metric", y="Value",
                color='Dataset', barmode='group',
                height=400, title ="Passive Rate Comparison between BBC and TNH")
    fig.show()

    # plot top agents, patients, attributes types between BBC and TNH improved
    bbc_agent_types = bbc_top_agents[1]
    tnh_agent_types = tnh_top_agents[1]
    bbc_patient_types = bbc_top_patients[1]
    tnh_patient_types = tnh_top_patients[1]
    data_agent_type = prepare_agent_type_data(bbc_agent_types, tnh_agent_types)
    data_patient_type = prepare_agent_type_data(bbc_patient_types, tnh_patient_types)
    df_agent = pd.DataFrame(data_agent_type)
    df_patient = pd.DataFrame(data_patient_type)
    fig_agent = px.bar(df_agent, x="Entity_Type", y="Percentage",
                color='Dataset', barmode='group',
                height=400, title ="Agent Types Distribution between BBC and TNH")
    fig_agent.show()
    fig_patient = px.bar(df_patient, x="Entity_Type", y="Percentage",
                color='Dataset', barmode='group',
                height=400, title ="Patient Types Distribution between BBC and TNH")
    fig_patient.show()

    # compare top agents, patients, attributes between BBC and TNH improved
    bbc_top_agents_dict = dict(bbc_top_agents[0])
    tnh_top_agents_dict = dict(tnh_top_agents[0])
    bbc_top_patients_dict = dict(bbc_top_patients[0])
    tnh_top_patients_dict = dict(tnh_top_patients[0])
    bbc_top_attributes_dict = dict(bbc_top_attributes[0])
    tnh_top_attributes_dict = dict(tnh_top_attributes[0])
    plot_top_actors(bbc_top_agents_dict, tnh_top_agents_dict, "Agent")
    plot_top_actors(bbc_top_patients_dict, tnh_top_patients_dict, "Patient")
    plot_top_actors(bbc_top_attributes_dict, tnh_top_attributes_dict, "Attribute")

if __name__ == "__main__":
    main()
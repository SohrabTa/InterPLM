import sys
import hashlib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import torch

from interplm.sae.inference import load_sae
from interplm.embedders.prott5 import ProtT5CrosscoderEmbedder
from interplm.dashboard.feature_activation_vis import visualize_protein_feature
from interplm.utils import get_device
import requests
from interplm.dashboard.view_structures import view_single_protein
from interplm.dashboard.colors import get_structure_palette_and_colormap


def get_uniprot_by_sequence(sequence: str):
    # Use MD5 checksum for UniProt lookup as it is the most reliable way to find exact sequence matches
    checksum = hashlib.md5(sequence.encode("utf-8")).hexdigest()
    url = "https://rest.uniprot.org/uniprotkb/search"
    
    # Query all potential matches for this checksum in one go
    params = {"query": f"checksum:{checksum}", "format": "json", "size": 25}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None, f"API Error: {response.status_code} - {response.text}"
            
        candidates = response.json().get("results", [])
        exact_matches = []
        
        # 1. Filter for exact sequence matches
        for cand in candidates:
            if cand.get("sequence", {}).get("value") == sequence:
                exact_matches.append(cand)
        
        if not exact_matches:
            return None, "No exact sequence match found in UniProt."
            
        # 2. Local prioritization logic
        # We prefer Reviewed (Swiss-Prot) entries and entries with AlphaFold structures
        def score_match(res):
            is_rev = res.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"
            has_af = any(ref.get("database") == "AlphaFoldDB" for ref in res.get("uniProtKBCrossReferences", []))
            return (is_rev, has_af)

        exact_matches.sort(key=score_match, reverse=True)
        best_match = exact_matches[0]
        
        entry_id = best_match.get("primaryAccession")
        try:
            p_desc = best_match.get("proteinDescription", {})
            rec_name = p_desc.get("recommendedName", {})
            full_name = rec_name.get("fullName", {}).get("value", "Unknown Protein Name")
            
            desc_parts = [full_name]
            
            # Add gene names
            genes = best_match.get("genes", [])
            gene_names = [g.get("geneName", {}).get("value") for g in genes if g.get("geneName")]
            if gene_names:
                desc_parts.append(f"({', '.join(gene_names)})")
                
            # Add alternative names
            alt_names = p_desc.get("alternativeNames", [])
            for alt in alt_names:
                if "fullName" in alt:
                    desc_parts.append(f"({alt['fullName']['value']})")
            
            # Add EC numbers
            ec_numbers = rec_name.get("ecNumbers", [])
            for ec in ec_numbers:
                desc_parts.append(f"(EC {ec['value']})")
                
            name = " ".join(desc_parts)
        except Exception:
            name = "Unknown Protein Name"
        return entry_id, name
        
    except Exception as e:
        return None, f"Exception: {str(e)}"


# Constants
CROSSCODER_PATH = "/Users/sohrab.tawana/private/crosscoder/model_checkpoints/crosscoder_l8192_k32_bs512_full_2026-03-12_06-03-41/crashed_epoch_0_step_2519836"
CONCEPT_CSV_PATH = "/Users/sohrab.tawana/private/crosscoder/data/crosscoder_eval/uniprotkb_modern_score45_67k/test_counts/heldout_all_top_pairings.csv"


@st.cache_resource
def load_models():
    device = get_device()
    embedder = ProtT5CrosscoderEmbedder(device=device)

    # Load SAE and wrap using the inference utils
    sae = load_sae(CROSSCODER_PATH, model_name="ae_normalized.pt", device=device)

    return embedder, sae


@st.cache_data
def load_concept_data():
    df = pd.read_csv(CONCEPT_CSV_PATH)
    return df.copy()


def run_inference_page():
    # Load custom CSS if available (optional)
    css_paths = [
        Path(__file__).parent / ".streamlit" / "style.css",
        Path(".streamlit/style.css"),
    ]
    for css_path in css_paths:
        if css_path.exists():
            with open(css_path) as f:
                css_content = f.read()
                st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            break

    st.title("InterPLM ProtT5 Crosscoder Inference")
    st.markdown("""
        Paste protein sequences below. They will be passed through ProtT5, and the 24-layer representations 
        will be mapped by the sparse crosscoder into human-interpretable Swiss-Prot concepts.
    """)

    # Input area
    seq_input = st.text_area(
        "Protein Sequences (one per line)",
        placeholder="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQ...",
        height=150,
    )

    if st.button("Analyze Sequences"):
        if not seq_input.strip():
            st.warning("Please enter at least one sequence.")
            return

        sequences = [s.strip().upper() for s in seq_input.split("\n") if s.strip()]

        # Enforce max length of 512 AAs
        for seq in sequences:
            if len(seq) > 512:
                st.error(
                    f"Sequence too long ({len(seq)} AAs). Maximum allowed is 512 AAs."
                )
                return

        with st.spinner("Loading models..."):
            embedder, sae = load_models()
            concept_df = load_concept_data()

        with st.spinner("Running ProtT5 and Crosscoder..."):
            # 1. Get embeddings
            # Returns [total_tokens, M=1, P=24, D=1024]
            embeddings = embedder.extract_embeddings(sequences, batch_size=4)

            # 2. Get SAE features
            features = sae.encode(embeddings, normalize_features=True)  # shape: [total_tokens, n_latents]

            # Move to CPU for analysis
            features = features.detach().cpu().numpy()

            # Re-split features into per-sequence lists
            seq_features = []
            curr_idx = 0
            for seq in sequences:
                seq_len = len(seq)
                seq_features.append(features[curr_idx : curr_idx + seq_len])
                curr_idx += seq_len

        # Save to session state so it persists across widget interactions
        st.session_state.analyzed_sequences = sequences
        st.session_state.seq_features = seq_features
        st.session_state.concept_df = concept_df

    # Out of the button scope, render if we have analyzed sequences
    if "analyzed_sequences" in st.session_state:
        sequences = st.session_state.analyzed_sequences
        seq_features = st.session_state.seq_features
        concept_df = st.session_state.concept_df

        # 3. Analyze and Visualize
        st.success(f"Successfully processed {len(sequences)} sequence(s).")

        for i, seq in enumerate(sequences):
            st.markdown("---")
            st.subheader(f"Sequence {i + 1} ({len(seq)} AA)")

            # Find active features for this sequence
            # seq_feats shape: [seq_len, n_latents]
            seq_feats = seq_features[i]
            max_activations = seq_feats.max(axis=0)  # shape [n_latents]

            active_latents = np.where(max_activations > 0)[0]

            # Find which active latents correspond to concepts
            matching_concepts = []

            for latent in active_latents:
                latent_concepts = concept_df[concept_df["feature"] == latent]

                # Check if this latent activates above the concept threshold anywhere in the sequence
                for _, row in latent_concepts.iterrows():
                    threshold = row["threshold_pct"]
                    if max_activations[latent] >= threshold:
                        matching_concepts.append(
                            {
                                "Feature": f"f/{latent}",
                                "Concept": row["concept"],
                                "F1 Score": row["f1_per_domain"],
                                "Max Activation": max_activations[latent],
                                "Threshold": threshold,
                                "Latent_ID": latent,
                            }
                        )

            if not matching_concepts:
                st.info("No interpretable Swiss-Prot concepts found for this sequence.")
                continue

            match_df = pd.DataFrame(matching_concepts)
            # Deduplicate by concept, keeping highest F1
            match_df = match_df.sort_values(
                by="F1 Score", ascending=False
            ).drop_duplicates(subset="Concept")

            st.write("### Active Biological Concepts")
            st.dataframe(
                match_df[["Feature", "Concept", "F1 Score"]],
                hide_index=True,
                column_config={
                    "F1 Score": st.column_config.NumberColumn(format="%.4f")
                },
            )

            # Add visualizations dropdown
            st.write("### Sequence Visualizations")

            # Format display names for the dropdown
            match_df["Display_Name"] = (
                match_df["Concept"] + " (f/" + match_df["Latent_ID"].astype(str) + ")"
            )

            # Key with unique index for the selectbox in case of multiple sequences
            selected_display_name = st.selectbox(
                "Choose a concept to view its activation across the sequence:",
                options=match_df["Display_Name"].tolist(),
                key=f"select_seq_{i}",
            )

            if selected_display_name:
                row = match_df[match_df["Display_Name"] == selected_display_name].iloc[
                    0
                ]
                latent_id = row["Latent_ID"]
                concept_name = row["Concept"]

                # Extract activation across sequence for this latent
                activation_sequence = seq_feats[:, latent_id]

                st.write(f"**{concept_name}** ({row['Feature']})")

                # We construct a dummy metadata mapping
                dummy_metadata = pd.Series(
                    {"sequence": seq, "protein_id": f"Seq {i + 1}"}
                )

                col1, col2, col3 = st.columns([5, 4, 1])

                with col1:
                    fig = visualize_protein_feature(
                        activation_sequence,
                        seq,
                        dummy_metadata,
                        characteristic_to_plot="Amino Acids",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    with st.spinner("Looking up AlphaFold structure..."):
                        uniprot_id, error_or_name = get_uniprot_by_sequence(seq)

                        if uniprot_id:
                            protein_name = error_or_name
                            st.write(f"**{uniprot_id}**: {protein_name}")
                            try:
                                min_val = 0.0
                                active_vals = activation_sequence[
                                    activation_sequence > 0
                                ]
                                if len(active_vals) > 0:
                                    median_val = float(np.median(active_vals))
                                    top_val = float(np.max(active_vals))
                                else:
                                    median_val = 0.5
                                    top_val = 1.0

                                if median_val <= min_val:
                                    median_val = min_val + 0.1
                                if top_val <= median_val:
                                    top_val = median_val + 0.1

                                color_range = (min_val, median_val, top_val)
                                colormap_fn, palette_fig = (
                                    get_structure_palette_and_colormap(color_range)
                                )

                                structure_html = view_single_protein(
                                    uniprot_id=uniprot_id,
                                    values_to_color=activation_sequence.tolist(),
                                    colormap_fn=colormap_fn,
                                    pymol_params={"width": "100%", "height": 300},
                                )
                                import streamlit.components.v1 as components

                                components.html(structure_html, height=300)

                                with col3:
                                    st.plotly_chart(
                                        palette_fig, use_container_width=True
                                    )

                            except Exception as e:
                                st.info(
                                    f"AlphaFold structure not available for this sequence. ({str(e)})"
                                )
                        else:
                            st.info(
                                f"No matching UniProt entry found for this sequence. ({error_or_name})"
                            )


def main():
    st.set_page_config(layout="wide", page_title="Crosscoder Inference", page_icon="🧬")
    run_inference_page()


if __name__ == "__main__":
    main()

help_note_lists = {
    "metrics": [
        "This section visualizes how each feature behaves across protein sequences in several ways:\n"
        "1. Activation Distribution - Shows how frequently and strongly the feature activates\n"
        "2. Structural vs Sequential Activation - Reveals whether the feature responds more to protein structure or sequence order\n"
        "3. UMAP Visualization - Provides a 2D map of how features relate to each other\n"
        "4. Swiss-Prot Concepts - Lists biological concepts associated with the feature\n\n"
        "These different views can help identify interesting features, such as:\n"
        "• Features that activate across many proteins (suggesting common biological elements)\n"
        "• Features that activate over large sequence regions (potentially indicating domains)\n"
        "• Features with distinct structural activation patterns\n"
        "• Features strongly associated with specific biological concepts from Swiss-Prot"
    ],
    "feature_details": [
        "To select a different feature or visualize feature activation patterns ",
        "on proteins, use the sidebar controls.",
    ],
    "autointerp": [
        "We provide [Claude](https://claude.ai) with information about proteins ",
        "that activate the feature at different levels and ask it to summarize the ",
        "feature and explain what activates it at different levels. We then use this summary ",
        "(shown below) and the longer description  to predict feature activation levels on new proteins. ",
        "The score here is the correlation between the predicted and actual activation ",
        "levels (higher is better, 1.0 is the max). See the paper for more details.",
    ],
    "act_distribution": [
        "The activation distribution shows how how often the feature activates at different levels across a random sample of amino acids"
    ],
    "swissprot_per_feat": [
        "Swiss-Prot is a database of protein sequences and annotations. ",
        "We use it to identify biological concepts that are associated with ",
        "features. The table below lists the concepts identified for this feature, ",
        "as measured by F1 score. The F1 score is a measure of how well the feature ",
        "activates on the concept, with higher scores indicating better performance. ",
        "While the table above only shows the top feature per concept (above F1 > 0.5)",
        "this table shows any concept with F1 > 0.2 for this specific feature.",
    ],
    "overall": [
        "This dashboard helps explore features extracted from the protein language model Rostlab/prot_t5_xl_uniref50 via Sparse Crosscoder. ",
        "Details on training, analysis, and visualizations in the [InterPLM paper](https://www.nature.com/articles/s41592-025-02836-7).",
    ],
    "select_esm_layer": [
        "Each ProtT5 layer has 8,192 features extracted by the crosscoder. Select any ",
        "one of the features (or get a random feature) to start exploring! When selecting ",
        "a random feature, we'll show you one that either has a Swiss-Prot concept or an LLM description.",
    ],
    "vis_sidebar": [
        "Use one of the following methods to select proteins for visualization. Feature activation will be shown on the sequence and, when available, predicted structure from [AFDB](https://alphafold.ebi.ac.uk/)."
    ],
    "uniprot": [
        "If you look select a protein in [UniProt](https://www.uniprot.org/uniprotkb?facets=reviewed%3Atrue&query=*), this is the 'Entry' field."
    ],
}

help_notes = {k: "".join(v) for k, v in help_note_lists.items()}

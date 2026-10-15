"""Shared organ classes for HESTA (human, `celltype`) and MOSTA (mouse, `annotation`) labels.

The two atlases use different vocabularies. HARMONIZE maps a lower-cased label of either one to a
coarse organ class. Blood and cavity are EXCLUDED; labels without a clear counterpart map to "Other".
Copied from demos/human_mouse_align_merged.ipynb; add new labels here.
"""
import numpy as np
import pandas as pd

EXCLUDED = "Excluded"
HARMONIZE = {
    # nervous system
    "brain": "Brain", "meninges": "Brain", "choroid plexus": "Brain", "pituitary": "Brain", "pituitary gland": "Brain",
    "spinal cord": "Spinal cord",
    "notochord": "Notochord",
    "ganglion": "Ganglion", "dorsal root ganglion": "Ganglion", "sympathetic nerve": "Ganglion",
    "facial nerve": "Ganglion", "neural crest": "Ganglion",
    "eye": "Eye", "lens": "Eye", "retina": "Eye", "optic nerve": "Eye",
    "inner ear": "Ear", "otic vesicle": "Ear",
    # organs
    "heart": "Heart",
    "liver": "Liver", "hepatic diverticulum": "Liver", "septum transversum": "Liver",
    "gut": "Gut", "gi tract": "Gut", "primitive gut": "Gut", "primitive gut tube": "Gut", "stomach": "Gut",
    "esophagus": "Gut", "pancreas": "Gut", "pancreas bud": "Gut", "pancreas primordium": "Gut",
    "lung": "Lung", "lung primordium": "Lung", "trachea": "Lung",
    "mesonephron": "Urogenital", "mesonephros": "Urogenital", "pronephron": "Urogenital", "kidney": "Urogenital",
    "metanephros": "Urogenital", "urogenital ridge": "Urogenital", "gonad": "Urogenital", "testis": "Urogenital",
    "ovary": "Urogenital", "adrenal gland": "Urogenital", "bladder": "Urogenital", "urethra": "Urogenital",
    "cloaca": "Urogenital",
    # musculoskeletal
    "somite": "Muscle", "myotome": "Muscle", "dermomyotome": "Muscle", "skeletal muscle": "Muscle", "muscle": "Muscle",
    "sclerotome": "Skeleton", "cartilage": "Skeleton", "cartilage primordium": "Skeleton", "bone": "Skeleton",
    "cranial skeleton": "Skeleton",
    "limb bud": "Limb", "limb": "Limb", "limb ectoderm": "Limb",
    # head and surface
    "branchial arch": "Craniofacial", "pharyngeal": "Craniofacial", "jaw and tooth": "Craniofacial",
    "tongue": "Craniofacial", "submandibular gland": "Craniofacial", "olfactory epithelium": "Craniofacial",
    "thyroid": "Craniofacial",
    "epidermis": "Epidermis", "surface ectoderm": "Epidermis",
    "mesenchyme": "Mesenchyme", "head mesenchyme": "Mesenchyme", "connective tissue": "Mesenchyme",
    "mesentery": "Mesenchyme", "mesothelium": "Mesenchyme", "adipose tissue": "Mesenchyme",
    # no clear counterpart
    "smooth muscle": "Other", "mucosal epithelium": "Other", "thymus": "Other", "spleen": "Other",
    # not scored
    "cavity": EXCLUDED, "extraembryonic tissue": EXCLUDED, "erythroblasts": EXCLUDED,
    "blood vessel": EXCLUDED, "endothelium": EXCLUDED, "agm": EXCLUDED,
}
NOT_SCORED = {"Other", EXCLUDED}

# Stage correspondence quality, HESTA (Carnegie stage) x MOSTA (embryonic day) (notes, Table 1)
STAGE_QUALITY = {
    ("CS12-13", "E9.5"): "good", ("CS14-15", "E10.5"): "good",
    ("CS17", "E11.5"): "ambiguous", ("CS17", "E12.5"): "ambiguous",
    ("CS18", "E12.5"): "good",
    ("CS19", "E12.5"): "ambiguous", ("CS19", "E13.5"): "ambiguous",
    ("CS20", "E13.5"): "good", ("CS23", "E14.5"): "weak",
}


def harmonize(labels):
    """Organ class per label ("Other" if unmapped) and the sorted lower-cased labels missing from HARMONIZE."""
    low = pd.Series(np.asarray(labels, dtype=str)).str.lower()
    unknown = sorted(set(low) - set(HARMONIZE))
    return low.map(HARMONIZE).fillna("Other").to_numpy(dtype=str), unknown

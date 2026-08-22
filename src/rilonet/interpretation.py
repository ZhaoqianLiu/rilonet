from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import bh_fdr, ensure_dir, write_json


DEFAULT_KEGG_BASE = "https://rest.kegg.jp"
DEFAULT_HMDB_BASE = "https://hmdb.ca/metabolites"
DEFAULT_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class CachedHttpClient:
    def __init__(
        self,
        cache_dir: str | Path,
        offline: bool = False,
        requests_per_second: float = 2.5,
    ) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.offline = bool(offline)
        self.minimum_interval = 1.0 / max(float(requests_per_second), 0.1)
        self.last_request = 0.0

    def get_text(self, url: str, suffix: str = ".txt") -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        target = self.cache_dir / f"{digest}{suffix}"
        if target.exists():
            return target.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"Offline cache miss for {url}")
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.minimum_interval:
            time.sleep(self.minimum_interval - elapsed)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "RILoNet/0.1 academic feature annotation",
                "Accept": "application/xml,text/plain,*/*",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
        self.last_request = time.monotonic()
        text = raw.decode("utf-8", errors="replace")
        target.write_text(text, encoding="utf-8")
        target.with_suffix(target.suffix + ".url").write_text(url, encoding="utf-8")
        return text


def _parse_kegg_pairs(text: str) -> list[tuple[str, str]]:
    rows = []
    for line in text.splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) >= 2:
            rows.append((fields[0], fields[1]))
    return rows


def _strip_kegg_prefix(value: str) -> str:
    return value.split(":", 1)[-1]


def annotate_ec_features(
    features: Iterable[str],
    client: CachedHttpClient,
    base_url: str = DEFAULT_KEGG_BASE,
) -> pd.DataFrame:
    feature_list = sorted({str(value) for value in features})
    enzyme_names = {
        _strip_kegg_prefix(identifier): name
        for identifier, name in _parse_kegg_pairs(
            client.get_text(f"{base_url}/list/enzyme")
        )
    }
    pathway_names = {
        _strip_kegg_prefix(identifier): name
        for identifier, name in _parse_kegg_pairs(
            client.get_text(f"{base_url}/list/pathway")
        )
    }
    module_names = {
        _strip_kegg_prefix(identifier): name
        for identifier, name in _parse_kegg_pairs(
            client.get_text(f"{base_url}/list/module")
        )
    }
    pathways: dict[str, list[str]] = defaultdict(list)
    for enzyme, pathway in _parse_kegg_pairs(
        client.get_text(f"{base_url}/link/pathway/enzyme")
    ):
        pathways[_strip_kegg_prefix(enzyme)].append(_strip_kegg_prefix(pathway))
    modules: dict[str, list[str]] = defaultdict(list)
    for enzyme, module in _parse_kegg_pairs(
        client.get_text(f"{base_url}/link/module/enzyme")
    ):
        modules[_strip_kegg_prefix(enzyme)].append(_strip_kegg_prefix(module))
    rows = []
    for feature in feature_list:
        ec = feature.removeprefix("ec:")
        pathway_ids = sorted(set(pathways.get(ec, [])))
        module_ids = sorted(set(modules.get(ec, [])))
        rows.append(
            {
                "omic": "MT",
                "feature": feature,
                "ec_number": ec,
                "enzyme_name": enzyme_names.get(ec, ""),
                "pathway_ids": "|".join(pathway_ids),
                "pathway_names": "|".join(
                    pathway_names.get(value, value) for value in pathway_ids
                ),
                "module_ids": "|".join(module_ids),
                "module_names": "|".join(
                    module_names.get(value, value) for value in module_ids
                ),
                "mapping_status": "mapped" if ec in enzyme_names else "unmapped",
            }
        )
    return pd.DataFrame(rows)


def normalize_hmdb_id(value: str) -> str:
    match = re.search(r"HMDB(\d+)", str(value), flags=re.IGNORECASE)
    if not match:
        return ""
    return f"HMDB{match.group(1).zfill(7)}"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _xml_descendant(element: ET.Element, path: list[str]) -> ET.Element | None:
    current = element
    for name in path:
        match = next(
            (child for child in current if _local_name(child.tag) == name), None
        )
        if match is None:
            return None
        current = match
    return current


def _parse_hmdb_metabolite(element: ET.Element) -> dict[str, str]:
    taxonomy = _xml_descendant(element, ["taxonomy"])
    pathway_parent = _xml_descendant(element, ["pathways"])
    pathway_names = []
    if pathway_parent is not None:
        for pathway in pathway_parent:
            name = _xml_child_text(pathway, "name")
            if name:
                pathway_names.append(name)
    result = {
        "primary_hmdb_id": _xml_child_text(element, "accession"),
        "common_name": _xml_child_text(element, "name"),
        "chemical_formula": _xml_child_text(element, "chemical_formula"),
        "kingdom": "",
        "superclass": "",
        "class": "",
        "subclass": "",
        "direct_parent": "",
        "kegg_compound_id": _xml_child_text(element, "kegg_id"),
        "chebi_id": _xml_child_text(element, "chebi_id"),
        "pubchem_compound_id": _xml_child_text(element, "pubchem_compound_id"),
        "pathway_names": "|".join(sorted(set(pathway_names))),
    }
    if taxonomy is not None:
        for key, xml_name in (
            ("kingdom", "kingdom"),
            ("superclass", "super_class"),
            ("class", "class"),
            ("subclass", "sub_class"),
            ("direct_parent", "direct_parent"),
        ):
            result[key] = _xml_child_text(taxonomy, xml_name)
    return result


def _normalized_metabolite_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _load_hmdb_bulk_xml(
    path: str | Path,
    desired_ids: set[str],
    desired_names: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    desired_names = desired_names or set()
    for _, element in ET.iterparse(path, events=("end",)):
        if _local_name(element.tag) != "metabolite":
            continue
        parsed = _parse_hmdb_metabolite(element)
        accession = parsed["primary_hmdb_id"]
        secondary = _xml_descendant(element, ["secondary_accessions"])
        identifiers = {accession}
        if secondary is not None:
            identifiers.update((child.text or "").strip() for child in secondary)
        for identifier in desired_ids & identifiers:
            found[identifier] = parsed
        names = {_normalized_metabolite_name(parsed["common_name"])}
        synonyms = _xml_descendant(element, ["synonyms"])
        if synonyms is not None:
            names.update(
                _normalized_metabolite_name(child.text or "") for child in synonyms
            )
        for name in desired_names & names:
            found[f"name::{name}"] = parsed
        element.clear()
    return found


def annotate_hmdb_features(
    features: Iterable[str],
    client: CachedHttpClient,
    bulk_xml: str | Path | None = None,
    base_url: str = DEFAULT_HMDB_BASE,
    online_lookup: bool = False,
) -> pd.DataFrame:
    feature_list = sorted({str(value) for value in features})
    desired = {normalize_hmdb_id(value) for value in feature_list}
    desired.discard("")
    desired_names = {
        _normalized_metabolite_name(
            value.split("_", 1)[1] if "_" in value else value
        )
        for value in feature_list
    }
    bulk_records = (
        _load_hmdb_bulk_xml(bulk_xml, desired, desired_names)
        if bulk_xml is not None
        else {}
    )
    rows = []
    for feature in feature_list:
        input_id_match = re.search(r"HMDB\d+", feature, flags=re.IGNORECASE)
        input_id = input_id_match.group(0).upper() if input_id_match else ""
        normalized = normalize_hmdb_id(input_id)
        parsed = bulk_records.get(normalized)
        mapped_by_name = False
        fallback_name = feature.split("_", 1)[1] if "_" in feature else feature
        if parsed is None:
            parsed = bulk_records.get(
                f"name::{_normalized_metabolite_name(fallback_name)}"
            )
            mapped_by_name = parsed is not None
        error = ""
        if parsed is None and normalized and online_lookup:
            try:
                xml = client.get_text(f"{base_url}/{normalized}.xml", suffix=".xml")
                parsed = _parse_hmdb_metabolite(ET.fromstring(xml))
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
        values = parsed or {
            "primary_hmdb_id": normalized,
            "common_name": fallback_name,
            "chemical_formula": "",
            "kingdom": "",
            "superclass": "",
            "class": "",
            "subclass": "",
            "direct_parent": "",
            "kegg_compound_id": "",
            "chebi_id": "",
            "pubchem_compound_id": "",
            "pathway_names": "",
        }
        rows.append(
            {
                "omic": "MB",
                "feature": feature,
                "input_hmdb_id": input_id,
                **values,
                "mapping_status": (
                    "mapped_by_name"
                    if mapped_by_name
                    else "mapped"
                    if parsed is not None and values.get("primary_hmdb_id")
                    else "normalized_id_only"
                    if normalized
                    else "unmapped"
                ),
                "mapping_error": error,
            }
        )
    return pd.DataFrame(rows)


def canonical_taxon_query(feature: str) -> tuple[str, str]:
    tokens = str(feature).replace("_", " ").split()
    if len(tokens) < 2:
        return " ".join(tokens), "unparsed"
    query = f"{tokens[0]} {tokens[1]}"
    match_type = "exact_species_name" if len(tokens) == 2 else "species_level_fallback"
    return query, match_type


def _parse_ncbi_taxonomy_xml(text: str) -> dict[str, dict[str, str]]:
    root = ET.fromstring(text)
    rows: dict[str, dict[str, str]] = {}
    for taxon in root.iter("Taxon"):
        taxid_element = taxon.find("TaxId")
        scientific_element = taxon.find("ScientificName")
        if taxid_element is None or scientific_element is None:
            continue
        taxid = (taxid_element.text or "").strip()
        values = {
            "taxid": taxid,
            "matched_name": (scientific_element.text or "").strip(),
            "rank": (taxon.findtext("Rank") or "").strip(),
            "species": "",
            "genus": "",
            "family": "",
            "order": "",
            "class": "",
            "phylum": "",
        }
        lineage = taxon.find("LineageEx")
        if lineage is not None:
            for ancestor in lineage.findall("Taxon"):
                rank = (ancestor.findtext("Rank") or "").strip().lower()
                name = (ancestor.findtext("ScientificName") or "").strip()
                if rank in values:
                    values[rank] = name
        own_rank = values["rank"].lower()
        if own_rank in values:
            values[own_rank] = values["matched_name"]
        rows[taxid] = values
    return rows


def annotate_taxonomy_features(
    features: Iterable[str],
    client: CachedHttpClient,
    email: str | None = None,
    base_url: str = DEFAULT_NCBI_BASE,
) -> pd.DataFrame:
    feature_list = sorted({str(value) for value in features})
    query_records = []
    taxids = []
    for feature in feature_list:
        query, match_type = canonical_taxon_query(feature)
        identifiers = []
        for field in ("Scientific Name", "All Names", None):
            if field != "Scientific Name" and any(
                token.lower() in {"bacterium", "sp.", "sp"} for token in query.split()
            ):
                continue
            term = f'"{query}"[{field}]' if field is not None else query
            parameters = {
                "db": "taxonomy",
                "term": term,
                "retmode": "json",
                "retmax": "5",
                "tool": "rilonet",
            }
            if email:
                parameters["email"] = email
            url = f"{base_url}/esearch.fcgi?{urllib.parse.urlencode(parameters)}"
            result = json.loads(client.get_text(url, suffix=".json"))
            identifiers = result.get("esearchresult", {}).get("idlist", [])
            if identifiers:
                if field == "All Names":
                    match_type = (
                        "taxonomy_synonym"
                        if match_type == "exact_species_name"
                        else "species_fallback_via_synonym"
                    )
                elif field is None:
                    match_type = (
                        "taxonomy_search_fallback"
                        if match_type == "exact_species_name"
                        else "species_search_fallback"
                    )
                break
        taxid = str(identifiers[0]) if identifiers else ""
        if taxid:
            taxids.append(taxid)
        query_records.append((feature, query, match_type, taxid))
    lineage: dict[str, dict[str, str]] = {}
    for start in range(0, len(set(taxids)), 100):
        batch = sorted(set(taxids))[start : start + 100]
        parameters = {
            "db": "taxonomy",
            "id": ",".join(batch),
            "retmode": "xml",
            "tool": "rilonet",
        }
        if email:
            parameters["email"] = email
        url = f"{base_url}/efetch.fcgi?{urllib.parse.urlencode(parameters)}"
        lineage.update(_parse_ncbi_taxonomy_xml(client.get_text(url, suffix=".xml")))
    rows = []
    for feature, query, match_type, taxid in query_records:
        values = lineage.get(
            taxid,
            {
                "taxid": "",
                "matched_name": "",
                "rank": "",
                "species": "",
                "genus": "",
                "family": "",
                "order": "",
                "class": "",
                "phylum": "",
            },
        )
        rows.append(
            {
                "omic": "MG",
                "feature": feature,
                "query_name": query,
                **values,
                "match_type": match_type if taxid else "unmapped",
                "mapping_confidence": (
                    "high"
                    if taxid and match_type == "exact_species_name"
                    else "moderate"
                    if taxid
                    else "none"
                ),
                "mapping_status": "mapped" if taxid else "unmapped",
            }
        )
    return pd.DataFrame(rows)


def _select_group_row(
    subset: pd.DataFrame, group: str, edge_id: str
) -> pd.Series:
    selected = subset[subset["group"].astype(str).eq(group)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {group!r} row for edge {edge_id!r}; found {len(selected)}"
        )
    return selected.iloc[0]


def _boolean_value(value: Any, default: bool = True) -> bool:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def build_edge_master(
    network_dir: str | Path,
    reference_group: str = "nonIBD",
    condition_group: str = "IBD",
    q_column: str = "contrast_q",
) -> pd.DataFrame:
    rows = []
    paths = sorted(Path(network_dir).glob("*_edges.tsv"))
    if not paths:
        raise FileNotFoundError(f"No *_edges.tsv files found in {network_dir}")
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        required = {
            "edge_id",
            "group",
            "effect",
            "contrast_vs_reference",
            q_column,
        }
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for edge_id, subset in frame.groupby("edge_id", sort=False):
            reference = _select_group_row(subset, reference_group, str(edge_id))
            condition = _select_group_row(subset, condition_group, str(edge_id))
            row = {
                column: condition.get(column, reference.get(column, np.nan))
                for column in (
                    "edge_id",
                    "analysis_type",
                    "block",
                    "window_name",
                    "baseline_mode",
                    "near_association_method",
                    "effect_scale",
                    "x_omic",
                    "y_omic",
                    "x_feature",
                    "y_feature",
                    "screen_correlation",
                    "n_pairs_model",
                    "n_subjects_model",
                    "fit_method",
                    "fit_converged",
                )
            }
            reference_effect = float(reference["effect"])
            condition_effect = float(condition["effect"])
            contrast = float(condition["contrast_vs_reference"])
            analysis_type = str(row.get("analysis_type", ""))
            sign_consistent = _boolean_value(
                condition.get("sign_consistent", True),
                default=True,
            )
            primary_network_eligible = _boolean_value(
                condition.get("primary_network_eligible", np.nan),
                default=(analysis_type != "near" or sign_consistent),
            )
            row.update(
                {
                    "source_file": path.name,
                    "reference_group": reference_group,
                    "condition_group": condition_group,
                    "effect_reference": reference_effect,
                    "effect_condition": condition_effect,
                    "effect_p_reference": float(reference.get("effect_p", np.nan)),
                    "effect_p_condition": float(condition.get("effect_p", np.nan)),
                    "effect_q_reference": float(reference.get("effect_q", np.nan)),
                    "effect_q_condition": float(condition.get("effect_q", np.nan)),
                    "contrast": contrast,
                    "contrast_p": float(condition.get("contrast_p", np.nan)),
                    "contrast_q": float(condition[q_column]),
                    "contrast_q_source": q_column,
                    "contrast_q_block": float(
                        condition.get("contrast_q_block", np.nan)
                    ),
                    "contrast_q_global": float(
                        condition.get("contrast_q_global", np.nan)
                    ),
                    "sign_consistent": sign_consistent,
                    "primary_network_eligible": primary_network_eligible,
                    "attenuation_score": (
                        abs(reference_effect) - abs(condition_effect)
                    ),
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    if result["edge_id"].duplicated().any():
        duplicates = result.loc[result["edge_id"].duplicated(), "edge_id"].head()
        raise ValueError(f"Duplicate edge IDs across network files: {duplicates.tolist()}")
    return result


def classify_edges(
    edges: pd.DataFrame, near_zero_threshold: float = 0.10
) -> pd.DataFrame:
    result = edges.copy()
    labels = []
    for row in result.itertuples(index=False):
        reference = float(row.effect_reference)
        condition = float(row.effect_condition)
        if abs(reference) >= near_zero_threshold > abs(condition):
            label = "weakened_or_lost"
        elif abs(reference) < near_zero_threshold <= abs(condition):
            label = "condition_emergent"
        elif reference * condition < 0:
            label = "sign_reversed"
        elif abs(condition) < abs(reference):
            label = "weakened_or_lost"
        elif abs(condition) > abs(reference):
            label = "strengthened"
        else:
            label = "stable"
        labels.append(label)
    result["edge_class"] = labels
    result["contrast_opposes_reference"] = (
        result["contrast"] * result["effect_reference"] < 0
    )
    return result


def _node_id(omic: str, feature: str) -> str:
    return f"{omic}::{feature}"


def discover_modules(
    discovery_edges: pd.DataFrame,
    resolution: float = 1.0,
    seed: int = 20260721,
    leiden_runs: int = 10,
    consensus_threshold: float = 0.55,
    min_module_size: int = 3,
    min_module_stability: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as exc:
        raise RuntimeError(
            "Consensus module discovery requires igraph>=0.11 and "
            "leidenalg>=0.10. Install RILoNet with the interpretation extra: "
            "pip install -e '.[interpretation]'."
        ) from exc
    if leiden_runs < 1:
        raise ValueError("leiden_runs must be at least 1")
    if not 0 < consensus_threshold <= 1:
        raise ValueError("consensus_threshold must be in (0, 1]")
    if min_module_size < 2:
        raise ValueError("min_module_size must be at least 2")
    if not 0 <= min_module_stability <= 1:
        raise ValueError("min_module_stability must be in [0, 1]")

    membership_columns = [
        "analysis_type",
        "module_id",
        "node_id",
        "omic",
        "feature",
        "module_size",
        "n_omics",
        "node_stability",
        "module_stability",
    ]
    assignment_columns = [
        "edge_id",
        "analysis_type",
        "source_module",
        "target_module",
        "module_id",
    ]
    diagnostic_columns = [
        "analysis_type",
        "stage",
        "run",
        "seed",
        "resolution",
        "modularity",
        "n_modules",
        "n_nodes",
        "n_edges",
        "retained_modules",
        "assigned_nodes",
    ]
    memberships = []
    edge_assignments = []
    diagnostics = []
    for layer, subset in discovery_edges.groupby("analysis_type", sort=True):
        edge_weights: dict[tuple[str, str], float] = defaultdict(float)
        for row in subset.itertuples(index=False):
            source = _node_id(str(row.x_omic), str(row.x_feature))
            target = _node_id(str(row.y_omic), str(row.y_feature))
            if source == target:
                continue
            weight = abs(float(row.contrast))
            key = tuple(sorted((source, target)))
            edge_weights[key] += weight
        nodes = sorted(
            {node for edge in edge_weights for node in edge}
        )
        node_index = {node: index for index, node in enumerate(nodes)}
        edges = [
            (node_index[source], node_index[target])
            for source, target in edge_weights
        ]
        weights = [float(edge_weights[edge]) for edge in edge_weights]
        graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        graph.vs["name"] = nodes
        graph.es["weight"] = weights

        resolutions = sorted(
            {
                round(max(float(resolution) * multiplier, 1e-6), 12)
                for multiplier in (0.75, 1.0, 1.25)
            }
        )
        seed_sequence = np.random.SeedSequence(
            [int(seed), 0 if str(layer) == "near" else 1]
        )
        run_seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            % np.iinfo(np.int32).max
            for child in seed_sequence.spawn(len(resolutions) * leiden_runs)
        ]
        consensus_run_memberships = []
        run_index = 0
        for local_resolution in resolutions:
            for replicate in range(1, leiden_runs + 1):
                run_seed = run_seeds[run_index]
                run_index += 1
                partition = la.find_partition(
                    graph,
                    la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=float(local_resolution),
                    n_iterations=-1,
                    seed=run_seed,
                )
                membership = np.asarray(partition.membership, dtype=int)
                if np.isclose(local_resolution, float(resolution)):
                    consensus_run_memberships.append(membership)
                modularity = (
                    float(
                        graph.modularity(
                            membership.tolist(),
                            weights="weight",
                        )
                    )
                    if graph.ecount()
                    else np.nan
                )
                diagnostics.append(
                    {
                        "analysis_type": layer,
                        "stage": "leiden_run",
                        "run": replicate,
                        "seed": run_seed,
                        "resolution": local_resolution,
                        "modularity": modularity,
                        "n_modules": int(np.unique(membership).size),
                        "n_nodes": graph.vcount(),
                        "n_edges": graph.ecount(),
                        "retained_modules": np.nan,
                        "assigned_nodes": np.nan,
                    }
                )

        membership_matrix = np.vstack(consensus_run_memberships)
        coassignment = np.mean(
            membership_matrix[:, :, None]
            == membership_matrix[:, None, :],
            axis=0,
        )
        consensus_edges = []
        consensus_weights = []
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                value = float(coassignment[left, right])
                if value >= consensus_threshold:
                    consensus_edges.append((left, right))
                    consensus_weights.append(value)
        consensus_graph = ig.Graph(
            n=len(nodes),
            edges=consensus_edges,
            directed=False,
        )
        consensus_graph.vs["name"] = nodes
        consensus_graph.es["weight"] = consensus_weights
        if consensus_graph.ecount():
            final_partition = la.find_partition(
                consensus_graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=1.0,
                n_iterations=-1,
                seed=int(seed),
            )
            final_membership = np.asarray(
                final_partition.membership, dtype=int
            )
            final_modularity = float(
                consensus_graph.modularity(
                    final_membership.tolist(),
                    weights="weight",
                )
            )
        else:
            final_membership = np.arange(len(nodes), dtype=int)
            final_modularity = np.nan

        candidate_communities = []
        for community in sorted(np.unique(final_membership)):
            indices = np.flatnonzero(final_membership == community)
            member_nodes = [nodes[index] for index in indices]
            omics = {node.split("::", 1)[0] for node in member_nodes}
            if len(indices) > 1:
                upper = coassignment[np.ix_(indices, indices)][
                    np.triu_indices(len(indices), k=1)
                ]
                module_stability = float(np.mean(upper))
            else:
                module_stability = 1.0
            if (
                len(indices) >= min_module_size
                and len(omics) >= 2
                and module_stability >= min_module_stability
            ):
                candidate_communities.append(
                    (indices, module_stability, len(omics))
                )
        candidate_communities.sort(
            key=lambda value: (
                -len(value[0]),
                [nodes[index] for index in value[0]],
            )
        )

        node_to_module = {}
        prefix = "N" if layer == "near" else "L"
        for index, (
            member_indices,
            module_stability,
            n_omics,
        ) in enumerate(candidate_communities, start=1):
            module_id = f"{prefix}{index:02d}"
            member_indices = np.asarray(member_indices, dtype=int)
            for node_index_value in member_indices:
                node = nodes[int(node_index_value)]
                omic, feature = node.split("::", 1)
                node_to_module[node] = module_id
                peers = member_indices[member_indices != node_index_value]
                node_stability = (
                    float(np.mean(coassignment[node_index_value, peers]))
                    if len(peers)
                    else 1.0
                )
                memberships.append(
                    {
                        "analysis_type": layer,
                        "module_id": module_id,
                        "node_id": node,
                        "omic": omic,
                        "feature": feature,
                        "module_size": len(member_indices),
                        "n_omics": n_omics,
                        "node_stability": node_stability,
                        "module_stability": module_stability,
                    }
                )
        diagnostics.append(
            {
                "analysis_type": layer,
                "stage": "consensus",
                "run": np.nan,
                "seed": int(seed),
                "resolution": 1.0,
                "modularity": final_modularity,
                "n_modules": int(np.unique(final_membership).size),
                "n_nodes": consensus_graph.vcount(),
                "n_edges": consensus_graph.ecount(),
                "retained_modules": len(candidate_communities),
                "assigned_nodes": len(node_to_module),
            }
        )
        for row in subset.itertuples(index=False):
            source = _node_id(str(row.x_omic), str(row.x_feature))
            target = _node_id(str(row.y_omic), str(row.y_feature))
            source_module = node_to_module.get(source, "unassigned")
            target_module = node_to_module.get(target, "unassigned")
            edge_assignments.append(
                {
                    "edge_id": row.edge_id,
                    "analysis_type": layer,
                    "source_module": source_module,
                    "target_module": target_module,
                    "module_id": (
                        source_module
                        if source_module == target_module
                        and source_module != "unassigned"
                        else "cross_module"
                    ),
                }
            )
    return (
        pd.DataFrame(memberships, columns=membership_columns),
        pd.DataFrame(edge_assignments, columns=assignment_columns),
        pd.DataFrame(diagnostics, columns=diagnostic_columns),
    )


def summarize_modules(
    discovery_edges: pd.DataFrame,
    memberships: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    joined = discovery_edges.merge(
        assignments[["edge_id", "module_id"]], on="edge_id", how="left"
    )
    rows = []
    for (layer, module_id), nodes in memberships.groupby(
        ["analysis_type", "module_id"], sort=True
    ):
        edges = joined[
            joined["analysis_type"].eq(layer) & joined["module_id"].eq(module_id)
        ]
        counts = Counter(nodes["omic"])
        classes = Counter(edges["edge_class"]) if len(edges) else Counter()
        rows.append(
            {
                "analysis_type": layer,
                "module_id": module_id,
                "n_nodes": int(len(nodes)),
                "n_MG": int(counts["MG"]),
                "n_MT": int(counts["MT"]),
                "n_MB": int(counts["MB"]),
                "n_internal_edges": int(len(edges)),
                "module_stability": float(nodes["module_stability"].iloc[0]),
                "mean_node_stability": float(nodes["node_stability"].mean()),
                "median_effect_reference": float(edges["effect_reference"].median())
                if len(edges)
                else np.nan,
                "median_effect_condition": float(edges["effect_condition"].median())
                if len(edges)
                else np.nan,
                "median_attenuation_score": float(
                    edges["attenuation_score"].median()
                )
                if len(edges)
                else np.nan,
                "weakened_fraction": classes["weakened_or_lost"] / max(len(edges), 1),
                "strengthened_fraction": classes["strengthened"] / max(len(edges), 1),
                "sign_reversed_fraction": classes["sign_reversed"]
                / max(len(edges), 1),
                "condition_emergent_fraction": classes["condition_emergent"]
                / max(len(edges), 1),
            }
        )
    return pd.DataFrame(rows)


def _split_terms(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {term.strip() for term in str(value).split("|") if term.strip()}


def module_enrichment(
    memberships: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    try:
        from scipy.stats import hypergeom
    except ImportError as exc:
        raise RuntimeError(
            "Module enrichment requires scipy; install project dependencies."
        ) from exc
    fields = {
        "MG": ["phylum", "family", "genus"],
        "MT": ["pathway_names", "module_names"],
        "MB": ["superclass", "class", "subclass", "direct_parent", "pathway_names"],
    }
    rows = []
    annotation_lookup = {
        (str(row["omic"]), str(row["feature"])): row.to_dict()
        for _, row in annotations.iterrows()
    }
    for layer, layer_nodes in memberships.groupby("analysis_type", sort=True):
        for omic, omic_nodes in layer_nodes.groupby("omic", sort=True):
            universe_features = sorted(
                set(
                    annotations.loc[
                        annotations["omic"].astype(str).eq(str(omic)), "feature"
                    ].astype(str)
                )
            )
            for field in fields.get(str(omic), []):
                feature_to_entity: dict[str, str] = {}
                universe_terms: dict[str, set[str]] = defaultdict(set)
                for feature in universe_features:
                    annotation = annotation_lookup.get((str(omic), feature))
                    entity = feature
                    if str(omic) == "MB" and annotation is not None:
                        primary = str(annotation.get("primary_hmdb_id") or "").strip()
                        if primary and primary.lower() != "nan":
                            entity = primary
                    feature_to_entity[feature] = entity
                    if annotation is None:
                        universe_terms[entity].update(set())
                    else:
                        universe_terms[entity].update(
                            _split_terms(annotation.get(field))
                        )
                term_totals = Counter(
                    term for terms in universe_terms.values() for term in terms
                )
                population = len(universe_terms)
                for module_id, module_nodes in omic_nodes.groupby(
                    "module_id", sort=True
                ):
                    module_features = sorted(
                        set(module_nodes["feature"].astype(str))
                    )
                    module_entities = {
                        feature_to_entity.get(feature, feature)
                        for feature in module_features
                    }
                    draws = len(module_entities)
                    module_counts = Counter(
                        term
                        for entity in module_entities
                        for term in universe_terms.get(entity, set())
                    )
                    for term, observed in module_counts.items():
                        total = term_totals[term]
                        pvalue = float(
                            hypergeom.sf(observed - 1, population, total, draws)
                        )
                        rows.append(
                            {
                                "analysis_type": layer,
                                "module_id": module_id,
                                "omic": omic,
                                "annotation_field": field,
                                "term": term,
                                "module_hits": observed,
                                "module_annotated_features": draws,
                                "universe_hits": total,
                                "universe_features": population,
                                "fold_enrichment": (
                                    (observed / draws) / (total / population)
                                    if draws and total
                                    else np.nan
                                ),
                                "pvalue": pvalue,
                            }
                        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["qvalue"] = np.nan
    for _, indices in result.groupby(
        ["analysis_type", "omic", "annotation_field"]
    ).groups.items():
        result.loc[indices, "qvalue"] = bh_fdr(result.loc[indices, "pvalue"])
    return result.sort_values(
        ["qvalue", "pvalue", "analysis_type", "module_id"], na_position="last"
    )


def _load_feature_universe(analysis_dir: Path) -> dict[str, list[str]]:
    result = {}
    for omic in ("MG", "MT", "MB"):
        path = analysis_dir / "01_prepared" / "matrices" / f"{omic}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Prepared matrix is missing: {path}")
        result[omic] = [str(value) for value in pd.read_parquet(path).columns]
    return result


def run_module_interpretation(
    analysis_dir: str | Path,
    output_dir: str | Path | None = None,
    network_dir: str | Path | None = None,
    reference_group: str = "nonIBD",
    condition_group: str = "IBD",
    q_column: str = "contrast_q",
    q_threshold: float = 0.05,
    near_zero_threshold: float = 0.10,
    resolution: float = 1.0,
    seed: int = 20260721,
    leiden_runs: int = 10,
    consensus_threshold: float = 0.55,
    min_module_size: int = 3,
    min_module_stability: float = 0.60,
    offline: bool = False,
    hmdb_xml: str | Path | None = None,
    ncbi_email: str | None = None,
) -> dict[str, Path]:
    analysis = Path(analysis_dir).resolve()
    network = (
        Path(network_dir).resolve()
        if network_dir is not None
        else analysis / "05_networks"
    )
    default_output_name = (
        "06_modules"
        if network.name == "05_networks"
        else network.name.replace("05_networks", "06_modules", 1)
    )
    output = ensure_dir(
        Path(output_dir).resolve()
        if output_dir is not None
        else analysis / default_output_name
    )
    annotation_dir = ensure_dir(output / "annotations")
    cache_dir = ensure_dir(output / "database_cache")
    master = build_edge_master(
        network,
        reference_group=reference_group,
        condition_group=condition_group,
        q_column=q_column,
    )
    master_path = output / "edge_master.tsv"
    master.to_csv(master_path, sep="\t", index=False)
    significant = pd.to_numeric(master["contrast_q"], errors="coerce").le(
        q_threshold
    )
    eligible = master["primary_network_eligible"].astype(bool)
    discovery = master[significant & eligible].copy()
    discovery = classify_edges(discovery, near_zero_threshold=near_zero_threshold)
    discovery_path = output / "edge_classification.tsv"
    discovery.to_csv(discovery_path, sep="\t", index=False)

    features = _load_feature_universe(analysis)
    kegg_client = CachedHttpClient(cache_dir / "kegg", offline=offline, requests_per_second=2.5)
    hmdb_client = CachedHttpClient(cache_dir / "hmdb", offline=offline, requests_per_second=1.0)
    ncbi_client = CachedHttpClient(cache_dir / "ncbi", offline=offline, requests_per_second=2.5)
    kegg_information = kegg_client.get_text(f"{DEFAULT_KEGG_BASE}/info/kegg")
    mt = annotate_ec_features(features["MT"], kegg_client)
    mb = annotate_hmdb_features(features["MB"], hmdb_client, bulk_xml=hmdb_xml)
    mg = annotate_taxonomy_features(
        features["MG"], ncbi_client, email=ncbi_email
    )
    annotation_paths = {}
    for name, frame in (("MG_taxonomy", mg), ("MT_KEGG", mt), ("MB_HMDB", mb)):
        path = annotation_dir / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        annotation_paths[name] = path
    annotations = pd.concat([mg, mt, mb], ignore_index=True, sort=False)
    annotations_path = annotation_dir / "feature_annotations.tsv"
    annotations.to_csv(annotations_path, sep="\t", index=False)
    annotation_qc = (
        annotations.groupby(["omic", "mapping_status"], dropna=False)
        .size()
        .rename("n_features")
        .reset_index()
    )
    annotation_qc["total_features"] = annotation_qc.groupby("omic")[
        "n_features"
    ].transform("sum")
    annotation_qc["fraction"] = (
        annotation_qc["n_features"] / annotation_qc["total_features"]
    )
    annotation_qc_path = annotation_dir / "annotation_qc.tsv"
    annotation_qc.to_csv(annotation_qc_path, sep="\t", index=False)
    guild_template = mg[["omic", "feature", "matched_name", "taxid"]].copy()
    for column in ("functional_guild", "evidence_type", "evidence_source", "evidence_id"):
        guild_template[column] = ""
    guild_template["confidence"] = "unassigned"
    guild_template_path = annotation_dir / "MG_functional_guild_template.tsv"
    guild_template.to_csv(guild_template_path, sep="\t", index=False)

    memberships, assignments, module_diagnostics = discover_modules(
        discovery,
        resolution=resolution,
        seed=seed,
        leiden_runs=leiden_runs,
        consensus_threshold=consensus_threshold,
        min_module_size=min_module_size,
        min_module_stability=min_module_stability,
    )
    membership_path = output / "module_membership.tsv"
    assignment_path = output / "edge_module_assignment.tsv"
    memberships.to_csv(membership_path, sep="\t", index=False)
    assignments.to_csv(assignment_path, sep="\t", index=False)
    module_diagnostics_path = output / "module_detection_runs.tsv"
    module_diagnostics.to_csv(
        module_diagnostics_path, sep="\t", index=False
    )
    summary = summarize_modules(discovery, memberships, assignments)
    summary_path = output / "module_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    enrichment = module_enrichment(memberships, annotations)
    enrichment_path = output / "module_enrichment.tsv"
    enrichment.to_csv(enrichment_path, sep="\t", index=False)

    qc = {
        "analysis_dir": str(analysis),
        "network_dir": str(network),
        "output_dir": str(output),
        "reference_group": reference_group,
        "condition_group": condition_group,
        "q_column": q_column,
        "q_threshold": q_threshold,
        "near_zero_threshold": near_zero_threshold,
        "module_algorithm": "consensus_leiden",
        "module_resolution_center": resolution,
        "module_resolutions": [
            0.75 * resolution,
            resolution,
            1.25 * resolution,
        ],
        "module_seed": seed,
        "leiden_runs_per_resolution": leiden_runs,
        "consensus_threshold": consensus_threshold,
        "min_module_size": min_module_size,
        "min_module_stability": min_module_stability,
        "offline": offline,
        "total_unique_edges": len(master),
        "discovery_edges": len(discovery),
        "orientation_unstable_near_edges": int(
            (
                master["analysis_type"].astype(str).eq("near")
                & ~master["sign_consistent"].astype(bool)
            ).sum()
        ),
        "significant_orientation_unstable_near_edges_excluded": int(
            (
                significant
                & master["analysis_type"].astype(str).eq("near")
                & ~master["sign_consistent"].astype(bool)
            ).sum()
        ),
        "discovery_edges_by_type": discovery["analysis_type"].value_counts().to_dict(),
        "features": {omic: len(values) for omic, values in features.items()},
        "annotation_mapping": {
            "MG": mg["mapping_status"].value_counts(dropna=False).to_dict(),
            "MT": mt["mapping_status"].value_counts(dropna=False).to_dict(),
            "MB": mb["mapping_status"].value_counts(dropna=False).to_dict(),
        },
        "hmdb_note": (
            "Parsed from user-supplied official HMDB XML."
            if hmdb_xml is not None
            else "Only accession normalization was performed; provide --hmdb-xml "
            "for HMDB chemical taxonomy because automated per-entry access is blocked."
        ),
        "modules_by_type": memberships.groupby("analysis_type")["module_id"]
        .nunique()
        .to_dict(),
        "database_sources": {
            "KEGG": DEFAULT_KEGG_BASE,
            "HMDB": str(hmdb_xml) if hmdb_xml is not None else DEFAULT_HMDB_BASE,
            "NCBI_Taxonomy": DEFAULT_NCBI_BASE,
        },
        "kegg_release_information": kegg_information.splitlines()[:5],
    }
    qc_path = output / "module_interpretation_qc.json"
    write_json(qc, qc_path)
    outputs = {
        "edge_master": master_path,
        "edge_classification": discovery_path,
        "feature_annotations": annotations_path,
        "annotation_qc": annotation_qc_path,
        "MG_functional_guild_template": guild_template_path,
        "module_membership": membership_path,
        "edge_module_assignment": assignment_path,
        "module_detection_runs": module_diagnostics_path,
        "module_summary": summary_path,
        "module_enrichment": enrichment_path,
        "qc": qc_path,
    }
    outputs.update({f"annotation_{key}": value for key, value in annotation_paths.items()})
    return outputs

"""
Define viral protease targets with their identifiers across different databases
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class ViralTarget:
    """Class to store viral protease target information"""
    name: str
    virus_name: str
    abbreviation: str
    uniprot_id: str
    chembl_id: str
    pdb_ids: List[str]
    pubchem_aid: List[int]  # PubChem Assay IDs
    bindingdb_target: str
    activity_threshold_nm: int  # Threshold for active/inactive classification
    description: str
    ec_number: Optional[str] = None
    organism: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class ViralTargetDatabase:
    """Database of viral protease targets"""

    def __init__(self):
        self.targets = self._initialize_targets()

    def _initialize_targets(self) -> Dict[str, ViralTarget]:
        """Initialize all viral protease targets with their database identifiers"""

        targets = {}

        # HIV-1 Protease
        targets['hiv1'] = ViralTarget(
            name="HIV-1 Protease",
            virus_name="Human Immunodeficiency Virus Type 1",
            abbreviation="HIV1-PR",
            uniprot_id="P03366",
            chembl_id="CHEMBL243",
            pdb_ids=["7BWJ", "1HXB", "1HXW", "2AQU", "3EKV", "4DJO", "5HVP"],
            pubchem_aid=[1706, 1903, 1945, 652038, 652105],
            bindingdb_target="HIV-1 protease",
            activity_threshold_nm=1000,
            description="Aspartic protease essential for viral maturation",
            ec_number="3.4.23.16",
            organism="Human immunodeficiency virus 1"
        )

        # HCV NS3/4A Protease
        targets['hcv'] = ViralTarget(
            name="HCV NS3/4A Protease",
            virus_name="Hepatitis C Virus",
            abbreviation="HCV-NS3/4A",
            uniprot_id="P26662",
            chembl_id="CHEMBL2094112",
            pdb_ids=["3SV6", "3M5L", "3KEE", "4A92", "4WF8", "5EPN"],
            pubchem_aid=[2302, 2575, 652042, 652109, 1322],
            bindingdb_target="Hepatitis C Virus NS3 Protease",
            activity_threshold_nm=1000,
            description="Serine protease critical for viral polyprotein processing",
            ec_number="3.4.21.98",
            organism="Hepatitis C virus genotype 1b"
        )

        # SARS-CoV-2 Main Protease
        targets['sars_cov2'] = ViralTarget(
            name="SARS-CoV-2 Main Protease",
            virus_name="Severe Acute Respiratory Syndrome Coronavirus 2",
            abbreviation="SARS-CoV-2-Mpro",
            uniprot_id="P0DTD1",
            chembl_id="CHEMBL4065109",
            pdb_ids=["7BQY", "6LU7", "6Y2E", "6Y2F", "6Y2G", "7JU7", "7K3T"],
            pubchem_aid=[1484, 1479, 1508, 1645, 1706, 1859],  # COVID-19 related assays
            bindingdb_target="SARS-CoV-2 3CLpro",
            activity_threshold_nm=1000,
            description="Cysteine protease essential for viral replication",
            ec_number="3.4.22.69",
            organism="Severe acute respiratory syndrome coronavirus 2"
        )

        # Dengue NS2B-NS3 Protease
        targets['dengue'] = ViralTarget(
            name="Dengue NS2B-NS3 Protease",
            virus_name="Dengue Virus",
            abbreviation="DENV-NS2B-NS3",
            uniprot_id="P29990",
            chembl_id="CHEMBL2366517",
            pdb_ids=["2FOM", "3U1I", "3L6P", "2M0R", "6MO0", "7CFL"],
            pubchem_aid=[588751, 652244, 720544, 1322241],
            bindingdb_target="Dengue virus NS3 protease",
            activity_threshold_nm=5000,  # Higher threshold due to weaker inhibitors
            description="Serine protease required for viral polyprotein processing",
            ec_number="3.4.21.91",
            organism="Dengue virus type 2"
        )

        # Zika Protease
        targets['zika'] = ViralTarget(
            name="Zika NS2B-NS3 Protease",
            virus_name="Zika Virus",
            abbreviation="ZIKV-NS2B-NS3",
            uniprot_id="Q32ZE1",
            chembl_id="CHEMBL4296421",
            pdb_ids=["5LC0", "5TFN", "5GJ4", "5H4I", "7M1V", "7VY9"],
            pubchem_aid=[1322257, 1322258, 1322259, 1159552],
            bindingdb_target="Zika virus NS3 protease",
            activity_threshold_nm=5000,
            description="Flavivirus serine protease essential for replication",
            ec_number="3.4.21.91",
            organism="Zika virus"
        )

        return targets

    def get_target(self, virus_key: str) -> ViralTarget:
        """Get a specific viral target"""
        if virus_key not in self.targets:
            raise ValueError(f"Unknown virus key: {virus_key}. Available: {list(self.targets.keys())}")
        return self.targets[virus_key]

    def get_all_targets(self) -> Dict[str, ViralTarget]:
        """Get all viral targets"""
        return self.targets

    def save_to_file(self, filepath: str = "configs/viral_targets.json"):
        """Save target information to JSON file"""
        data = {key: target.to_dict() for key, target in self.targets.items()}

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_chembl_ids(self) -> Dict[str, str]:
        """Get ChEMBL IDs for all targets"""
        return {key: target.chembl_id for key, target in self.targets.items()}

    def get_pdb_ids(self) -> Dict[str, List[str]]:
        """Get PDB IDs for all targets"""
        return {key: target.pdb_ids for key, target in self.targets.items()}

    def get_activity_thresholds(self) -> Dict[str, int]:
        """Get activity thresholds for all targets"""
        return {key: target.activity_threshold_nm for key, target in self.targets.items()}

    def print_summary(self):
        """Print a summary of all targets"""
        print("=" * 80)
        print("VIRAL PROTEASE TARGETS SUMMARY")
        print("=" * 80)

        for key, target in self.targets.items():
            print(f"\n{key.upper()} - {target.name}")
            print("-" * 40)
            print(f"  Virus: {target.virus_name}")
            print(f"  UniProt: {target.uniprot_id}")
            print(f"  ChEMBL: {target.chembl_id}")
            print(f"  PDB structures: {len(target.pdb_ids)} available")
            print(f"  PubChem assays: {len(target.pubchem_aid)} available")
            print(f"  Activity threshold: {target.activity_threshold_nm} nM")
            print(f"  Description: {target.description}")


def main():
    """Main function to initialize and save viral targets"""

    print("=" * 60)
    print("Defining Viral Protease Targets")
    print("=" * 60)

    # Initialize target database
    db = ViralTargetDatabase()

    # Print summary
    db.print_summary()

    # Save to configuration file
    print("\n" + "=" * 60)
    print("Saving target information...")

    db.save_to_file("configs/viral_targets.json")
    print("✓ Saved to configs/viral_targets.json")

    # Create target-specific YAML for easy access
    yaml_data = {
        'viral_targets': {
            key: {
                'name': target.name,
                'chembl_id': target.chembl_id,
                'uniprot_id': target.uniprot_id,
                'threshold_nm': target.activity_threshold_nm
            }
            for key, target in db.get_all_targets().items()
        }
    }

    with open("configs/targets.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print("✓ Saved to configs/targets.yaml")

    # Create a simple mapping file for quick reference
    mapping = {
        'chembl_ids': db.get_chembl_ids(),
        'activity_thresholds': db.get_activity_thresholds(),
        'pdb_structures': db.get_pdb_ids()
    }

    with open("configs/target_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)
    print("✓ Saved to configs/target_mapping.json")

    print("\n" + "=" * 60)
    print("Target definition complete!")
    print(f"Total targets defined: {len(db.get_all_targets())}")
    print("\nNext step: Run data_collection/04_chembl_downloader.py to download ChEMBL data")


if __name__ == "__main__":
    main()

## SQL Query from data to get virous data only

```sql
SELECT 
    mol.chembl_id AS compound_id, 
    cs.canonical_smiles AS smiles, 
    act.standard_value AS activity_value, 
    act.standard_type AS activity_type, 
    act.standard_units AS activity_unit, 
    tgt.pref_name AS target_name, 
    tgt.organism AS virus_name, 
    src.src_description AS source_database
FROM activities AS act
JOIN compound_structures AS cs ON act.molregno = cs.molregno
JOIN molecule_dictionary AS mol ON act.molregno = mol.molregno
JOIN assays AS a ON act.assay_id = a.assay_id
JOIN target_dictionary AS tgt ON a.tid = tgt.tid
JOIN organism_class AS oc ON tgt.tax_id = oc.tax_id
LEFT JOIN source AS src ON act.src_id = src.src_id
WHERE mol.chembl_id IS NOT NULL
  AND cs.canonical_smiles IS NOT NULL
  AND act.standard_value IS NOT NULL
  AND act.standard_type IS NOT NULL
  AND act.standard_units IS NOT NULL
  AND tgt.pref_name IS NOT NULL
  AND tgt.organism IS NOT NULL
  AND oc.l1 = 'Viruses'

```
# mysql query 

```sql
-- ============================================================================
-- BINDINGDB VIRUS DATA EXTRACTION - PERFECT QUERY
-- ============================================================================
-- Extracts drug activity data for VIRAL TARGETS ONLY
-- NO NULL values except source_database (always 'BindingDB')
-- Matches ChEMBL query structure
-- ============================================================================

SELECT 
    COALESCE(m.chembl_id, CONCAT('BDB_', m.monomerid)) AS compound_id,
    m.smiles_string AS smiles,
    c.affinity_value AS activity_value,
    c.affinity_type AS activity_type,
    CASE 
        WHEN c.affinity_type IN ('Ki', 'Kd', 'IC50', 'EC50') THEN 'nM'
        WHEN c.affinity_type = 'Inhibition' THEN '%'
        ELSE 'nM'
    END AS activity_unit,
    c.target_name AS target_name,
    CASE 
        WHEN c.source_organism LIKE '%HIV%' THEN 'HIV-1'
        WHEN c.source_organism LIKE '%SARS%CoV%2%' OR c.source_organism LIKE '%SARS-CoV-2%' THEN 'SARS-CoV-2'
        WHEN c.source_organism LIKE '%SARS%' AND c.source_organism NOT LIKE '%CoV%2%' THEN 'SARS-CoV'
        WHEN c.source_organism LIKE '%influenza%' OR c.source_organism LIKE '%Influenza%' THEN 'Influenza'
        WHEN c.source_organism LIKE '%hepatitis%C%' OR c.source_organism LIKE '%Hepatitis C%' OR c.source_organism LIKE '%HCV%' THEN 'HCV'
        WHEN c.source_organism LIKE '%hepatitis%B%' OR c.source_organism LIKE '%Hepatitis B%' OR c.source_organism LIKE '%HBV%' THEN 'HBV'
        WHEN c.source_organism LIKE '%Zika%' THEN 'Zika virus'
        WHEN c.source_organism LIKE '%Dengue%' THEN 'Dengue virus'
        WHEN c.source_organism LIKE '%Ebola%' THEN 'Ebola virus'
        WHEN c.source_organism LIKE '%herpes%' OR c.source_organism LIKE '%Herpes%' THEN 'Herpes virus'
        WHEN c.source_organism LIKE '%CMV%' OR c.source_organism LIKE '%cytomegalovirus%' THEN 'CMV'
        WHEN c.source_organism LIKE '%RSV%' OR c.source_organism LIKE '%respiratory syncytial%' THEN 'RSV'
        ELSE c.source_organism
    END AS virus_name,
    'BindingDB' AS source_database
FROM cobweb_bdb AS c
INNER JOIN monomer AS m ON c.monomer_id = m.monomerid
WHERE m.smiles_string IS NOT NULL 
    AND m.smiles_string != ''
    AND m.smiles_string NOT LIKE '%*%'  -- Remove invalid SMILES
    AND c.affinity_value IS NOT NULL
    AND c.affinity_value > 0
    AND c.affinity_type IS NOT NULL
    AND c.affinity_type != ''
    AND c.target_name IS NOT NULL
    AND c.target_name != ''
    AND c.source_organism IS NOT NULL
    AND c.source_organism != ''
    -- FILTER FOR VIRUSES ONLY
    AND (
        c.source_organism LIKE '%virus%'
        OR c.source_organism LIKE '%Virus%'
        OR c.source_organism LIKE '%viral%'
        OR c.source_organism LIKE '%Viral%'
        OR c.source_organism LIKE '%HIV%'
        OR c.source_organism LIKE '%SARS%'
        OR c.source_organism LIKE '%influenza%'
        OR c.source_organism LIKE '%Influenza%'
        OR c.source_organism LIKE '%hepatitis%'
        OR c.source_organism LIKE '%Hepatitis%'
        OR c.source_organism LIKE '%HCV%'
        OR c.source_organism LIKE '%HBV%'
        OR c.source_organism LIKE '%herpes%'
        OR c.source_organism LIKE '%Herpes%'
        OR c.source_organism LIKE '%CMV%'
        OR c.source_organism LIKE '%RSV%'
    )

UNION ALL

-- Extract from ki_result table (kinetic/inhibition data)
SELECT 
    COALESCE(m.chembl_id, CONCAT('BDB_', m.monomerid)) AS compound_id,
    m.smiles_string AS smiles,
    CASE 
        WHEN k.ic50 IS NOT NULL AND k.ic50 != '' THEN CAST(k.ic50 AS DECIMAL(20,4))
        WHEN k.ki IS NOT NULL AND k.ki != '' THEN CAST(k.ki AS DECIMAL(20,4))
        WHEN k.kd IS NOT NULL AND k.kd != '' THEN CAST(k.kd AS DECIMAL(20,4))
        WHEN k.ec50 IS NOT NULL AND k.ec50 != '' THEN CAST(k.ec50 AS DECIMAL(20,4))
    END AS activity_value,
    CASE 
        WHEN k.ic50 IS NOT NULL AND k.ic50 != '' THEN 'IC50'
        WHEN k.ki IS NOT NULL AND k.ki != '' THEN 'Ki'
        WHEN k.kd IS NOT NULL AND k.kd != '' THEN 'Kd'
        WHEN k.ec50 IS NOT NULL AND k.ec50 != '' THEN 'EC50'
    END AS activity_type,
    'nM' AS activity_unit,
    COALESCE(p.display_name, ers.enzyme) AS target_name,
    CASE 
        WHEN p.source_organism LIKE '%HIV%' THEN 'HIV-1'
        WHEN p.source_organism LIKE '%SARS%CoV%2%' OR p.source_organism LIKE '%SARS-CoV-2%' THEN 'SARS-CoV-2'
        WHEN p.source_organism LIKE '%SARS%' AND p.source_organism NOT LIKE '%CoV%2%' THEN 'SARS-CoV'
        WHEN p.source_organism LIKE '%influenza%' OR p.source_organism LIKE '%Influenza%' THEN 'Influenza'
        WHEN p.source_organism LIKE '%hepatitis%C%' OR p.source_organism LIKE '%Hepatitis C%' OR p.source_organism LIKE '%HCV%' THEN 'HCV'
        WHEN p.source_organism LIKE '%hepatitis%B%' OR p.source_organism LIKE '%Hepatitis B%' OR p.source_organism LIKE '%HBV%' THEN 'HBV'
        WHEN p.source_organism LIKE '%Zika%' THEN 'Zika virus'
        WHEN p.source_organism LIKE '%Dengue%' THEN 'Dengue virus'
        WHEN p.source_organism LIKE '%Ebola%' THEN 'Ebola virus'
        WHEN p.source_organism LIKE '%herpes%' OR p.source_organism LIKE '%Herpes%' THEN 'Herpes virus'
        WHEN p.source_organism LIKE '%CMV%' OR p.source_organism LIKE '%cytomegalovirus%' THEN 'CMV'
        WHEN p.source_organism LIKE '%RSV%' OR p.source_organism LIKE '%respiratory syncytial%' THEN 'RSV'
        ELSE p.source_organism
    END AS virus_name,
    'BindingDB' AS source_database
FROM ki_result AS k
INNER JOIN enzyme_reactant_set AS ers ON k.reactant_set_id = ers.reactant_set_id AND k.entryid = ers.entryid
INNER JOIN monomer AS m ON ers.inhibitor_monomerid = m.monomerid
INNER JOIN polymer AS p ON ers.enzyme_polymerid = p.polymerid
WHERE m.smiles_string IS NOT NULL 
    AND m.smiles_string != ''
    AND m.smiles_string NOT LIKE '%*%'  -- Remove invalid SMILES
    AND (
        (k.ic50 IS NOT NULL AND k.ic50 != '' AND CAST(k.ic50 AS DECIMAL(20,4)) > 0)
        OR (k.ki IS NOT NULL AND k.ki != '' AND CAST(k.ki AS DECIMAL(20,4)) > 0)
        OR (k.kd IS NOT NULL AND k.kd != '' AND CAST(k.kd AS DECIMAL(20,4)) > 0)
        OR (k.ec50 IS NOT NULL AND k.ec50 != '' AND CAST(k.ec50 AS DECIMAL(20,4)) > 0)
    )
    AND COALESCE(p.display_name, ers.enzyme) IS NOT NULL
    AND COALESCE(p.display_name, ers.enzyme) != ''
    AND p.source_organism IS NOT NULL
    AND p.source_organism != ''
    -- FILTER FOR VIRUSES ONLY
    AND (
        p.source_organism LIKE '%virus%'
        OR p.source_organism LIKE '%Virus%'
        OR p.source_organism LIKE '%viral%'
        OR p.source_organism LIKE '%Viral%'
        OR p.source_organism LIKE '%HIV%'
        OR p.source_organism LIKE '%SARS%'
        OR p.source_organism LIKE '%influenza%'
        OR p.source_organism LIKE '%Influenza%'
        OR p.source_organism LIKE '%hepatitis%'
        OR p.source_organism LIKE '%Hepatitis%'
        OR p.source_organism LIKE '%HCV%'
        OR p.source_organism LIKE '%HBV%'
        OR p.source_organism LIKE '%herpes%'
        OR p.source_organism LIKE '%Herpes%'
        OR p.source_organism LIKE '%CMV%'
        OR p.source_organism LIKE '%RSV%'
    )

UNION ALL

-- Extract from ITC results (biophysical data)
SELECT 
    COALESCE(m.chembl_id, CONCAT('BDB_', m.monomerid)) AS compound_id,
    m.smiles_string AS smiles,
    ABS(i.k) AS activity_value,
    'Ka' AS activity_type,
    'M^-1' AS activity_unit,
    p.display_name AS target_name,
    CASE 
        WHEN p.source_organism LIKE '%HIV%' THEN 'HIV-1'
        WHEN p.source_organism LIKE '%SARS%CoV%2%' OR p.source_organism LIKE '%SARS-CoV-2%' THEN 'SARS-CoV-2'
        WHEN p.source_organism LIKE '%SARS%' AND p.source_organism NOT LIKE '%CoV%2%' THEN 'SARS-CoV'
        WHEN p.source_organism LIKE '%influenza%' OR p.source_organism LIKE '%Influenza%' THEN 'Influenza'
        WHEN p.source_organism LIKE '%hepatitis%C%' OR p.source_organism LIKE '%Hepatitis C%' OR p.source_organism LIKE '%HCV%' THEN 'HCV'
        WHEN p.source_organism LIKE '%hepatitis%B%' OR p.source_organism LIKE '%Hepatitis B%' OR p.source_organism LIKE '%HBV%' THEN 'HBV'
        WHEN p.source_organism LIKE '%Zika%' THEN 'Zika virus'
        WHEN p.source_organism LIKE '%Dengue%' THEN 'Dengue virus'
        WHEN p.source_organism LIKE '%Ebola%' THEN 'Ebola virus'
        WHEN p.source_organism LIKE '%herpes%' OR p.source_organism LIKE '%Herpes%' THEN 'Herpes virus'
        WHEN p.source_organism LIKE '%CMV%' OR p.source_organism LIKE '%cytomegalovirus%' THEN 'CMV'
        WHEN p.source_organism LIKE '%RSV%' OR p.source_organism LIKE '%respiratory syncytial%' THEN 'RSV'
        ELSE p.source_organism
    END AS virus_name,
    'BindingDB' AS source_database
FROM itc_result_a_b_ab AS i
INNER JOIN monomer AS m ON (i.syr_monomerid = m.monomerid OR i.cell_monomerid = m.monomerid)
INNER JOIN polymer AS p ON (i.syr_polymerid = p.polymerid OR i.cell_polymerid = p.polymerid)
WHERE m.smiles_string IS NOT NULL 
    AND m.smiles_string != ''
    AND m.smiles_string NOT LIKE '%*%'  -- Remove invalid SMILES
    AND i.k IS NOT NULL 
    AND i.k != 0
    AND p.display_name IS NOT NULL
    AND p.display_name != ''
    AND p.source_organism IS NOT NULL
    AND p.source_organism != ''
    -- FILTER FOR VIRUSES ONLY
    AND (
        p.source_organism LIKE '%virus%'
        OR p.source_organism LIKE '%Virus%'
        OR p.source_organism LIKE '%viral%'
        OR p.source_organism LIKE '%Viral%'
        OR p.source_organism LIKE '%HIV%'
        OR p.source_organism LIKE '%SARS%'
        OR p.source_organism LIKE '%influenza%'
        OR p.source_organism LIKE '%Influenza%'
        OR p.source_organism LIKE '%hepatitis%'
        OR p.source_organism LIKE '%Hepatitis%'
        OR p.source_organism LIKE '%HCV%'
        OR p.source_organism LIKE '%HBV%'
        OR p.source_organism LIKE '%herpes%'
        OR p.source_organism LIKE '%Herpes%'
        OR p.source_organism LIKE '%CMV%'
        OR p.source_organism LIKE '%RSV%'
    )

ORDER BY compound_id, activity_value;
```
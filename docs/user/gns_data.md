# GNS data (upstream, particulate domain — not used by EQGNS)

The particulate-domain `.npz`/`metadata.json` format below belongs to the
upstream [geoelements/gns](https://github.com/geoelements/gns) particle
simulator (`gns/`), which this fork carries but does not use: EQGNS trains
and rolls out exclusively on the mesh-based domain (`meshnet/`). For the
format EQGNS actually reads and writes, see
[MeshNet data](flow_data.md) and [Data preparation](data_preparation.md).

If you need the particulate-domain format or sample datasets (`Sand`,
`SandRamps`, `WaterDropSample`), see the upstream repository's README and
[DesignSafe Data Depot](https://doi.org/10.17603/ds2-0phb-dg64).

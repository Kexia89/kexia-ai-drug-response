# Fuentes, créditos y política de datos

## Decisión de inclusión

Este repositorio **no incluye datos crudos, tablas procesadas, predicciones por fila ni modelos binarios entrenados sobre los ensayos**. La razón combina tamaño, reproducibilidad y respeto por términos upstream. Los scripts descargan los recursos y generan localmente `data/processed/` y los artefactos. Sólo se incluyen resultados agregados, manifiestos y auditorías.

| Fuente | Uso | Crédito / términos |
|---|---|---|
| Cell Model Passports / GDSC2 | dosis-respuesta, líneas, mutaciones, CNV, crecimiento, compuestos y RNA-seq | Genome Research Limited, CC BY-NC-ND 4.0; atribución y enlace requeridos; uso comercial requiere consentimiento |
| Sanger DepMap documentation | definiciones de ensayo y expresión | citar la documentación y publicaciones indicadas por el proveedor |
| ChEMBL | resolución de moléculas y estructuras exactas | EMBL-EBI ChEMBL, CC BY-SA 3.0; conservar identificadores, atribución y términos share-alike |
| Howard et al., PLoS Comput Biol (2021) | conteos longitudinales con doxorrubicina | citar DOI `10.1371/journal.pcbi.1009104` y el repositorio original; al verificarse esta entrega, el repositorio no mostraba una licencia explícita |

## Enlaces oficiales

- Cell Model Passports: <https://cellmodelpassports.sanger.ac.uk/downloads/gdsc>
- Política de uso Sanger DepMap: <https://depmap.sanger.ac.uk/documentation/data-usage-policy/>
- Sensibilidad farmacológica: <https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/>
- Expresión: <https://depmap.sanger.ac.uk/documentation/datasets/expression/>
- ChEMBL API: <https://www.ebi.ac.uk/chembl/api/data/docs>
- Artículo temporal: <https://doi.org/10.1371/journal.pcbi.1009104>
- Datos/código temporal: <https://github.com/brocklab/Drug-Response-Dynamics-Model-Calibration>

## Integridad

`data/metadata/download_manifest.json` conserva para cada recurso la URL descubierta, marca de tiempo upstream, tamaño esperado, SHA-256, fecha de descarga y nota de acceso/licencia. Las descargas son reanudables e idempotentes: un archivo completo del tamaño esperado se reutiliza; una ausencia se registra y nunca se reemplaza silenciosamente con datos sintéticos.

La política Sanger verificada el 23 de noviembre de 2026 identifica la base como CC BY-NC-ND 4.0, exige atribución a Genome Research Limited y prohíbe uso comercial sin consentimiento previo. Antes de publicar una copia con datos o modelos, revise nuevamente los términos vigentes y consulte al proveedor si el caso no es claro. El código MIT de este proyecto no relicencia contenido externo.


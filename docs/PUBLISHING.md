# Publicar esta carpeta en GitHub

Ejecute los comandos desde la carpeta `GitHub`. Reemplace `TU_USUARIO` y `TU_REPOSITORIO`.

## Opción A: repositorio creado previamente en github.com

```powershell
Set-Location "C:\ruta\a\kexia-cancer-research\GitHub"
git init
git add .
git commit -m "Initial reproducible research release"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

Si `origin` ya existe:

```powershell
git remote set-url origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

## Opción B: GitHub CLI

Tras instalar y autenticar `gh`:

```powershell
Set-Location "C:\ruta\a\kexia-cancer-research\GitHub"
git init
git add .
git commit -m "Initial reproducible research release"
git branch -M main
gh repo create TU_REPOSITORIO --public --source . --remote origin --push
```

Use `--private` en lugar de `--public` si aún desea revisar autores, propiedad intelectual o términos. Antes del primer push, ejecute `git status`, confirme que `data/raw/` y `data/processed/` sólo contienen `.gitkeep`, y revise los términos vigentes de las fuentes.

## Comprobaciones recomendadas

```powershell
python -m pytest
git status
git ls-files data/raw data/processed
git ls-files | ForEach-Object { Get-Item $_ } | Sort-Object Length -Descending | Select-Object -First 10 Name,Length
```

No hay archivos cercanos al límite de 100 MB de GitHub en esta entrega. Los datos y modelos binarios están excluidos deliberadamente.


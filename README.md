# Donut - Instrucciones de levantamiento

## Requisitos
- Windows 10/11
- Git
- Python 3.11 (recomendado)

## Si no tienes Python 3.11

### Opcion A: Instalar Python 3.11 (recomendado)
1. Descarga e instala Python 3.11 desde el sitio oficial.
2. Durante la instalacion, marca la opcion "Add Python to PATH".
3. Verifica la version:
   - `python --version`

### Opcion B: Usar pyenv-win (si ya lo usas)
1. Instala Python 3.11:
   - `pyenv install 3.11.9`
2. Activa la version en el proyecto:
   - `pyenv local 3.11.9`
3. Verifica la version:
   - `python --version`

### Opcion C: Usar Docker (si prefieres no instalar Python local)
1. Asegurate de tener Docker Desktop instalado.
2. Levanta el contenedor:
   - `docker compose up --build`

## Levantar el proyecto en local (con Python 3.11)
1. Crear y activar un entorno virtual:
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
2. Instalar dependencias:
   - `pip install -r requirements.txt`
3. Ejecutar la aplicacion:
   - `python app.py`

## Notas
- Si PowerShell bloquea la activacion del entorno virtual, ejecuta:
  - `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Si usas otro shell, adapta el comando de activacion del entorno virtual.

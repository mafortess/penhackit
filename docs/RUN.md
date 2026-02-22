# PenHACKit – Chuleta rápida

## 0️⃣ Crear entorno virtual (con versión correcta de Python)

⚠️ Usar Python 3.10 (recomendado para Torch + Transformers)

### Windows

```
py -3.10 -m venv venv
```

### Kali / Linux

```
python3.10 -m venv venv
```

## 1️⃣ Activar entorno virtual

### Windows (PowerShell)

```
.\venv\Scripts\Activate.ps1
```

### Kali / Linux

```
source venv/bin/activate
```

Verificar versión:

```
python --version
```

Debe decir: `Python 3.10.x`


---

## 2️⃣ Instalar dependencias (si es entorno nuevo)

Si no se hacen cambios:
```
pip install -r requirements.txt
```

Si se instalan nuevas dependencias:
```
pip install scikit-learn torch numpy pandas rich
```

Y actualizar el requirements.txt:
```
pip freeze > requirements.txt
```

---

## 3️⃣ Instalar el paquete (layout src)

Este comando crea un enlace simbólico entre tu código y el entorno de Python.

```
pip install -e .
```

⚠️ Necesario para que funcione:
```
python -m penhackit
```

---

## 4️⃣ Ejecutar

```
python -m penhackit
```

O si está registrado el script:

```
penhackit
```

---

## 5️⃣ Si algo falla (reset limpio)

Borrar entorno:

### Windows
```
rmdir /s /q venv
```

### Kali / Linux
```
rm -rf venv
```

Y repetir desde el paso 0.

---

## Flujo completo desde cero

```
py -3.10 -m venv venv        # o python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m penhackit
penhackit
```
# zrive-ds
Repositorio para el proyecto de predicción de user churn propuesto por MasMovil para el programa Zrive Applied Data Science (https://zriveapp.com/cursos/zrive-applied-data-science).

## Set up
Para asegurar consistencia utilizaremos todos **python version = 3.11.0**, además usaremos `pyenv` para la gestión de versiones de python y `poetry` para gestionar virtualenvs y dependencias. Estas tecnologías son una elección en cierta medida arbitraria, aunque las herramientas utilizadas son ampliamente utilizadas, hay otras opciones como `venv`, `virtualenv` o `pipenv` para virtualenvs, o conda para gestión de paquetes, que también se utilizan de forma extensa.

**Es importante no utilizar el python del sistema ya que acabaremos teniendo conflictos entre proyectos muy rápido. Por eso, utilizaremos siempre un virtualenv separado para cada proyecto.**

Para configurar nuestro entorno de trabajo primero instalamos python 3.11.0 en el ordenador usando pyenv `pyenv install 3.11.0`. 
Después, una vez estemos en el repositorio de trabajo (tras haber hecho un fork del repositorio original), nos aseguramos de que estamos utilizando ese python localmente `pyenv local 3.11.0`. Podemos comprobar que todo está correcto mediante el comando `pyenv versions`, que nos debería mostrar 3.11.0 con un asterisco que denota que es la versión utilizada en el directorio actual.

Una vez asegurada que la versión de python es correcta pasamos a instalar las dependencias iniciales definidas en `pyproject.toml` mediante el comando `poetry install`. Una vez instaladas, si necesitamos añadir una nueva dependencia podemos hacerlo con `poetry add <paquete>`.

Además, para garantizar la estandarización del código utilizamos:
1. `Black`: formatter no configurable que fuerza que el código tome el mismo formato en todos los proyectos.
2. `Flake8`: is a code linter. It warns you of syntax errors, possible bugs, stylistic errors, etc.

Finalmente, también tenemos instalado `mypy` para checkear tipos en nuestro código, sin embargo no lo tenemos en uso activo en `make test` ya que requiere cierta configuración. _Mypy is a static type checker for Python). Python is a dynamic language, so usually you'll only see errors in your code when you attempt to run it. Mypy is a static checker, so it finds bugs in your programs without even running them!_

## Repo structure
Utilizaremos la estructura por defecto que genera poetry al cual le añadimos unas carpetas de `notebooks`, `documentation` para separar esos ficheros. Además, aconsejamos crear una carpeta de `data` que será excluida de sincronizarse por estar añadida al `.gitignore` (sincronizar datos es normalmente mala idea ya que hace el repositorio muy pesado y lento.)


Utilizar `make lint` y `make test`, definidos en la Makefile para estandarizar el linting y asegurar que todos los tests corren. Esto facilitará también la revisión del código.
```
zrive-ds-4q23-churn
├── pyproject.toml
├── README.md
├── notebooks
├── src
│   ├── __init__.py
└── tests
    └── __init__.py
```

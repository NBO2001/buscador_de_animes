# Trabalho de RI


Baixe o dataset "anime.csv":

https://www.kaggle.com/datasets/svanoo/myanimelist-dataset?select=anime.csv


Depois mova pra a pasta:  charge_elastic

clone o backend e front:

```bash
git clone https://github.com/NBO2001/buscador_de_animes_backend.git ./backend
git clone https://github.com/NBO2001/-buscador_de_animes_frontend ./front
```


Remone tudo que é ".env.example" por ".env"


Depois execute o script para criar o index.

```bash
python3 charge.py
```

Ele depende das bibliotecas:

```bash
pip install python-dotenv
pip install elasticsearch
```
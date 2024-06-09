# Trabalho de RI

1. Baixe o dataset "anime.csv" do link abaixo:

   [Anime Dataset - Kaggle](https://www.kaggle.com/datasets/svanoo/myanimelist-dataset?select=anime.csv)

2. Mova o arquivo baixado para a pasta `charge_elastic`.

3. Clone os repositórios do backend e frontend:

   ```bash
   git clone https://github.com/NBO2001/buscador_de_animes_backend.git ./backend
   git clone https://github.com/NBO2001/-buscador_de_animes_frontend.git ./front
   ```

4. Renomeie todos os arquivos `.env.example` para `.env`.

5. Suba os containers Docker:

   ```bash
   docker-compose up
   ```

6. Execute o script para criar o índice, que está dentro da pasta `charge_elastic`:

   ```bash
   python3 charge.py
   ```

   Este script depende das seguintes bibliotecas:

   ```bash
   pip install python-dotenv
   pip install elasticsearch
   ```

Se tudo funcionou corretamente, você pode acessar os seguintes endereços:

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:4466](http://localhost:4466)
- Kibana: [http://localhost:5601](http://localhost:5601)
- Elasticsearch: [http://localhost:9200](http://localhost:9200)
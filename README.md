# 🐄 Sistema de Monitoramento Inteligente de Gado com Visão Computacional e IA

Este projeto utiliza inteligência artificial (YOLOv8) e rastreamento com SORT + Kalman Filter para **monitorar em tempo real** animais como gado, cavalos, porcos e outros, a partir de vídeos gravados ou câmera ao vivo.

---

## 🎯 Objetivo

Desenvolver uma solução robusta, replicável e extensível para:

- 📍 Monitoramento automatizado de animais em tempo real
- 📍 Rastreamento individual com IDs persistentes
- 📍 Aplicações em fazendas, agroindústrias e pesquisa zootécnica

---

## 📌 Funcionalidades

- ✅ Detecção com YOLOv8
- ✅ Rastreamento com filtro de Kalman (SORT)
- ✅ Suporte a vídeos `.mp4` ou webcam ao vivo
- ✅ Código modular, tipado e em conformidade com PEP 8
- ✅ Fácil integração com outras soluções (IoT, banco de dados, dashboards)

---

## 🧠 Tecnologias Utilizadas

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV (cv2)
- FilterPy
- NumPy
- Python 3.10+

---

## 📁 Estrutura do Projeto

```

projeto\_gado\_ai/
├── main.py                # Módulo principal (execução e interface)
├── modulocvdsa.py         # Módulo de rastreamento (SORT + Kalman)
├── data/
│   └── gado.mp4           # Vídeo de entrada de exemplo
├── outputs/
│   └── tracking\_log.csv   # Log de rastreamento (opcional)
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do projeto

````

---

## ▶️ Como Executar

### 🔧 1. Instalar dependências

```bash
pip install -r requirements.txt
````

📄 Arquivo `requirements.txt`:

```txt
numpy==1.24.3
opencv-python==4.9.0.80
ultralytics==8.1.1
filterpy==1.4.5
pygame==2.5.2
```

---

### 🎞️ 2. Execução com vídeo gravado

1. Coloque o vídeo `.mp4` na pasta `data/`, por exemplo: `data/gado.mp4`
2. No `main.py`, mantenha a linha:

```python
cap = cv2.VideoCapture("data/gado.mp4")
```

3. Rode o script:

```bash
python main.py
```

---

### 📷 3. Execução com câmera em tempo real (webcam)

1. No `main.py`, substitua a linha anterior por:

```python
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # largura
cap.set(4, 720)   # altura
```

2. Execute normalmente:

```bash
python main.py
```

🛑 Pressione `Q` na janela para encerrar o programa.

---

## 📸 Exemplos de Saída

### 🎥 Rastreamento em vídeo

> *(Substituir por um vídeo gravado da execução)*

![demo\_video](docs/demo_video.gif)

### 🐄 Detecção e ID no frame

> *(Substituir por um screenshot da execução)*

![frame\_resultado](docs/frame_resultado.jpg)

---

## 💾 Log de Rastreamento (opcional)

O sistema pode exportar um log `.csv` com:

* ID do animal
* Coordenadas da caixa (x1, y1, x2, y2)
* Frame/time

Local: `outputs/tracking_log.csv`

---

## 🚀 Possíveis Extensões

* Painel com Power BI ou Grafana
* Cerca virtual (geofencing inteligente)
* Banco de dados para controle de rebanho
* Integração com sensores ambientais (IoT)
* Interface Web (FastAPI, Streamlit)

---

## 👨‍💻 Autor

**Marcos Bruno**
Engenheiro Mecânico | Cientista de Visão Computacional
📍 Brasil 🇧🇷

---
## 📄 Licença

Este projeto é disponibilizado sob a **Licença MIT**.

Você pode:

- ✅ Usar, modificar e distribuir livremente
- ✅ Incluir em portfólios e apresentações
- ❌ **Sem garantias** de funcionamento para aplicações comerciais

Caso utilize em pesquisas, por favor, **cite o repositório ou o autor**.

Leia o arquivo [`LICENSE`](LICENSE) para mais detalhes.
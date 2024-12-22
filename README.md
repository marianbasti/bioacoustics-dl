# Deep Learning para Bioacústica

Este proyecto implementa un sistema de aprendizaje profundo para el análisis de datos bioacústicos utilizando el modelo BEATs (Bioacoustic Event Analysis and Training system).

## Características

- Entrenamiento auto-supervisado y supervisado de modelos BEATs
- Visualización interactiva de características aprendidas
- Soporte para entrenamiento distribuido con Accelerate
- Análisis de clustering y patrones temporales
- Memoria bank para contrastive learning mejorado
- Soporte para datos etiquetados y no etiquetados

## Instalación

```bash
git clone https://github.com/marianbasti/bioacoustics
cd bioacoustics
pip install -r requirements.txt
```

## Uso

### Entrenamiento

El entrenamiento puede realizarse en modo auto-supervisado o supervisado:

```bash
# Entrenamiento auto-supervisado completo
accelerate launch beats/train.py \
    --data_dir /ruta/a/audios \
    --model_path /ruta/a/checkpoint.pt \
    --positive_dir /ruta/a/positivos \
    --negative_dir /ruta/a/negativos \
    --epochs 10 \
    --checkpoint_freq 2 \
    --output_dir /ruta/salida/entrenamiento \
    --segment_duration 10

# Entrenamiento supervisado
accelerate launch beats/train.py \
    --data_dir /ruta/a/audios \
    --labeled_dir /ruta/a/datos_etiquetados \
    --model_path /ruta/a/checkpoint.pt \
    --supervised_weight 0.3 \
    --epochs 10 \
    --output_dir /ruta/salida/entrenamiento
```

### Visualización Interactiva

```bash
streamlit run beats/interactive_vis.py -- \
    --data_dir /ruta/a/audios \
    --checkpoint_path /ruta/a/checkpoint.pt \
    --max_samples 1000
```

## Estructura de Datos

El proyecto espera la siguiente estructura de directorios:

```
data/
├── unlabeled/           # Datos no etiquetados para entrenamiento auto-supervisado
│   ├── *.wav           # Archivos de audio en cualquier formato soportado
│   └── ...
├── labeled/            # Datos etiquetados (opcional)
│   ├── audio/         # Archivos de audio etiquetados
│   │   ├── *.wav
│   │   └── ...
│   └── labels.csv     # Archivo CSV con etiquetas
├── positive/          # Ejemplos positivos para contrastive learning (opcional)
│   ├── *.wav
│   └── ...
└── negative/          # Ejemplos negativos para contrastive learning (opcional)
    ├── *.wav
    └── ...
```

### Formato del archivo labels.csv

```csv
filename,labels
audio1.wav,"label1,label2"
audio2.wav,"label2,label3"
```

## Configuración

El entrenamiento puede configurarse mediante argumentos de línea de comandos:

### Parámetros de Entrenamiento
- `--data_dir`: Directorio con archivos de audio
- `--model_path`: Ruta al checkpoint pre-entrenado (opcional)
- `--epochs`: Número de épocas de entrenamiento
- `--batch_size`: Tamaño del batch
- `--lr`: Learning rate
- `--segment_duration`: Duración de segmentos de audio en segundos
- `--positive_dir`: Directorio con ejemplos positivos para contrastive learning
- `--negative_dir`: Directorio con ejemplos negativos para contrastive learning

### Parámetros de Modelo
- `--encoder_layers`: Número de capas del encoder
- `--encoder_embed_dim`: Dimensión de embeddings
- `--supervised_weight`: Peso para pérdida supervisada

### Parámetros de Visualización
- Perplexity (t-SNE): Balance entre estructura local y global
- N_neighbors (UMAP): Control de preservación de estructura
- Min_dist (UMAP): Control de agrupamiento de puntos

## Características Avanzadas

### Memory Bank
El sistema implementa un memory bank para mejorar el aprendizaje contrastivo:
- Tamaño configurable de banco de memoria
- Actualización continua durante entrenamiento
- Normalización de características

### Análisis de Características
La visualización interactiva incluye:
- Reducción de dimensionalidad (t-SNE y UMAP)
- Clustering K-means
- Análisis de componentes principales
- Visualización de patrones temporales
- Exploración interactiva de características

## Cita

Si utilizas este código en tu investigación, por favor cita:
```bibtex
@article{Chen2022beats,
  title = {BEATs: Audio Pre-Training with Acoustic Tokenizers},
  author  = {Sanyuan Chen and Yu Wu and Chengyi Wang and Shujie Liu and Daniel Tompkins and Zhuo Chen and Furu Wei},
  eprint={2212.09058},
  archivePrefix={arXiv},
  year={2022}
}
```
```
@article{chen2024eat,
  title={EAT: Self-Supervised Pre-Training with Efficient Audio Transformer},
  author={Chen, Wenxi and Liang, Yuzhe and Ma, Ziyang and Zheng, Zhisheng and Chen, Xie},
  journal={arXiv preprint arXiv:2401.03497},
  year={2024}
}
```
```bibtex
@misc{bioacoustics,
  author = {Marian Basti},
  title = {Deep Learning para Bioacústica},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/marianbasti/bioacoustics}
}
```

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.
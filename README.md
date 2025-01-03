# Deep Learning para Bioacústica

Este proyecto implementa un sistema de aprendizaje profundo para el análisis de datos bioacústicos utilizando el modelo BEATs (Bioacoustic Event Analysis and Training system) y EAT (Efficient Audio Transformer).

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

## Uso
## Entrenamiento del Modelo EAT

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

## Entrenamiento del Modelo EAT

### Preentrenamiento

El modelo EAT puede ser preentrenado en datos de audio no etiquetados utilizando modelado de audio enmascarado:

```bash
./scripts/pretrain.sh \
  --save_dir ./checkpoints/pretrain \
  --data_dir ./data/unlabeled \
  --cuda "0,1,2,3" \
  --batch_size 12 \
  --world_size 4 \
  --target_length 1024 \
  --mask_ratio 0.75 \
  --num_updates 400000 \
  --learning_rate 1.5e-4 \
  --update_freq 1
```

Principales parámetros de preentrenamiento:
- `--mask_ratio`: Proporción de audio de entrada a enmascarar (por defecto: 0.75)
- `--target_length`: Longitud fija de segmentos de audio (por defecto: 1024)
- `--world_size`: Número de GPUs para entrenamiento distribuido
- `--learning_rate`: Tasa de aprendizaje inicial
- `--num_updates`: Número total de actualizaciones de entrenamiento

### Finetuning

Después del preentrenamiento, podemos entrenar al modelo para que catogorice audios. Se le añade una layer final y se continúa entrenando con datos etiquetados.

```bash
./scripts/finetune.sh \
  --model_path ./checkpoints/pretrain/checkpoint_best.pt \
  --save_dir ./checkpoints/finetune \
  --data_dir ./data/raw/audio \
  --labels ./data/labeled/labels_descriptors.csv \
  --batch_size 96 \
  --target_length 1024 \
  --mixup 0.8 \
  --mask_ratio 0.2 \
```

Principales parámetros de ajuste fino:
- `--mixup`: Intensidad de la augmentación mixup (por defecto: 0.8)
- `--mask_ratio`: Proporción de audio de entrada a enmascarar (por defecto: 0.2)
- `--num_classes`: Número de clases de clasificación
- `--prediction_mode`: Cómo generar predicciones (CLS_TOKEN, MEAN_POOLING, o LIN_SOFTMAX)

El modelo soporta tres modos de predicción:
- `CLS_TOKEN`: Usar la incrustación del token [CLS] (por defecto)
- `MEAN_POOLING`: Promediar todas las incrustaciones de tokens
- `LIN_SOFTMAX`: Usar softmax lineal sobre todos los tokens

Características avanzadas durante el ajuste fino:
- SpecAugment: Augmentación de enmascaramiento de frecuencia y tiempo
- Mixup: Augmentación de mezcla de audio
- Decaimiento de la tasa de aprendizaje por capa
- Suavizado de etiquetas

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
@article{chen2024eat,
  title={EAT: Self-Supervised Pre-Training with Efficient Audio Transformer},
  author={Chen, Wenxi and Liang, Yuzhe and Ma, Ziyang and Zheng, Zhisheng and Chen, Xie},
  journal={arXiv preprint arXiv:2401.03497},
  year={2024}
}
``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
```bibtex
@article{Cañas2023dataset,
  title = {A dataset for benchmarking Neotropical anuran calls identification in passive acoustic monitoring},
  author = {Cañas, J.S. and Toro-Gómez, M.P. and Sugai, L.S.M. and others},
  publisher = {Nature Publishing Group},
  journal = {Sci Data},
  year = {2023},
  doi = {10.1038/s41597-023-02666-2}
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
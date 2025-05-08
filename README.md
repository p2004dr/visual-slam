# visual-slam


## Estructura del proyecto
visual-slam-monocular-video/
├── configs/
│ └── monocular.yaml # ruta vídeo, parámetros cámara, output map.ply
│
├── data/
│ └── dataset/ # vídeos de prueba y secuencias
│
├── src/
│ ├── orbslam2/
│ │ ├── init.py
│ │ ├── extractor.py
│ │ ├── matcher.py
│ │ ├── initializer.py
│ │ ├── tracker.py
│ │ ├── local_mapper.py
│ │ └── utils.py
│ │
│ └── run_video.py # importa así: from orbslam2.tracker import Tracker
│
├── tests/ # import orbslam2.extractor as extractor
│ ├── test_extractor.py
│ ├── test_tracker.py
│ └── test_pipeline.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py